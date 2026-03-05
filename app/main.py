from __future__ import annotations

import json

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.db import Base, SessionLocal, engine, get_session
from app.models import AuditLog
from app.schemas import (
    GoogleOAuthCallbackRequest,
    GoogleOAuthStatusResponse,
    IngestRequest,
    IngestResponse,
    LlamaExtractRequest,
    LlamaExtractResponse,
    ProcessorRunResult,
    SlackActionRequest,
)
from app.scheduler import start_scheduler, stop_scheduler
from app.services.approvals import ApprovalService
from app.services.budget import BudgetService
from app.services.extraction_utils import heuristic_extract
from app.services.ingest import IngestService
from app.services.integrations import GoogleOAuthService, SlackActionParser, SlackParsedAction
from app.services.maintenance import MaintenanceService
from app.services.ollama_adapter import OllamaExtractorClient
from app.services.processor import ProcessorService

app = FastAPI(title="Scheduler Backend", version="0.1.0")


def _process_slack_action(parsed_action: SlackParsedAction) -> None:
    with SessionLocal() as session:
        try:
            svc = ApprovalService(session)
            status = svc.handle_action(parsed_action.suggestion_id, parsed_action.action, parsed_action.edited_title)
            session.add(
                AuditLog(
                    event_type="slack_action_processed",
                    payload=json.dumps(
                        {
                            "suggestion_id": parsed_action.suggestion_id,
                            "action": parsed_action.action,
                            "status": status,
                        }
                    ),
                )
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            session.add(
                AuditLog(
                    event_type="slack_action_failed",
                    payload=json.dumps(
                        {
                            "suggestion_id": parsed_action.suggestion_id,
                            "action": parsed_action.action,
                            "error": str(exc),
                        }
                    ),
                )
            )
            session.commit()


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    start_scheduler()


@app.on_event("shutdown")
def on_shutdown() -> None:
    stop_scheduler()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ingest/messages", response_model=IngestResponse)
def ingest_messages(payload: IngestRequest, session: Session = Depends(get_session)) -> IngestResponse:
    svc = IngestService(session)
    result = svc.ingest_batch(payload.batch_id, payload.messages)
    session.commit()
    return IngestResponse(accepted=True, deduped=result.deduped, queued_for_extraction=result.queued)


@app.post("/v1/processor/run", response_model=ProcessorRunResult)
def run_processor(session: Session = Depends(get_session)) -> ProcessorRunResult:
    svc = ProcessorService(session)
    result = svc.run_once()
    session.commit()
    return ProcessorRunResult(
        processed=result.processed,
        deferred=result.deferred,
        stopped_for_budget=result.stopped_for_budget,
    )


@app.post("/v1/llama/extract", response_model=LlamaExtractResponse)
def llama_extract(payload: LlamaExtractRequest) -> LlamaExtractResponse:
    allowed = [action for action in payload.allowed_actions if action in {"create", "update", "ignore"}]
    if not allowed:
        allowed = ["create", "update", "ignore"]

    client = OllamaExtractorClient()
    if client.configured():
        try:
            result = client.extract(
                text=payload.message_text,
                has_existing_thread_event=payload.has_existing_thread_event,
                allowed_actions=allowed,
            )
            return LlamaExtractResponse(
                action=str(result["action"]),
                title=str(result["title"]),
                confidence=float(result["confidence"]),
            )
        except Exception:
            pass

    action, title, confidence = heuristic_extract(
        payload.message_text,
        payload.has_existing_thread_event,
    )
    if action not in allowed:
        action = "ignore"
    return LlamaExtractResponse(action=action, title=title, confidence=confidence)


@app.post("/v1/slack/actions")
async def slack_actions(
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    x_slack_signature: str | None = Header(default=None),
    x_slack_request_timestamp: str | None = Header(default=None),
) -> dict[str, str]:
    parsed_action: SlackParsedAction | None = None
    raw = await request.body()

    content_type = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" in content_type:
        if not x_slack_signature or not x_slack_request_timestamp:
            raise HTTPException(status_code=400, detail="missing Slack signature headers")
        if not SlackActionParser.verify_signature(raw, x_slack_request_timestamp, x_slack_signature):
            raise HTTPException(status_code=401, detail="invalid Slack signature")

        parsed_action = SlackActionParser.parse_form_encoded_payload(raw)
        if parsed_action is None:
            raise HTTPException(status_code=400, detail="invalid Slack payload")
        background_tasks.add_task(_process_slack_action, parsed_action)
        return {"status": "accepted"}
    else:
        payload = SlackActionRequest.model_validate_json(raw)
        parsed_action = SlackParsedAction(
            suggestion_id=payload.suggestion_id,
            action=payload.action,
            edited_title=payload.edited_title,
        )

    try:
        svc = ApprovalService(session)
        status = svc.handle_action(parsed_action.suggestion_id, parsed_action.action, parsed_action.edited_title)
        session.commit()
        return {"status": status}
    except Exception as exc:
        session.rollback()
        session.add(
            AuditLog(
                event_type="slack_action_failed",
                payload=json.dumps(
                    {
                        "suggestion_id": parsed_action.suggestion_id,
                        "action": parsed_action.action,
                        "error": str(exc),
                    }
                ),
            )
        )
        session.commit()
        raise HTTPException(status_code=500, detail="failed to process Slack action") from exc


@app.get("/v1/google/oauth/start")
def google_oauth_start(session: Session = Depends(get_session)) -> dict[str, str]:
    svc = GoogleOAuthService(session)
    try:
        return {"auth_url": svc.auth_url()}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/google/oauth/callback")
def google_oauth_callback(
    payload: GoogleOAuthCallbackRequest,
    session: Session = Depends(get_session),
) -> GoogleOAuthStatusResponse:
    svc = GoogleOAuthService(session)
    try:
        token = svc.exchange_code(payload.code)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return GoogleOAuthStatusResponse(
        connected=True,
        has_refresh_token=bool(token.refresh_token),
        expiry=token.expiry,
    )


@app.get("/v1/google/oauth/callback")
def google_oauth_callback_get(
    code: str = Query(...),
    session: Session = Depends(get_session),
) -> GoogleOAuthStatusResponse:
    svc = GoogleOAuthService(session)
    try:
        token = svc.exchange_code(code)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return GoogleOAuthStatusResponse(
        connected=True,
        has_refresh_token=bool(token.refresh_token),
        expiry=token.expiry,
    )


@app.get("/v1/google/oauth/status", response_model=GoogleOAuthStatusResponse)
def google_oauth_status(session: Session = Depends(get_session)) -> GoogleOAuthStatusResponse:
    svc = GoogleOAuthService(session)
    status = svc.status()
    return GoogleOAuthStatusResponse(
        connected=bool(status["connected"]),
        has_refresh_token=bool(status["has_refresh_token"]),
        expiry=status["expiry"],  # type: ignore[arg-type]
    )


@app.post("/v1/budget/reset")
def budget_reset(session: Session = Depends(get_session)) -> dict[str, str]:
    svc = BudgetService(session)
    svc.manual_reset_current_month()
    session.commit()
    return {"status": "ok"}


@app.post("/v1/maintenance/purge-raw")
def purge_raw(session: Session = Depends(get_session)) -> dict[str, int]:
    svc = MaintenanceService(session)
    deleted = svc.purge_raw_messages()
    session.commit()
    return {"deleted": deleted}
