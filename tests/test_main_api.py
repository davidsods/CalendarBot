from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import AuditLog
from app.services.approvals import ApprovalService
from app.services.integrations import GoogleOAuthService


def _slack_signature(raw_body: bytes, timestamp: str) -> str:
    base = f"v0:{timestamp}:{raw_body.decode('utf-8')}".encode("utf-8")
    digest = hmac.new(settings.slack_signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return f"v0={digest}"


def test_slack_form_requires_signature_headers(client: TestClient) -> None:
    payload = {"actions": [{"action_id": "approve", "value": "42"}]}
    form_body = f"payload={json.dumps(payload)}".encode("utf-8")

    response = client.post(
        "/v1/slack/actions",
        content=form_body,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "missing Slack signature headers"


def test_slack_form_rejects_bad_signature(client: TestClient) -> None:
    settings.slack_signing_secret = "test-secret"
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    payload = {"actions": [{"action_id": "approve", "value": "42"}]}
    form_body = f"payload={json.dumps(payload)}".encode("utf-8")

    response = client.post(
        "/v1/slack/actions",
        content=form_body,
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "x-slack-request-timestamp": timestamp,
            "x-slack-signature": "v0=bad",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid Slack signature"


def test_slack_form_rejects_malformed_payload(client: TestClient) -> None:
    settings.slack_signing_secret = "test-secret"
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    payload = {"actions": [{"action_id": "approve", "value": "not-an-int"}]}
    form_body = f"payload={json.dumps(payload)}".encode("utf-8")

    response = client.post(
        "/v1/slack/actions",
        content=form_body,
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "x-slack-request-timestamp": timestamp,
            "x-slack-signature": _slack_signature(form_body, timestamp),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid Slack payload"


def test_slack_json_failure_returns_500_and_logs_audit(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(self: ApprovalService, suggestion_id: int, action: str, edited_title: str | None = None) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(ApprovalService, "handle_action", _raise)

    response = client.post(
        "/v1/slack/actions",
        json={
            "suggestion_id": 1,
            "action": "approve_create",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "failed to process Slack action"

    log = db_session.scalar(select(AuditLog).order_by(AuditLog.id.desc()))
    assert log is not None
    assert log.event_type == "slack_action_failed"


def test_google_oauth_start_returns_400_when_not_configured(client: TestClient) -> None:
    settings.google_client_id = None
    settings.google_redirect_uri = None

    response = client.get("/v1/google/oauth/start")

    assert response.status_code == 400


def test_google_oauth_callback_endpoints_return_400_on_exchange_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(self: GoogleOAuthService, code: str, user_id: str = "default"):
        raise RuntimeError("bad-code")

    monkeypatch.setattr(GoogleOAuthService, "exchange_code", _raise)

    post_response = client.post("/v1/google/oauth/callback", json={"code": "x"})
    get_response = client.get("/v1/google/oauth/callback", params={"code": "x"})

    assert post_response.status_code == 400
    assert get_response.status_code == 400


def test_google_oauth_status_disconnected_without_token(client: TestClient) -> None:
    response = client.get("/v1/google/oauth/status")

    assert response.status_code == 200
    assert response.json() == {
        "connected": False,
        "has_refresh_token": False,
        "expiry": None,
    }


def test_llama_extract_allowed_actions_filter_behavior(client: TestClient) -> None:
    settings.ollama_base_url = None

    restricted = client.post(
        "/v1/llama/extract",
        json={
            "message_text": "call tomorrow at 5pm",
            "has_existing_thread_event": False,
            "allowed_actions": ["update"],
        },
    )
    assert restricted.status_code == 200
    assert restricted.json()["action"] == "ignore"

    fallback = client.post(
        "/v1/llama/extract",
        json={
            "message_text": "call tomorrow at 5pm",
            "has_existing_thread_event": False,
            "allowed_actions": ["invalid-action"],
        },
    )
    assert fallback.status_code == 200
    assert fallback.json()["action"] == "create"


def test_processor_run_includes_window_skip_flag(client: TestClient) -> None:
    settings.processor_tz = "America/Los_Angeles"
    settings.processor_active_start_hour = 6
    settings.processor_active_end_hour = 24

    response = client.post("/v1/processor/run")
    assert response.status_code == 200
    body = response.json()
    assert "skipped_for_window" in body


def test_costs_summary_endpoint_shape(client: TestClient, db_session: Session) -> None:
    db_session.add(
        AuditLog(
            event_type="llama_invoked",
            payload=json.dumps({"thread_id": "t1"}),
        )
    )
    db_session.add(
        AuditLog(
            event_type="thread_processed",
            payload=json.dumps({"thread_id": "t1"}),
        )
    )
    db_session.commit()

    response = client.get("/v1/costs/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["lookback_days"] == 30
    assert "points" in body
    assert "totals" in body
    assert body["totals"]["model_invocations"] >= 1
