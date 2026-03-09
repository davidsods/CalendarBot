from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime

from app.config import settings
from app.services.extraction_utils import heuristic_extract, parse_schedule, summarize_thread
from app.services.ollama_adapter import OllamaExtractorClient


@dataclass
class ExtractedCandidate:
    action: str
    title: str
    confidence: float
    event_date: date | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None
    is_all_day: bool = False
    timezone: str | None = None
    reason_summary: str | None = None
    evidence_message_ids: list[int] | None = None


class LlamaExtractor:
    def _heuristic(self, text: str, has_existing_thread_event: bool) -> ExtractedCandidate:
        action, title, confidence = heuristic_extract(text, has_existing_thread_event)
        return ExtractedCandidate(action=action, title=title, confidence=confidence)

    def extract(self, text: str, has_existing_thread_event: bool) -> ExtractedCandidate:
        if not settings.llama_extract_url:
            # Prefer direct Ollama integration when configured (common Railway setup).
            client = OllamaExtractorClient()
            if client.configured():
                try:
                    result = client.extract(
                        text=text,
                        has_existing_thread_event=has_existing_thread_event,
                        allowed_actions=["create", "update", "ignore"],
                    )
                    return ExtractedCandidate(
                        action=str(result["action"]),
                        title=str(result["title"]),
                        confidence=float(result["confidence"]),
                    )
                except Exception:
                    return self._heuristic(text, has_existing_thread_event)

        if not settings.llama_extract_url:
            return self._heuristic(text, has_existing_thread_event)

        payload = {
            "message_text": text,
            "has_existing_thread_event": has_existing_thread_event,
            "allowed_actions": ["create", "update", "ignore"],
        }

        headers = {"Content-Type": "application/json"}
        if settings.llama_api_key:
            headers["Authorization"] = f"Bearer {settings.llama_api_key}"

        req = urllib.request.Request(
            settings.llama_extract_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=settings.llama_timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                action = str(body.get("action", "ignore"))
                title = str(body.get("title", "Meeting"))
                confidence = float(body.get("confidence", 0.5))

                if action not in {"create", "update", "ignore"}:
                    return self._heuristic(text, has_existing_thread_event)
                return ExtractedCandidate(action=action, title=title, confidence=confidence)
        except Exception:
            return self._heuristic(text, has_existing_thread_event)

    def extract_thread(
        self,
        messages: list[dict[str, object]],
        has_existing_thread_event: bool,
        reference_utc: datetime,
    ) -> ExtractedCandidate:
        if not messages:
            return ExtractedCandidate(
                action="ignore",
                title="",
                confidence=0.1,
                timezone=settings.default_timezone,
                reason_summary="No messages in thread context.",
                evidence_message_ids=[],
            )

        combined_text = "\n".join(str(m.get("text", "")) for m in messages if m.get("text"))
        llm_candidate = self.extract(combined_text, has_existing_thread_event)
        parsed = parse_schedule(combined_text, reference_utc, settings.default_timezone)

        evidence = [int(m["id"]) for m in messages[-6:] if isinstance(m.get("id"), int)]
        summary = summarize_thread(messages)

        # High recall: if schedule signals exist, treat as create unless this clearly looks like an update.
        if parsed.event_date and llm_candidate.action == "ignore":
            llm_candidate.action = "update" if has_existing_thread_event else "create"
            llm_candidate.confidence = max(llm_candidate.confidence, 0.65)

        if has_existing_thread_event and llm_candidate.action == "create":
            lowered = combined_text.lower()
            if any(token in lowered for token in ["move", "resched", "reschedule", "push", "delay", "instead"]):
                llm_candidate.action = "update"

        return ExtractedCandidate(
            action=llm_candidate.action,
            title=llm_candidate.title or "Meeting",
            confidence=llm_candidate.confidence,
            event_date=parsed.event_date,
            start_at=parsed.start_at_utc,
            end_at=parsed.end_at_utc,
            is_all_day=parsed.is_all_day,
            timezone=parsed.timezone,
            reason_summary=summary,
            evidence_message_ids=evidence,
        )
