from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime

from app.config import settings
from app.services.extraction_utils import (
    SlotCandidate,
    ThreadDecision,
    evaluate_thread_state,
    heuristic_extract,
)
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
    thread_state: str = "exploring"
    confidence_tier: str = "ambiguous"
    recommended_slot_key: str | None = None
    slot_candidates: list[SlotCandidate] | None = None
    decision_rationale: str | None = None
    should_generate: bool = False
    conflict_note: str | None = None


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
        existing_slots: list[SlotCandidate] | None = None,
    ) -> ExtractedCandidate:
        if not messages:
            return ExtractedCandidate(
                action="ignore",
                title="",
                confidence=0.1,
                timezone=settings.default_timezone,
                reason_summary="No messages in thread context.",
                evidence_message_ids=[],
                slot_candidates=[],
            )

        combined_text = "\n".join(str(m.get("text", "")) for m in messages if m.get("text"))
        llm_candidate = self.extract(combined_text, has_existing_thread_event)
        decision: ThreadDecision = evaluate_thread_state(
            messages=messages,
            default_timezone=settings.default_timezone,
            existing_slots=existing_slots,
        )
        recommended_slot = None
        if decision.recommended_slot_key:
            recommended_slot = next((s for s in decision.slot_candidates if s.slot_key == decision.recommended_slot_key), None)

        # High recall: if schedule signals exist, treat as create unless this clearly looks like an update.
        if recommended_slot and llm_candidate.action == "ignore":
            llm_candidate.action = "update" if has_existing_thread_event else "create"
            llm_candidate.confidence = max(llm_candidate.confidence, 0.6)

        if has_existing_thread_event and llm_candidate.action == "create":
            lowered = combined_text.lower()
            if any(token in lowered for token in ["move", "resched", "reschedule", "push", "delay", "instead"]):
                llm_candidate.action = "update"

        if not decision.should_generate:
            llm_candidate.action = "ignore"

        return ExtractedCandidate(
            action=llm_candidate.action,
            title=llm_candidate.title or "Meeting",
            confidence=max(llm_candidate.confidence, decision.decision_confidence),
            event_date=recommended_slot.event_date if recommended_slot else None,
            start_at=recommended_slot.start_at if recommended_slot else None,
            end_at=recommended_slot.end_at if recommended_slot else None,
            is_all_day=bool(recommended_slot.is_all_day) if recommended_slot else False,
            timezone=recommended_slot.timezone if recommended_slot else settings.default_timezone,
            reason_summary=decision.decision_rationale,
            evidence_message_ids=decision.evidence_message_ids,
            thread_state=decision.thread_state,
            confidence_tier=decision.confidence_tier,
            recommended_slot_key=decision.recommended_slot_key,
            slot_candidates=decision.slot_candidates,
            decision_rationale=decision.decision_rationale,
            should_generate=decision.should_generate,
            conflict_note=decision.conflict_note,
        )
