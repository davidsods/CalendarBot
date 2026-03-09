from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app.config import settings
from app.services.extraction_utils import (
    SlotCandidate,
    ThreadDecision,
    evaluate_thread_state,
    heuristic_extract,
)
from app.services.llama_decision_schema import LlamaThreadDecisionPayload
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
    slack_summary: str | None = None
    decision_source: str = "fallback"
    context_version: str = "v1"
    fallback_reason: str | None = None


class LlamaExtractor:
    def _heuristic(self, text: str, has_existing_thread_event: bool) -> ExtractedCandidate:
        action, title, confidence = heuristic_extract(text, has_existing_thread_event)
        return ExtractedCandidate(action=action, title=title, confidence=confidence)

    def extract(self, text: str, has_existing_thread_event: bool) -> ExtractedCandidate:
        if not settings.llama_extract_url:
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
                        decision_source="llama",
                        context_version=settings.llama_context_version,
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
                return ExtractedCandidate(
                    action=action,
                    title=title,
                    confidence=confidence,
                    decision_source="llama",
                    context_version=settings.llama_context_version,
                )
        except Exception:
            return self._heuristic(text, has_existing_thread_event)

    def extract_thread(
        self,
        messages: list[dict[str, object]],
        has_existing_thread_event: bool,
        reference_utc: datetime,
        existing_slots: list[SlotCandidate] | None = None,
        llama_context: dict[str, Any] | None = None,
    ) -> ExtractedCandidate:
        if not messages:
            return ExtractedCandidate(
                action="ignore",
                title="",
                confidence=0.1,
                timezone=settings.default_timezone,
                reason_summary="No thread context available.",
                evidence_message_ids=[],
                slot_candidates=[],
                decision_source="fallback",
                context_version=settings.llama_context_version,
                fallback_reason="no_thread_messages",
            )

        fallback_reason = "fallback_state_machine"
        if not llama_context:
            fallback_reason = "missing_llama_context"
        elif not bool(llama_context.get("context_ready")):
            fallback_reason = str(llama_context.get("fallback_reason") or "context_not_ready")

        llm_candidate = self._try_llama_thread_decision(
            llama_context=llama_context,
            has_existing_thread_event=has_existing_thread_event,
            existing_slots=existing_slots or [],
        )
        if llm_candidate is not None:
            return llm_candidate

        if fallback_reason == "fallback_state_machine":
            client = OllamaExtractorClient()
            if client.configured():
                fallback_reason = "llama_invalid_output_or_error"

        combined_text = "\n".join(str(m.get("text", "")) for m in messages if m.get("text"))
        llm_action_candidate = self.extract(combined_text, has_existing_thread_event)
        decision: ThreadDecision = evaluate_thread_state(
            messages=messages,
            default_timezone=settings.default_timezone,
            existing_slots=existing_slots,
        )
        recommended_slot = None
        if decision.recommended_slot_key:
            recommended_slot = next((s for s in decision.slot_candidates if s.slot_key == decision.recommended_slot_key), None)

        if recommended_slot and llm_action_candidate.action == "ignore":
            llm_action_candidate.action = "update" if has_existing_thread_event else "create"
            llm_action_candidate.confidence = max(llm_action_candidate.confidence, 0.6)

        if has_existing_thread_event and llm_action_candidate.action == "create":
            lowered = combined_text.lower()
            if any(token in lowered for token in ["move", "resched", "reschedule", "push", "delay", "instead"]):
                llm_action_candidate.action = "update"

        if not decision.should_generate:
            llm_action_candidate.action = "ignore"

        return ExtractedCandidate(
            action=llm_action_candidate.action,
            title=llm_action_candidate.title or "Meeting",
            confidence=max(llm_action_candidate.confidence, decision.decision_confidence),
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
            slack_summary=decision.decision_rationale,
            decision_source="fallback",
            context_version=settings.llama_context_version,
            fallback_reason=fallback_reason,
        )

    def _try_llama_thread_decision(
        self,
        llama_context: dict[str, Any] | None,
        has_existing_thread_event: bool,
        existing_slots: list[SlotCandidate],
    ) -> ExtractedCandidate | None:
        if not llama_context:
            return None
        if not bool(llama_context.get("context_ready")):
            return None

        client = OllamaExtractorClient()
        if not client.configured():
            return None

        try:
            raw = client.extract_thread_decision(llama_context)
            parsed = LlamaThreadDecisionPayload.model_validate(raw)
        except Exception:
            return None

        slot_candidates = [
            SlotCandidate(
                slot_key=slot.slot_key,
                event_date=slot.event_date,
                start_at=slot.start_at.replace(tzinfo=None) if slot.start_at and slot.start_at.tzinfo else slot.start_at,
                end_at=slot.end_at.replace(tzinfo=None) if slot.end_at and slot.end_at.tzinfo else slot.end_at,
                is_all_day=slot.is_all_day,
                timezone=slot.timezone,
                title=slot.title,
                proposer_message_id=slot.proposer_message_id,
                supporting_message_ids=slot.supporting_message_ids,
                contradicting_message_ids=slot.contradicting_message_ids,
                score=slot.score,
                recency_score=slot.recency_score,
                last_evidence_at=None,
            )
            for slot in parsed.slot_candidates
        ]
        recommended_slot = next((s for s in slot_candidates if s.slot_key == parsed.recommended_slot_key), None)

        action = parsed.action
        if parsed.should_generate and action == "ignore":
            action = "update" if has_existing_thread_event else "create"
        if not parsed.should_generate:
            action = "ignore"
        if has_existing_thread_event and action == "create":
            action = "update"

        return ExtractedCandidate(
            action=action,
            title=parsed.title or "Meeting",
            confidence=parsed.decision_confidence,
            event_date=recommended_slot.event_date if recommended_slot else parsed.event_date,
            start_at=recommended_slot.start_at if recommended_slot else parsed.start_at,
            end_at=recommended_slot.end_at if recommended_slot else parsed.end_at,
            is_all_day=bool(recommended_slot.is_all_day) if recommended_slot else parsed.is_all_day,
            timezone=(recommended_slot.timezone if recommended_slot else parsed.timezone) or settings.default_timezone,
            reason_summary=parsed.slack_summary or parsed.decision_rationale,
            evidence_message_ids=parsed.evidence_message_ids,
            thread_state=parsed.thread_state,
            confidence_tier=parsed.confidence_tier,
            recommended_slot_key=parsed.recommended_slot_key,
            slot_candidates=slot_candidates or existing_slots,
            decision_rationale=parsed.decision_rationale,
            should_generate=parsed.should_generate,
            conflict_note=parsed.conflict_note,
            slack_summary=parsed.slack_summary or parsed.decision_rationale,
            decision_source="llama",
            context_version=str(llama_context.get("context_version") or settings.llama_context_version),
        )
