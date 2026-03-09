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
            if not self._has_minimum_decision_signal(raw):
                return None
            normalized = self._normalize_thread_decision_payload(raw, llama_context)
            parsed = LlamaThreadDecisionPayload.model_validate(normalized)
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

    def _normalize_thread_decision_payload(
        self,
        raw: dict[str, Any],
        llama_context: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(raw or {})
        default_tz = str(llama_context.get("default_timezone") or settings.default_timezone)

        out["thread_state"] = str(
            out.get("thread_state")
            or out.get("state")
            or out.get("conversation_state")
            or "exploring"
        ).strip().lower()
        state_aliases = {
            "pending": "candidate_slots",
            "consensus": "likely_consensus",
            "likely": "likely_consensus",
            "agreed": "confirmed",
            "reschedule": "reschedule_pending",
            "rescheduled": "reschedule_pending",
            "cancel": "canceled",
            "cancelled": "canceled",
        }
        out["thread_state"] = state_aliases.get(out["thread_state"], out["thread_state"])
        out["confidence_tier"] = str(
            out.get("confidence_tier")
            or out.get("confidence_label")
            or out.get("tier")
            or "ambiguous"
        ).strip().lower()
        tier_aliases = {"high": "likely", "medium": "ambiguous", "low": "conflicted", "uncertain": "ambiguous"}
        out["confidence_tier"] = tier_aliases.get(out["confidence_tier"], out["confidence_tier"])
        out["action"] = str(out.get("action") or out.get("decision_action") or "ignore").strip().lower()
        action_aliases = {
            "schedule": "create",
            "create_invite": "create",
            "create_event": "create",
            "modify": "update",
            "reschedule": "update",
            "none": "ignore",
            "no_action": "ignore",
        }
        out["action"] = action_aliases.get(out["action"], out["action"])
        out["decision_confidence"] = self._coerce_float(out.get("decision_confidence"), 0.0)
        out["should_generate"] = self._coerce_bool(out.get("should_generate"))
        out["title"] = str(out.get("title") or "Meeting").strip() or "Meeting"
        out["timezone"] = str(out.get("timezone") or default_tz).strip() or default_tz
        out["recommended_slot_key"] = out.get("recommended_slot_key") or out.get("recommended_slot") or out.get(
            "recommendedSlotKey"
        )
        out["decision_rationale"] = str(
            out.get("decision_rationale") or out.get("rationale") or out.get("why_now") or ""
        ).strip()
        out["slack_summary"] = str(out.get("slack_summary") or out.get("summary") or out["decision_rationale"]).strip() or None
        out["conflict_note"] = self._coerce_optional_str(out.get("conflict_note") or out.get("conflict"))
        out["is_all_day"] = self._coerce_bool(out.get("is_all_day"))
        out["event_date"] = out.get("event_date") or out.get("date")
        out["start_at"] = out.get("start_at") or out.get("start_time")
        out["end_at"] = out.get("end_at") or out.get("end_time")

        evidence_ids = out.get("evidence_message_ids")
        if evidence_ids is None:
            evidence_ids = out.get("evidence_ids") or out.get("evidenceMessageIds") or []
        out["evidence_message_ids"] = self._coerce_int_list(evidence_ids)

        slots_raw = out.get("slot_candidates")
        if slots_raw is None:
            slots_raw = out.get("slotCandidates") or out.get("candidates") or []
        normalized_slots: list[dict[str, Any]] = []
        if isinstance(slots_raw, list):
            for idx, slot_raw in enumerate(slots_raw):
                if not isinstance(slot_raw, dict):
                    continue
                slot = dict(slot_raw)
                slot["event_date"] = slot.get("event_date") or slot.get("date") or out["event_date"]
                slot["start_at"] = slot.get("start_at") or slot.get("start_time")
                slot["end_at"] = slot.get("end_at") or slot.get("end_time")
                slot["is_all_day"] = self._coerce_bool(slot.get("is_all_day"))
                slot["timezone"] = str(slot.get("timezone") or out["timezone"] or default_tz).strip() or default_tz
                slot["title"] = str(slot.get("title") or out["title"] or "Meeting").strip() or "Meeting"
                slot["supporting_message_ids"] = self._coerce_int_list(
                    slot.get("supporting_message_ids") or slot.get("support_ids")
                )
                slot["contradicting_message_ids"] = self._coerce_int_list(
                    slot.get("contradicting_message_ids") or slot.get("contradict_ids")
                )
                slot["score"] = self._coerce_float(slot.get("score"), 0.0)
                slot["recency_score"] = self._coerce_float(slot.get("recency_score"), 0.0)
                slot_key = slot.get("slot_key")
                if not slot_key:
                    slot_key = self._derive_slot_key(
                        event_date=slot.get("event_date"),
                        start_at=slot.get("start_at"),
                        end_at=slot.get("end_at"),
                        timezone_name=slot["timezone"],
                        index=idx,
                    )
                slot["slot_key"] = str(slot_key)
                normalized_slots.append(slot)

        if not normalized_slots and any([out["event_date"], out["start_at"], out["end_at"]]):
            synthetic_slot_key = self._derive_slot_key(
                event_date=out["event_date"],
                start_at=out["start_at"],
                end_at=out["end_at"],
                timezone_name=out["timezone"],
                index=0,
            )
            normalized_slots.append(
                {
                    "slot_key": synthetic_slot_key,
                    "event_date": out["event_date"],
                    "start_at": out["start_at"],
                    "end_at": out["end_at"],
                    "is_all_day": out["is_all_day"],
                    "timezone": out["timezone"],
                    "title": out["title"],
                    "supporting_message_ids": out["evidence_message_ids"],
                    "contradicting_message_ids": [],
                    "score": out["decision_confidence"],
                    "recency_score": 0.0,
                }
            )

        if not out["recommended_slot_key"] and normalized_slots:
            out["recommended_slot_key"] = normalized_slots[0]["slot_key"]
        out["slot_candidates"] = normalized_slots

        if not out["event_date"] and out["start_at"]:
            out["event_date"] = str(out["start_at"]).split("T", 1)[0]

        if out["should_generate"] and not out["event_date"]:
            out["should_generate"] = False
            out["action"] = "ignore"

        return out

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "likely"}
        return False

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int_list(value: Any) -> list[int]:
        if isinstance(value, list):
            out: list[int] = []
            for item in value:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    continue
            return sorted(set(out))
        return []

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _derive_slot_key(
        event_date: Any,
        start_at: Any,
        end_at: Any,
        timezone_name: str,
        index: int,
    ) -> str:
        return f"{event_date or ''}|{start_at or ''}|{end_at or ''}|{timezone_name}|{index}"

    @staticmethod
    def _has_minimum_decision_signal(raw: Any) -> bool:
        if not isinstance(raw, dict):
            return False
        signal_keys = {
            "should_generate",
            "action",
            "decision_action",
            "thread_state",
            "state",
            "recommended_slot_key",
            "recommended_slot",
            "recommendedSlotKey",
            "slot_candidates",
            "slotCandidates",
            "event_date",
            "date",
            "start_at",
            "start_time",
        }
        return any(key in raw for key in signal_keys)
