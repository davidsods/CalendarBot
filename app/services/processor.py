from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from sqlalchemy import asc, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    AuditLog,
    CalendarLink,
    EventSuggestion,
    MessageRecord,
    ProcessingCheckpoint,
    QueueItem,
    QueueStatus,
    SuggestionStatus,
    ThreadPlanningState,
    ThreadSlotCandidate,
)
from app.services.budget import BudgetService
from app.services.extractor import ExtractedCandidate, LlamaExtractor
from app.services.extraction_utils import SlotCandidate, classify_signal, parse_schedule
from app.services.integrations import SlackNotifier


@dataclass
class RunResult:
    processed: int
    deferred: int
    stopped_for_budget: bool


class ProcessorService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.budget = BudgetService(session)
        self.extractor = LlamaExtractor()
        self.slack = SlackNotifier()

    def _checkpoint(self) -> ProcessingCheckpoint:
        checkpoint = self.session.get(ProcessingCheckpoint, 1)
        if checkpoint:
            return checkpoint
        checkpoint = ProcessingCheckpoint(id=1, last_successful_processed_at=None)
        self.session.add(checkpoint)
        self.session.flush()
        return checkpoint

    def _next_pending(self) -> QueueItem | None:
        stmt = (
            select(QueueItem)
            .join(MessageRecord, QueueItem.message_id == MessageRecord.id)
            .where(QueueItem.status == QueueStatus.pending)
            .order_by(asc(MessageRecord.sent_at), asc(QueueItem.id))
            .limit(1)
        )
        return self.session.scalar(stmt)

    def _load_thread_messages(self, thread_id: str, upto: datetime) -> list[dict[str, object]]:
        stmt = (
            select(MessageRecord)
            .where(MessageRecord.thread_id == thread_id, MessageRecord.sent_at <= upto)
            .order_by(asc(MessageRecord.sent_at), asc(MessageRecord.id))
        )
        rows = self.session.scalars(stmt).all()
        if settings.thread_context_window_messages > 0:
            rows = rows[-settings.thread_context_window_messages :]
        return [
            {
                "id": m.id,
                "sender_role": m.sender_role,
                "text": m.text,
                "sent_at": m.sent_at,
            }
            for m in rows
        ]

    def _load_all_thread_messages(self, thread_id: str, upto: datetime) -> list[MessageRecord]:
        stmt = (
            select(MessageRecord)
            .where(MessageRecord.thread_id == thread_id, MessageRecord.sent_at <= upto)
            .order_by(asc(MessageRecord.sent_at), asc(MessageRecord.id))
        )
        return self.session.scalars(stmt).all()

    def build_llama_thread_context(
        self,
        thread_id: str,
        upto_message: MessageRecord,
        has_existing_thread_event: bool,
        target_event_ref: str | None,
        existing_slots: list[SlotCandidate],
    ) -> dict[str, object]:
        all_rows = self._load_all_thread_messages(thread_id, upto_message.sent_at)
        if not all_rows:
            return {
                "context_ready": False,
                "fallback_reason": "no_thread_messages",
                "context_version": settings.llama_context_version,
            }

        recent_n = max(settings.thread_context_window_messages, 1)
        recent_rows = all_rows[-recent_n:]
        older_rows = all_rows[:-recent_n]

        def _row_to_payload(row: MessageRecord) -> dict[str, object]:
            return {
                "id": row.id,
                "sender_role": row.sender_role,
                "text": row.text,
                "sent_at": row.sent_at,
            }

        anchor_messages: list[dict[str, object]] = []
        for row in reversed(older_rows):
            parsed = parse_schedule(row.text, row.sent_at, settings.default_timezone)
            signal = classify_signal(row.text, parsed)
            if signal.signal_type in {"proposal", "accept", "reject", "reschedule", "cancel"}:
                anchor_messages.append(_row_to_payload(row))
            if len(anchor_messages) >= max(settings.thread_anchor_window_messages, 1):
                break
        anchor_messages.reverse()

        planning = self.session.scalar(select(ThreadPlanningState).where(ThreadPlanningState.thread_id == thread_id))
        planning_state = {
            "thread_state": planning.state if planning else "exploring",
            "recommended_slot_key": planning.recommended_slot_key if planning else None,
            "confidence_tier": None,
            "decision_confidence": float(planning.decision_confidence) if planning else 0.0,
            "decision_rationale": planning.decision_rationale if planning else None,
            "decision_source": planning.decision_source if planning else None,
        }

        slot_snapshot = []
        for slot in sorted(existing_slots, key=lambda s: (s.score + 0.2 * s.recency_score), reverse=True)[
            : max(settings.thread_slot_snapshot_limit, 1)
        ]:
            slot_snapshot.append(
                {
                    "slot_key": slot.slot_key,
                    "event_date": slot.event_date,
                    "start_at": slot.start_at,
                    "end_at": slot.end_at,
                    "is_all_day": slot.is_all_day,
                    "timezone": slot.timezone,
                    "title": slot.title,
                    "supporting_message_ids": slot.supporting_message_ids,
                    "contradicting_message_ids": slot.contradicting_message_ids,
                    "score": slot.score,
                    "recency_score": slot.recency_score,
                }
            )

        context = {
            "context_ready": True,
            "context_version": settings.llama_context_version,
            "thread_id": thread_id,
            "thread_messages_recent": [_row_to_payload(row) for row in recent_rows],
            "thread_messages_anchor": anchor_messages,
            "has_existing_thread_event": has_existing_thread_event,
            "target_event_ref": target_event_ref,
            "prior_planning_state": planning_state,
            "slot_candidates_snapshot": slot_snapshot,
            "default_timezone": settings.default_timezone,
            "now_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        required_ok = bool(context["thread_messages_recent"]) and bool(context["default_timezone"])
        if not required_ok:
            context["context_ready"] = False
            context["fallback_reason"] = "missing_required_context_fields"
        return context

    def _load_existing_slots(self, thread_id: str) -> list[SlotCandidate]:
        stmt = (
            select(ThreadSlotCandidate)
            .where(ThreadSlotCandidate.thread_id == thread_id, ThreadSlotCandidate.status == "active")
            .order_by(asc(ThreadSlotCandidate.updated_at), asc(ThreadSlotCandidate.id))
        )
        rows = self.session.scalars(stmt).all()
        slots: list[SlotCandidate] = []
        for row in rows:
            supporting = []
            contradicting = []
            try:
                supporting = json.loads(row.supporting_message_ids_json or "[]")
            except Exception:
                supporting = []
            try:
                contradicting = json.loads(row.contradicting_message_ids_json or "[]")
            except Exception:
                contradicting = []
            slots.append(
                SlotCandidate(
                    slot_key=row.slot_key,
                    event_date=row.event_date,
                    start_at=row.start_at,
                    end_at=row.end_at,
                    is_all_day=bool(row.is_all_day),
                    timezone=row.timezone or settings.default_timezone,
                    title=row.title,
                    proposer_message_id=row.proposer_message_id,
                    supporting_message_ids=[int(x) for x in supporting if isinstance(x, int)],
                    contradicting_message_ids=[int(x) for x in contradicting if isinstance(x, int)],
                    score=row.score,
                    recency_score=row.recency_score,
                    last_evidence_at=row.last_evidence_at,
                )
            )
        return slots

    def _persist_thread_decision(self, thread_id: str, candidate: ExtractedCandidate) -> int | None:
        planning = self.session.scalar(select(ThreadPlanningState).where(ThreadPlanningState.thread_id == thread_id))
        if planning is None:
            planning = ThreadPlanningState(thread_id=thread_id)
            self.session.add(planning)
            self.session.flush()

        planning.state = candidate.thread_state
        planning.summary = candidate.reason_summary
        planning.recommended_slot_key = candidate.recommended_slot_key
        planning.decision_confidence = candidate.confidence
        planning.decision_rationale = candidate.decision_rationale
        planning.decision_source = candidate.decision_source
        planning.context_version = candidate.context_version

        active_rows = self.session.scalars(
            select(ThreadSlotCandidate).where(
                ThreadSlotCandidate.thread_id == thread_id,
                ThreadSlotCandidate.status == "active",
            )
        ).all()
        active_by_key = {row.slot_key: row for row in active_rows}
        candidate_by_key = {slot.slot_key: slot for slot in (candidate.slot_candidates or [])}

        for key, row in active_by_key.items():
            if key not in candidate_by_key:
                row.status = "superseded"

        selected_id: int | None = None
        for key, slot in candidate_by_key.items():
            row = active_by_key.get(key)
            if row is None:
                row = ThreadSlotCandidate(thread_id=thread_id, slot_key=key, status="active")
                self.session.add(row)
                self.session.flush()
            row.event_date = slot.event_date
            row.start_at = slot.start_at
            row.end_at = slot.end_at
            row.is_all_day = slot.is_all_day
            row.timezone = slot.timezone
            row.title = slot.title
            row.proposer_message_id = slot.proposer_message_id
            row.supporting_message_ids_json = json.dumps(sorted(set(slot.supporting_message_ids)))
            row.contradicting_message_ids_json = json.dumps(sorted(set(slot.contradicting_message_ids)))
            row.score = slot.score
            row.recency_score = slot.recency_score
            row.last_evidence_at = slot.last_evidence_at
            row.status = "active"
            row.version = max(row.version, 1)
            if key == candidate.recommended_slot_key:
                selected_id = row.id

        self.session.flush()
        return selected_id

    def _upsert_pending_suggestion(
        self,
        msg: MessageRecord,
        existing_link: CalendarLink | None,
        candidate: ExtractedCandidate,
        slot_candidate_id: int | None,
        context_window_size: int,
    ) -> tuple[EventSuggestion, bool]:
        pending_stmt = select(EventSuggestion).where(
            EventSuggestion.thread_id == msg.thread_id,
            EventSuggestion.status == SuggestionStatus.pending_approval,
        )
        existing_pending = self.session.scalar(pending_stmt)

        action = "update" if existing_link else "create"
        if candidate.action in {"create", "update"}:
            action = candidate.action

        is_new = existing_pending is None
        suggestion = existing_pending or EventSuggestion(message_id=msg.id, thread_id=msg.thread_id)
        suggestion.message_id = msg.id
        suggestion.action = action
        suggestion.target_event_ref = existing_link.google_event_id if existing_link else None
        suggestion.title = candidate.title or "Meeting"
        suggestion.start_at = candidate.start_at
        suggestion.end_at = candidate.end_at
        suggestion.event_date = candidate.event_date
        suggestion.is_all_day = bool(candidate.is_all_day)
        suggestion.timezone = candidate.timezone or settings.default_timezone
        suggestion.confidence = candidate.confidence
        suggestion.reason_summary = candidate.reason_summary
        suggestion.evidence_message_ids_json = json.dumps(candidate.evidence_message_ids or [])
        suggestion.context_window_size = context_window_size
        suggestion.thread_state = candidate.thread_state
        suggestion.confidence_tier = candidate.confidence_tier
        suggestion.slot_candidate_id = slot_candidate_id
        suggestion.decision_source = candidate.decision_source
        suggestion.context_version = candidate.context_version
        if existing_pending is None:
            self.session.add(suggestion)
        self.session.flush()
        return suggestion, is_new

    def _process_one(self, item: QueueItem) -> None:
        item.status = QueueStatus.processing
        item.attempts += 1
        self.session.flush()

        msg = self.session.get(MessageRecord, item.message_id)
        assert msg is not None

        existing_link = self.session.scalar(select(CalendarLink).where(CalendarLink.thread_id == msg.thread_id))

        thread_messages = self._load_thread_messages(msg.thread_id, msg.sent_at)
        existing_slots = self._load_existing_slots(msg.thread_id)
        llama_context = self.build_llama_thread_context(
            thread_id=msg.thread_id,
            upto_message=msg,
            has_existing_thread_event=bool(existing_link),
            target_event_ref=existing_link.google_event_id if existing_link else None,
            existing_slots=existing_slots,
        )
        candidate = self.extractor.extract_thread(
            thread_messages,
            has_existing_thread_event=bool(existing_link),
            reference_utc=msg.sent_at,
            existing_slots=existing_slots,
            llama_context=llama_context,
        )
        slot_candidate_id = self._persist_thread_decision(msg.thread_id, candidate)

        if candidate.fallback_reason:
            self.session.add(
                AuditLog(
                    event_type="llama_fallback",
                    payload=json.dumps(
                        {
                            "thread_id": msg.thread_id,
                            "message_id": msg.id,
                            "reason": candidate.fallback_reason,
                            "context_version": candidate.context_version,
                        }
                    ),
                )
            )

        if candidate.action == "ignore" or not candidate.should_generate:
            item.status = QueueStatus.completed
            self.budget.record_usage(settings.estimated_llama_cost_per_message_usd)
            self.session.flush()
            return

        suggestion, is_new = self._upsert_pending_suggestion(
            msg,
            existing_link,
            candidate,
            slot_candidate_id,
            context_window_size=len(thread_messages),
        )

        evidence_msgs: list[dict[str, object]] = []
        if candidate.evidence_message_ids:
            evidence_stmt = (
                select(MessageRecord)
                .where(MessageRecord.id.in_(candidate.evidence_message_ids))
                .order_by(asc(MessageRecord.sent_at), asc(MessageRecord.id))
            )
            evidence_rows = self.session.scalars(evidence_stmt).all()
            evidence_msgs = [
                {"sender_role": row.sender_role, "text": row.text, "sent_at": row.sent_at}
                for row in evidence_rows
            ]

        if is_new:
            try:
                self.slack.send_suggestion(
                    suggestion_id=suggestion.id,
                    action=suggestion.action,
                    title=suggestion.title,
                    thread_id=suggestion.thread_id,
                    source_text=msg.text,
                    sent_at=msg.sent_at,
                    start_at=suggestion.start_at,
                    end_at=suggestion.end_at,
                    event_date=suggestion.event_date,
                    is_all_day=suggestion.is_all_day,
                    timezone_name=suggestion.timezone,
                    target_event_ref=suggestion.target_event_ref,
                    confidence=suggestion.confidence,
                    thread_summary=suggestion.reason_summary,
                    evidence_messages=evidence_msgs,
                    confidence_tier=suggestion.confidence_tier,
                    decision_rationale=suggestion.reason_summary,
                    conflict_note=candidate.conflict_note,
                    decision_source=suggestion.decision_source,
                )
            except Exception as exc:
                self.session.add(
                    AuditLog(
                        event_type="slack_suggestion_failed",
                        payload=json.dumps(
                            {
                                "suggestion_id": suggestion.id,
                                "action": suggestion.action,
                                "error": str(exc),
                            }
                        ),
                    )
                )

        item.status = QueueStatus.completed
        self.budget.record_usage(settings.estimated_llama_cost_per_message_usd)
        self.session.flush()

    def run_once(self) -> RunResult:
        self.budget.refresh_month_and_requeue_if_needed()
        checkpoint = self._checkpoint()

        processed = 0
        stopped_for_budget = False
        latest_processed_sent_at = checkpoint.last_successful_processed_at

        while True:
            next_item = self._next_pending()
            if not next_item:
                checkpoint.last_successful_processed_at = latest_processed_sent_at
                self.session.flush()
                return RunResult(processed=processed, deferred=0, stopped_for_budget=stopped_for_budget)

            decision = self.budget.can_claim_next_unit(settings.estimated_llama_cost_per_message_usd)
            if not decision.allowed:
                deferred = self.budget.enforce_cap()
                stopped_for_budget = True
                self.session.flush()
                return RunResult(processed=processed, deferred=deferred, stopped_for_budget=stopped_for_budget)

            self._process_one(next_item)
            msg = self.session.get(MessageRecord, next_item.message_id)
            if msg and (latest_processed_sent_at is None or msg.sent_at > latest_processed_sent_at):
                latest_processed_sent_at = msg.sent_at
            processed += 1
