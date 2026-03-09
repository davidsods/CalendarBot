from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo
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
    skipped_for_window: bool = False


class ProcessorService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.budget = BudgetService(session)
        self.extractor = LlamaExtractor()
        self.slack = SlackNotifier()

    def _log_event(self, event_type: str, payload: dict[str, object]) -> None:
        self.session.add(
            AuditLog(
                event_type=event_type,
                payload=json.dumps(payload),
            )
        )

    def _is_within_processing_window(self, now_utc: datetime | None = None) -> bool:
        start_hour = max(0, min(23, int(settings.processor_active_start_hour)))
        end_hour = max(1, min(24, int(settings.processor_active_end_hour)))
        if start_hour >= end_hour:
            return True
        now = now_utc or datetime.now(ZoneInfo("UTC"))
        aware_utc = now if now.tzinfo else now.replace(tzinfo=ZoneInfo("UTC"))
        local_now = aware_utc.astimezone(ZoneInfo(settings.processor_tz))
        return start_hour <= local_now.hour < end_hour

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

    def _thread_pending_items_in_window(self, seed: QueueItem) -> list[QueueItem]:
        seed_msg = self.session.get(MessageRecord, seed.message_id)
        if not seed_msg:
            return [seed]
        window_minutes = max(0, int(settings.thread_coalesce_window_minutes))
        if window_minutes <= 0:
            return [seed]

        window_end = seed_msg.sent_at + timedelta(minutes=window_minutes)
        stmt = (
            select(QueueItem)
            .join(MessageRecord, QueueItem.message_id == MessageRecord.id)
            .where(
                QueueItem.status == QueueStatus.pending,
                MessageRecord.thread_id == seed_msg.thread_id,
                MessageRecord.sent_at >= seed_msg.sent_at,
                MessageRecord.sent_at <= window_end,
            )
            .order_by(asc(MessageRecord.sent_at), asc(QueueItem.id))
        )
        rows = self.session.scalars(stmt).all()
        return rows or [seed]

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
        if existing_pending is None:
            self.session.add(suggestion)
        self.session.flush()
        return suggestion, is_new

    def _latest_message_is_meta(self, thread_messages: list[dict[str, object]]) -> bool:
        if not thread_messages:
            return True
        latest = thread_messages[-1]
        text = str(latest.get("text", "") or "")
        sent_at = latest.get("sent_at")
        if not isinstance(sent_at, datetime):
            sent_at = datetime.now(ZoneInfo("UTC"))
        schedule = parse_schedule(text, sent_at, settings.default_timezone)
        signal = classify_signal(text, schedule)
        return signal.signal_type == "meta"

    def _process_one(self, item: QueueItem) -> None:
        item.status = QueueStatus.processing
        item.attempts += 1
        self.session.flush()

        msg = self.session.get(MessageRecord, item.message_id)
        assert msg is not None

        existing_link = self.session.scalar(select(CalendarLink).where(CalendarLink.thread_id == msg.thread_id))
        existing_pending = self.session.scalar(
            select(EventSuggestion).where(
                EventSuggestion.thread_id == msg.thread_id,
                EventSuggestion.status == SuggestionStatus.pending_approval,
            )
        )

        thread_messages = self._load_thread_messages(msg.thread_id, msg.sent_at)
        if existing_pending is not None and self._latest_message_is_meta(thread_messages):
            item.status = QueueStatus.completed
            self._log_event(
                "llama_gate_skipped",
                {
                    "reason": "pending_suggestion_meta_message",
                    "thread_id": msg.thread_id,
                    "message_id": msg.id,
                },
            )
            self.session.flush()
            return

        existing_slots = self._load_existing_slots(msg.thread_id)
        candidate = self.extractor.extract_thread(
            thread_messages,
            has_existing_thread_event=bool(existing_link),
            reference_utc=msg.sent_at,
            existing_slots=existing_slots,
        )
        if candidate.model_invoked:
            self._log_event(
                "llama_invoked",
                {
                    "thread_id": msg.thread_id,
                    "message_id": msg.id,
                    "thread_state": candidate.thread_state,
                },
            )
        else:
            self._log_event(
                "llama_gate_skipped",
                {
                    "reason": "heuristic_gate",
                    "thread_id": msg.thread_id,
                    "message_id": msg.id,
                    "thread_state": candidate.thread_state,
                },
            )
        slot_candidate_id = self._persist_thread_decision(msg.thread_id, candidate)

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
        self._log_event(
            "thread_processed",
            {
                "thread_id": msg.thread_id,
                "message_id": msg.id,
                "model_invoked": candidate.model_invoked,
                "estimated_cost_usd": settings.estimated_llama_cost_per_message_usd if candidate.model_invoked else 0.0,
            },
        )
        self.session.flush()

    def run_once(self, now_utc: datetime | None = None) -> RunResult:
        if not self._is_within_processing_window(now_utc=now_utc):
            self._log_event(
                "processor_window_skip",
                {
                    "processor_tz": settings.processor_tz,
                    "start_hour": settings.processor_active_start_hour,
                    "end_hour": settings.processor_active_end_hour,
                },
            )
            self.session.flush()
            return RunResult(processed=0, deferred=0, stopped_for_budget=False, skipped_for_window=True)

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
                self._log_event(
                    "budget_cap_enforced",
                    {
                        "deferred": deferred,
                        "cap_usd": settings.monthly_budget_cap_usd,
                    },
                )
                stopped_for_budget = True
                self.session.flush()
                return RunResult(
                    processed=processed,
                    deferred=deferred,
                    stopped_for_budget=stopped_for_budget,
                    skipped_for_window=False,
                )

            coalesced_items = self._thread_pending_items_in_window(next_item)
            process_item = coalesced_items[-1]
            skipped_items = coalesced_items[:-1]
            for skipped in skipped_items:
                skipped.status = QueueStatus.completed
                skipped.deferred_reason = "coalesced_thread_window"
            if skipped_items:
                process_msg = self.session.get(MessageRecord, process_item.message_id)
                self._log_event(
                    "thread_coalesced",
                    {
                        "thread_id": process_msg.thread_id if process_msg else "",
                        "skipped_messages": len(skipped_items),
                    },
                )

            self._process_one(process_item)
            msg = self.session.get(MessageRecord, process_item.message_id)
            if msg and (latest_processed_sent_at is None or msg.sent_at > latest_processed_sent_at):
                latest_processed_sent_at = msg.sent_at
            processed += 1
