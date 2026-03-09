from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
)
from app.services.budget import BudgetService
from app.services.extractor import LlamaExtractor
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

    def _process_one(self, item: QueueItem) -> None:
        item.status = QueueStatus.processing
        item.attempts += 1
        self.session.flush()

        msg = self.session.get(MessageRecord, item.message_id)
        assert msg is not None

        existing_link = self.session.scalar(select(CalendarLink).where(CalendarLink.thread_id == msg.thread_id))

        pending_stmt = select(EventSuggestion).where(
            EventSuggestion.thread_id == msg.thread_id,
            EventSuggestion.status == SuggestionStatus.pending_approval,
        )
        existing_pending = self.session.scalar(pending_stmt)
        if existing_pending is not None:
            item.status = QueueStatus.completed
            self.budget.record_usage(settings.estimated_llama_cost_per_message_usd)
            self.session.flush()
            return

        thread_messages = self._load_thread_messages(msg.thread_id, msg.sent_at)
        candidate = self.extractor.extract_thread(
            thread_messages,
            has_existing_thread_event=bool(existing_link),
            reference_utc=msg.sent_at,
        )
        if candidate.action == "ignore":
            item.status = QueueStatus.completed
            self.budget.record_usage(settings.estimated_llama_cost_per_message_usd)
            self.session.flush()
            return

        suggestion = EventSuggestion(
            message_id=msg.id,
            thread_id=msg.thread_id,
            action=candidate.action,
            target_event_ref=existing_link.google_event_id if existing_link else None,
            title=candidate.title or "Meeting",
            start_at=candidate.start_at,
            end_at=candidate.end_at,
            event_date=candidate.event_date,
            is_all_day=bool(candidate.is_all_day),
            timezone=candidate.timezone or settings.default_timezone,
            confidence=candidate.confidence,
            reason_summary=candidate.reason_summary,
            evidence_message_ids_json=json.dumps(candidate.evidence_message_ids or []),
            context_window_size=len(thread_messages),
        )
        self.session.add(suggestion)
        self.session.flush()

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
