from __future__ import annotations

from dataclasses import dataclass
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

    def _process_one(self, item: QueueItem) -> None:
        item.status = QueueStatus.processing
        item.attempts += 1
        self.session.flush()

        msg = self.session.get(MessageRecord, item.message_id)
        assert msg is not None

        existing_link = self.session.scalar(select(CalendarLink).where(CalendarLink.thread_id == msg.thread_id))
        candidate = self.extractor.extract(msg.text, has_existing_thread_event=bool(existing_link))

        suggestion = EventSuggestion(
            message_id=msg.id,
            thread_id=msg.thread_id,
            action=candidate.action,
            target_event_ref=existing_link.google_event_id if existing_link else None,
            title=candidate.title or "Meeting",
            confidence=candidate.confidence,
        )
        self.session.add(suggestion)
        self.session.flush()

        try:
            self.slack.send_suggestion(suggestion.id, suggestion.action)
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
