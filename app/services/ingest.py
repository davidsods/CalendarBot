from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import MessageRecord, ProcessingCheckpoint, QueueItem, QueueStatus
from app.schemas import IngestMessage


@dataclass
class IngestResult:
    deduped: int
    queued: int


class IngestService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _checkpoint(self) -> ProcessingCheckpoint:
        checkpoint = self.session.get(ProcessingCheckpoint, 1)
        if checkpoint:
            return checkpoint
        checkpoint = ProcessingCheckpoint(id=1, last_successful_processed_at=None)
        self.session.add(checkpoint)
        self.session.flush()
        return checkpoint

    def ingest_batch(self, batch_id: str, messages: list[IngestMessage]) -> IngestResult:
        checkpoint = self._checkpoint()
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        checkpoint_cutoff = checkpoint.last_successful_processed_at
        if checkpoint_cutoff and checkpoint_cutoff.tzinfo:
            checkpoint_cutoff = checkpoint_cutoff.astimezone(timezone.utc).replace(tzinfo=None)
        # Guard against a future-skewed checkpoint blocking all new ingestion.
        if checkpoint_cutoff and checkpoint_cutoff > now_utc + timedelta(minutes=5):
            checkpoint_cutoff = None

        deduped = 0
        queued = 0

        for msg in messages:
            sent_at = msg.sent_at.astimezone(timezone.utc).replace(tzinfo=None) if msg.sent_at.tzinfo else msg.sent_at
            received_at = (
                msg.received_at.astimezone(timezone.utc).replace(tzinfo=None)
                if msg.received_at.tzinfo
                else msg.received_at
            )

            existing = self.session.scalar(
                select(MessageRecord).where(MessageRecord.external_message_id == msg.external_message_id)
            )
            if existing:
                deduped += 1
                continue

            if checkpoint_cutoff and sent_at <= checkpoint_cutoff:
                deduped += 1
                continue

            record = MessageRecord(
                external_message_id=msg.external_message_id,
                batch_id=batch_id,
                thread_id=msg.thread_id,
                sender_role=msg.sender_role,
                text=msg.text,
                sent_at=sent_at,
                received_at=received_at,
            )
            self.session.add(record)
            try:
                self.session.flush()
            except IntegrityError:
                self.session.rollback()
                deduped += 1
                continue

            queue_item = QueueItem(message_id=record.id, status=QueueStatus.pending)
            self.session.add(queue_item)
            queued += 1

        self.session.flush()
        return IngestResult(deduped=deduped, queued=queued)
