from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import MessageRecord, ProcessingCheckpoint, QueueItem
from app.schemas import IngestMessage
from app.services.ingest import IngestService


def _msg(message_id: str, sent_at: datetime, thread_id: str = "thread-1") -> IngestMessage:
    return IngestMessage(
        external_message_id=message_id,
        thread_id=thread_id,
        sender_role="other",
        text="ping",
        sent_at=sent_at,
        received_at=sent_at,
    )


def test_ingest_respects_checkpoint_cutoff(db_session: Session) -> None:
    cutoff = datetime(2026, 3, 1, 12, 0, 0)
    db_session.add(ProcessingCheckpoint(id=1, last_successful_processed_at=cutoff))
    db_session.flush()

    service = IngestService(db_session)
    result = service.ingest_batch(
        "batch-1",
        [
            _msg("m-old", cutoff),
            _msg("m-new", cutoff + timedelta(minutes=1)),
        ],
    )

    queued = db_session.scalars(select(QueueItem)).all()
    assert result.deduped == 1
    assert result.queued == 1
    assert len(queued) == 1


def test_ingest_ignores_future_skewed_checkpoint(db_session: Session) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    db_session.add(ProcessingCheckpoint(id=1, last_successful_processed_at=now + timedelta(days=1)))
    db_session.flush()

    service = IngestService(db_session)
    result = service.ingest_batch(
        "batch-1",
        [
            _msg("m-now", now),
        ],
    )

    record = db_session.scalar(select(MessageRecord).where(MessageRecord.external_message_id == "m-now"))
    assert result.deduped == 0
    assert result.queued == 1
    assert record is not None


def test_ingest_normalizes_timezone_aware_datetimes_to_naive_utc(db_session: Session) -> None:
    pacific = timezone(timedelta(hours=-8))
    aware = datetime(2026, 3, 1, 9, 30, tzinfo=pacific)

    result = IngestService(db_session).ingest_batch("batch-1", [_msg("m-tz", aware)])

    record = db_session.scalar(select(MessageRecord).where(MessageRecord.external_message_id == "m-tz"))
    assert result.queued == 1
    assert record is not None
    assert record.sent_at.tzinfo is None
    assert record.received_at.tzinfo is None
    assert record.sent_at == datetime(2026, 3, 1, 17, 30)


def test_ingest_handles_timezone_aware_checkpoint(db_session: Session) -> None:
    aware_cutoff = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    db_session.add(ProcessingCheckpoint(id=1, last_successful_processed_at=aware_cutoff))
    db_session.flush()

    service = IngestService(db_session)
    result = service.ingest_batch(
        "batch-1",
        [
            _msg("m-aware-old", datetime(2026, 3, 1, 11, 59, 0, tzinfo=timezone.utc)),
            _msg("m-aware-new", datetime(2026, 3, 1, 12, 1, 0, tzinfo=timezone.utc)),
        ],
    )

    ids = db_session.scalars(select(MessageRecord.external_message_id).order_by(MessageRecord.id)).all()
    assert result.deduped == 1
    assert result.queued == 1
    assert ids == ["m-aware-new"]


def test_ingest_dedupes_duplicate_external_id_in_same_batch(db_session: Session) -> None:
    now = datetime.now(timezone.utc)
    service = IngestService(db_session)
    result = service.ingest_batch(
        "batch-1",
        [
            _msg("m-dupe", now),
            _msg("m-dupe", now + timedelta(seconds=1)),
        ],
    )

    rows = db_session.scalars(select(MessageRecord).where(MessageRecord.external_message_id == "m-dupe")).all()
    assert result.queued == 1
    assert result.deduped == 1
    assert len(rows) == 1


def test_ingest_handles_integrity_error_and_continues(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = IngestService(db_session)
    original_flush = db_session.flush
    raised = {"done": False}

    def _flaky_flush() -> None:
        should_fail = any(
            isinstance(obj, MessageRecord) and obj.external_message_id == "m-first" for obj in db_session.new
        )
        if should_fail and not raised["done"]:
            raised["done"] = True
            raise IntegrityError("insert", {}, Exception("duplicate"))
        original_flush()

    monkeypatch.setattr(db_session, "flush", _flaky_flush)

    result = service.ingest_batch(
        "batch-1",
        [
            _msg("m-first", datetime.now(timezone.utc)),
            _msg("m-second", datetime.now(timezone.utc) + timedelta(minutes=1)),
        ],
    )

    queued_ids = db_session.scalars(select(MessageRecord.external_message_id).order_by(MessageRecord.id)).all()
    assert result.deduped == 1
    assert result.queued == 1
    assert queued_ids == ["m-second"]
