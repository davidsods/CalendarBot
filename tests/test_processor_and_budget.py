from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from zoneinfo import ZoneInfo

from app.config import settings
from app.db import Base
from app.models import EventSuggestion, MessageRecord, ProcessingCheckpoint, QueueItem, QueueStatus
from app.schemas import IngestMessage
from app.services.approvals import ApprovalService
from app.services.ingest import IngestService
from app.services.processor import ProcessorService


def _session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    return factory()


def _ingest_messages(session: Session, count: int, thread_id: str = "t1", start_minute: int = 0) -> None:
    svc = IngestService(session)
    now = datetime.now(timezone.utc)
    msgs = [
        IngestMessage(
            external_message_id=f"m-{thread_id}-{i}-{start_minute}",
            thread_id=thread_id,
            sender_role="other",
            text="let's meet tomorrow at 2pm",
            sent_at=now + timedelta(minutes=i + start_minute),
            received_at=now + timedelta(minutes=i + start_minute),
        )
        for i in range(count)
    ]
    svc.ingest_batch(batch_id="b1", messages=msgs)
    session.commit()


def test_budget_cap_defers_remaining_and_finishes_inflight() -> None:
    settings.monthly_budget_cap_usd = 15.0
    settings.budget_safety_buffer_usd = 1.0
    settings.estimated_llama_cost_per_message_usd = 8.0

    session = _session()
    _ingest_messages(session, 3)

    result = ProcessorService(session).run_once()
    session.commit()

    assert result.processed == 1
    assert result.deferred == 2
    assert result.stopped_for_budget is True

    statuses = session.scalars(select(QueueItem.status)).all()
    assert statuses.count(QueueStatus.completed) == 1
    assert statuses.count(QueueStatus.deferred_budget_cap) == 2


def test_checkpoint_only_advances_after_full_drain() -> None:
    settings.monthly_budget_cap_usd = 15.0
    settings.budget_safety_buffer_usd = 1.0
    settings.estimated_llama_cost_per_message_usd = 8.0

    session = _session()
    _ingest_messages(session, 3)

    ProcessorService(session).run_once()
    session.commit()

    checkpoint = session.get(ProcessingCheckpoint, 1)
    assert checkpoint is not None
    assert checkpoint.last_successful_processed_at is None


def test_next_cycle_does_not_resume_when_capped() -> None:
    settings.monthly_budget_cap_usd = 15.0
    settings.budget_safety_buffer_usd = 1.0
    settings.estimated_llama_cost_per_message_usd = 8.0

    session = _session()
    _ingest_messages(session, 2)
    ProcessorService(session).run_once()
    session.commit()

    # New message arrives while month is capped.
    svc = IngestService(session)
    now = datetime.now(timezone.utc)
    svc.ingest_batch(
        batch_id="b2",
        messages=[
            IngestMessage(
                external_message_id="m-new",
                thread_id="t1",
                sender_role="other",
                text="call tomorrow 5pm",
                sent_at=now,
                received_at=now,
            )
        ],
    )
    session.commit()

    result = ProcessorService(session).run_once()
    session.commit()

    assert result.processed == 0
    assert result.stopped_for_budget is True

    new_item = session.scalar(select(QueueItem).join(MessageRecord).where(MessageRecord.external_message_id == "m-new"))
    assert new_item is not None
    assert new_item.status == QueueStatus.deferred_budget_cap


def test_reschedule_creates_update_suggestion_not_duplicate() -> None:
    settings.monthly_budget_cap_usd = 100.0
    settings.budget_safety_buffer_usd = 1.0
    settings.estimated_llama_cost_per_message_usd = 0.1

    session = _session()
    now = datetime.now(timezone.utc)
    ingest = IngestService(session)
    ingest.ingest_batch(
        "b1",
        [
            IngestMessage(
                external_message_id="m1",
                thread_id="thread-123",
                sender_role="other",
                text="let's meet tomorrow at 9am",
                sent_at=now,
                received_at=now,
            )
        ],
    )
    session.commit()

    ProcessorService(session).run_once()
    session.commit()

    first = session.scalars(select(EventSuggestion).order_by(EventSuggestion.id)).first()
    assert first is not None
    ApprovalService(session).handle_action(first.id, "approve_create")
    session.commit()

    ingest.ingest_batch(
        "b2",
        [
            IngestMessage(
                external_message_id="m2",
                thread_id="thread-123",
                sender_role="other",
                text="let's move this to 3pm",
                sent_at=now + timedelta(minutes=5),
                received_at=now + timedelta(minutes=5),
            )
        ],
    )
    session.commit()

    ProcessorService(session).run_once()
    session.commit()

    suggestions = session.scalars(select(EventSuggestion).order_by(EventSuggestion.id)).all()
    assert len(suggestions) == 2
    assert suggestions[-1].action == "update"
    assert suggestions[-1].target_event_ref is not None


def test_run_once_skips_outside_active_window() -> None:
    settings.processor_tz = "America/Los_Angeles"
    settings.processor_active_start_hour = 6
    settings.processor_active_end_hour = 24

    session = _session()
    _ingest_messages(session, 1)

    # 2026-03-08 12:30 UTC = 04:30 PT (outside window)
    now_utc = datetime(2026, 3, 8, 12, 30, tzinfo=ZoneInfo("UTC"))
    result = ProcessorService(session).run_once(now_utc=now_utc)
    session.commit()

    assert result.processed == 0
    assert result.deferred == 0
    assert result.stopped_for_budget is False
    assert result.skipped_for_window is True

    statuses = session.scalars(select(QueueItem.status)).all()
    assert statuses.count(QueueStatus.pending) == 1


def test_thread_coalescing_processes_latest_message_once() -> None:
    settings.thread_coalesce_window_minutes = 60
    settings.monthly_budget_cap_usd = 100.0
    settings.budget_safety_buffer_usd = 1.0
    settings.estimated_llama_cost_per_message_usd = 0.1

    session = _session()
    _ingest_messages(session, 3, thread_id="thread-coalesce")

    result = ProcessorService(session).run_once()
    session.commit()

    assert result.processed == 1
    statuses = session.scalars(select(QueueItem.status)).all()
    assert statuses.count(QueueStatus.completed) == 3

    deferred_reasons = session.scalars(select(QueueItem.deferred_reason)).all()
    assert deferred_reasons.count("coalesced_thread_window") == 2


def test_active_window_boundaries_midnight_and_six_am_pt_on_dst_date() -> None:
    settings.processor_tz = "America/Los_Angeles"
    settings.processor_active_start_hour = 6
    settings.processor_active_end_hour = 24

    session = _session()
    _ingest_messages(session, 1, thread_id="dst-thread")

    svc = ProcessorService(session)
    # 2026-03-08 08:01 UTC = 00:01 PT (DST start day), must skip.
    skipped = svc.run_once(now_utc=datetime(2026, 3, 8, 8, 1, tzinfo=ZoneInfo("UTC")))
    assert skipped.skipped_for_window is True

    # 2026-03-08 13:05 UTC = 06:05 PT (same day), must process.
    processed = svc.run_once(now_utc=datetime(2026, 3, 8, 13, 5, tzinfo=ZoneInfo("UTC")))
    session.commit()
    assert processed.skipped_for_window is False
    assert processed.processed == 1
