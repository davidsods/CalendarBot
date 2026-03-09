from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.schemas import IngestMessage
from app.services.extraction_utils import evaluate_thread_state
from app.services.ingest import IngestService
from app.services.processor import ProcessorService
from app.models import EventSuggestion


def test_group_chat_likely_consensus_from_proposal_and_soft_accept() -> None:
    now = datetime(2026, 3, 9, 18, 0, tzinfo=timezone.utc)
    decision = evaluate_thread_state(
        messages=[
            {"id": 1, "sender_role": "other", "text": "How about Friday 3pm PT for planning?", "sent_at": now},
            {"id": 2, "sender_role": "self", "text": "I think so, should work for me.", "sent_at": now + timedelta(minutes=1)},
            {"id": 3, "sender_role": "other", "text": "great let's do it", "sent_at": now + timedelta(minutes=2)},
        ],
        default_timezone="America/Los_Angeles",
        existing_slots=[],
    )
    assert decision.thread_state in {"likely_consensus", "confirmed"}
    assert decision.should_generate is True
    assert decision.recommended_slot_key is not None


def test_latest_confirmed_wins_when_new_slot_supersedes_old() -> None:
    now = datetime(2026, 3, 9, 18, 0, tzinfo=timezone.utc)
    decision = evaluate_thread_state(
        messages=[
            {"id": 1, "sender_role": "other", "text": "Tuesday at 2pm works?", "sent_at": now},
            {"id": 2, "sender_role": "self", "text": "works for me", "sent_at": now + timedelta(minutes=1)},
            {"id": 3, "sender_role": "other", "text": "actually move it to Wednesday at 3pm", "sent_at": now + timedelta(minutes=2)},
            {"id": 4, "sender_role": "self", "text": "yes that works", "sent_at": now + timedelta(minutes=3)},
        ],
        default_timezone="America/Los_Angeles",
        existing_slots=[],
    )
    assert decision.recommended_slot_key is not None
    assert "2026-03-11" in decision.recommended_slot_key
    assert decision.should_generate is True


def test_logistics_chatter_without_slot_does_not_generate() -> None:
    now = datetime(2026, 3, 9, 18, 0, tzinfo=timezone.utc)
    decision = evaluate_thread_state(
        messages=[
            {"id": 1, "sender_role": "other", "text": "let's sync later", "sent_at": now},
            {"id": 2, "sender_role": "self", "text": "sounds good", "sent_at": now + timedelta(minutes=1)},
        ],
        default_timezone="America/Los_Angeles",
        existing_slots=[],
    )
    assert decision.should_generate is False
    assert decision.thread_state in {"exploring", "candidate_slots"}


def test_pending_suggestion_is_revised_not_duplicated(db_session: Session) -> None:
    now = datetime.now(timezone.utc)
    ingest = IngestService(db_session)
    ingest.ingest_batch(
        "b1",
        [
            IngestMessage(
                external_message_id="m1",
                thread_id="t-revise",
                sender_role="other",
                text="Friday at 2pm?",
                sent_at=now,
                received_at=now,
            )
        ],
    )
    db_session.commit()
    ProcessorService(db_session).run_once()
    db_session.commit()

    ingest.ingest_batch(
        "b2",
        [
            IngestMessage(
                external_message_id="m2",
                thread_id="t-revise",
                sender_role="self",
                text="actually 3pm works better, I think so",
                sent_at=now + timedelta(minutes=5),
                received_at=now + timedelta(minutes=5),
            )
        ],
    )
    db_session.commit()
    ProcessorService(db_session).run_once()
    db_session.commit()

    suggestions = db_session.scalars(select(EventSuggestion).where(EventSuggestion.thread_id == "t-revise")).all()
    pending = [s for s in suggestions if s.status.value == "pending_approval"]
    assert len(pending) == 1
