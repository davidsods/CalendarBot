from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import CalendarLink, EventSuggestion, MessageRecord, SuggestionStatus
from app.services.approvals import ApprovalService
from app.services.integrations import GoogleCalendarClient


def _message(session: Session, external_id: str, thread_id: str = "thread-1") -> MessageRecord:
    now = datetime.now(timezone.utc)
    msg = MessageRecord(
        external_message_id=external_id,
        batch_id="b1",
        thread_id=thread_id,
        sender_role="other",
        text="hello",
        sent_at=now,
        received_at=now,
    )
    session.add(msg)
    session.flush()
    return msg


def _suggestion(
    session: Session,
    action: str,
    thread_id: str = "thread-1",
    target_event_ref: str | None = None,
) -> EventSuggestion:
    msg = _message(session, external_id=f"msg-{action}-{thread_id}", thread_id=thread_id)
    suggestion = EventSuggestion(
        message_id=msg.id,
        thread_id=thread_id,
        action=action,
        target_event_ref=target_event_ref,
        title="Original",
        event_date=date(2026, 3, 10),
        is_all_day=True,
        timezone="America/Los_Angeles",
        confidence=0.9,
    )
    session.add(suggestion)
    session.flush()
    return suggestion


def test_handle_action_returns_not_found_for_unknown_suggestion(db_session: Session) -> None:
    status = ApprovalService(db_session).handle_action(999, "approve_create")
    assert status == "not_found"


def test_reject_sets_status_rejected(db_session: Session) -> None:
    suggestion = _suggestion(db_session, action="create")

    status = ApprovalService(db_session).handle_action(suggestion.id, "reject")

    assert status == "rejected"
    assert suggestion.status == SuggestionStatus.rejected


def test_edit_then_approve_create_creates_calendar_link(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suggestion = _suggestion(db_session, action="create", thread_id="thread-create")

    def _create_event(self: GoogleCalendarClient, title: str, **kwargs: object) -> str:
        assert title == "Renamed"
        assert kwargs["event_date"] == date(2026, 3, 10)
        assert kwargs["is_all_day"] is True
        return "evt-123"

    monkeypatch.setattr(GoogleCalendarClient, "create_event", _create_event)

    status = ApprovalService(db_session).handle_action(suggestion.id, "edit_then_approve", edited_title="Renamed")

    link = db_session.scalar(select(CalendarLink).where(CalendarLink.thread_id == suggestion.thread_id))
    assert status == "approved"
    assert suggestion.status == SuggestionStatus.approved
    assert suggestion.title == "Renamed"
    assert link is not None
    assert link.google_event_id == "evt-123"


def test_edit_then_approve_update_calls_google_update(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suggestion = _suggestion(
        db_session,
        action="update",
        thread_id="thread-update",
        target_event_ref="existing-evt",
    )
    called: dict[str, object] = {}

    def _update_event(self: GoogleCalendarClient, google_event_id: str, title: str, **kwargs: object) -> str:
        called["google_event_id"] = google_event_id
        called["title"] = title
        called["is_all_day"] = kwargs["is_all_day"]
        return google_event_id

    monkeypatch.setattr(GoogleCalendarClient, "update_event", _update_event)

    status = ApprovalService(db_session).handle_action(suggestion.id, "edit_then_approve", edited_title="Moved")

    assert status == "approved"
    assert suggestion.status == SuggestionStatus.approved
    assert called == {"google_event_id": "existing-evt", "title": "Moved", "is_all_day": True}


def test_mismatched_approve_action_is_invalid_for_create(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suggestion = _suggestion(db_session, action="create")

    def _fail_create(self: GoogleCalendarClient, title: str) -> str:
        raise AssertionError("create_event should not be called")

    def _fail_update(self: GoogleCalendarClient, google_event_id: str, title: str) -> str:
        raise AssertionError("update_event should not be called")

    monkeypatch.setattr(GoogleCalendarClient, "create_event", _fail_create)
    monkeypatch.setattr(GoogleCalendarClient, "update_event", _fail_update)

    status = ApprovalService(db_session).handle_action(suggestion.id, "approve_update", edited_title="Nope")

    assert status == "invalid_action"
    assert suggestion.status == SuggestionStatus.pending_approval
    assert suggestion.title == "Original"


def test_mismatched_approve_action_is_invalid_for_update(db_session: Session) -> None:
    suggestion = _suggestion(db_session, action="update", target_event_ref="evt-1")

    status = ApprovalService(db_session).handle_action(suggestion.id, "approve_create")

    assert status == "invalid_action"
    assert suggestion.status == SuggestionStatus.pending_approval


def test_unknown_action_is_invalid_and_does_not_mutate(db_session: Session) -> None:
    suggestion = _suggestion(db_session, action="create")

    status = ApprovalService(db_session).handle_action(suggestion.id, "ship_it", edited_title="ShouldNotApply")

    assert status == "invalid_action"
    assert suggestion.status == SuggestionStatus.pending_approval
    assert suggestion.title == "Original"
