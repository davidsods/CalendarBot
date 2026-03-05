from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import CalendarLink, EventSuggestion, SuggestionStatus
from app.services.integrations import GoogleCalendarClient


class ApprovalService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.google = GoogleCalendarClient(session)

    def handle_action(self, suggestion_id: int, action: str, edited_title: str | None = None) -> str:
        suggestion = self.session.get(EventSuggestion, suggestion_id)
        if not suggestion:
            return "not_found"

        if action not in {"approve_create", "approve_update", "edit_then_approve", "reject"}:
            return "invalid_action"

        if action == "reject":
            suggestion.status = SuggestionStatus.rejected
            self.session.flush()
            return "rejected"

        normalized_action = action
        if action == "edit_then_approve":
            if suggestion.action == "update":
                normalized_action = "approve_update"
            elif suggestion.action == "create":
                normalized_action = "approve_create"
            else:
                return "invalid_action"

        if normalized_action == "approve_create" and suggestion.action != "create":
            return "invalid_action"
        if normalized_action == "approve_update" and suggestion.action != "update":
            return "invalid_action"

        if edited_title:
            suggestion.title = edited_title

        if normalized_action == "approve_create" and suggestion.action == "create":
            google_event_id = self.google.create_event(suggestion.title)
            link = self.session.query(CalendarLink).filter(CalendarLink.thread_id == suggestion.thread_id).one_or_none()
            if link:
                link.google_event_id = google_event_id
            else:
                self.session.add(CalendarLink(thread_id=suggestion.thread_id, google_event_id=google_event_id))

        if normalized_action == "approve_update" and suggestion.action == "update":
            if suggestion.target_event_ref:
                self.google.update_event(suggestion.target_event_ref, suggestion.title)

        suggestion.status = SuggestionStatus.approved
        self.session.flush()
        return "approved"
