from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class IngestMessage(BaseModel):
    external_message_id: str
    thread_id: str
    sender_role: str
    text: str
    sent_at: datetime
    received_at: datetime


class IngestRequest(BaseModel):
    device_id: str
    batch_id: str
    messages: list[IngestMessage]


class IngestResponse(BaseModel):
    accepted: bool
    deduped: int
    queued_for_extraction: int


class SlackActionRequest(BaseModel):
    suggestion_id: int
    action: str  # approve_create|approve_update|edit_then_approve|reject
    edited_title: str | None = None


class ProcessorRunResult(BaseModel):
    processed: int
    deferred: int
    stopped_for_budget: bool


class GoogleOAuthCallbackRequest(BaseModel):
    code: str


class GoogleOAuthStatusResponse(BaseModel):
    connected: bool
    has_refresh_token: bool
    expiry: datetime | None = None
