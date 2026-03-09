from __future__ import annotations

import enum
from datetime import date, datetime, timezone

from sqlalchemy import Boolean, Date, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class QueueStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    deferred_budget_cap = "deferred_budget_cap"


class SuggestionStatus(str, enum.Enum):
    pending_approval = "pending_approval"
    approved = "approved"
    rejected = "rejected"


class BudgetStatus(str, enum.Enum):
    ok = "ok"
    capped = "capped"


class MessageRecord(Base):
    __tablename__ = "message_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_message_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    batch_id: Mapped[str] = mapped_column(String(255), index=True)
    thread_id: Mapped[str] = mapped_column(String(255), index=True)
    sender_role: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(Text)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    queue_item: Mapped["QueueItem | None"] = relationship(back_populates="message", uselist=False)


class QueueItem(Base):
    __tablename__ = "queue_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("message_records.id"), unique=True)
    status: Mapped[QueueStatus] = mapped_column(Enum(QueueStatus), default=QueueStatus.pending, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    deferred_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    message: Mapped[MessageRecord] = relationship(back_populates="queue_item")


class ProcessingCheckpoint(Base):
    __tablename__ = "processing_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_successful_processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BudgetState(Base):
    __tablename__ = "budget_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    month: Mapped[str] = mapped_column(String(7), unique=True, index=True)  # YYYY-MM
    spend_estimate_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cap_usd: Mapped[float] = mapped_column(Float)
    status: Mapped[BudgetStatus] = mapped_column(Enum(BudgetStatus), default=BudgetStatus.ok)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class EventSuggestion(Base):
    __tablename__ = "event_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("message_records.id"), index=True)
    thread_id: Mapped[str] = mapped_column(String(255), index=True)
    action: Mapped[str] = mapped_column(String(20))  # create|update|ignore
    target_event_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str] = mapped_column(String(255), default="Meeting")
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_all_day: Mapped[bool] = mapped_column(Boolean, default=False, nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reason_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_message_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_window_size: Mapped[int] = mapped_column(Integer, default=0, nullable=True)
    thread_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence_tier: Mapped[str | None] = mapped_column(String(20), nullable=True)
    slot_candidate_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[SuggestionStatus] = mapped_column(Enum(SuggestionStatus), default=SuggestionStatus.pending_approval)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CalendarLink(Base):
    __tablename__ = "calendar_links"
    __table_args__ = (UniqueConstraint("thread_id", name="uq_calendar_link_thread"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(255), index=True)
    google_event_id: Mapped[str] = mapped_column(String(255), unique=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ThreadPlanningState(Base):
    __tablename__ = "thread_planning_states"
    __table_args__ = (UniqueConstraint("thread_id", name="uq_thread_planning_state_thread"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(255), index=True)
    state: Mapped[str] = mapped_column(String(32), default="exploring", index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended_slot_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    decision_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ThreadSlotCandidate(Base):
    __tablename__ = "thread_slot_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(255), index=True)
    slot_key: Mapped[str] = mapped_column(String(128), index=True)
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_all_day: Mapped[bool] = mapped_column(Boolean, default=False)
    timezone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str] = mapped_column(String(255), default="Meeting")
    proposer_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    supporting_message_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    contradicting_message_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    recency_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    last_evidence_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(100), index=True)
    payload: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GoogleOAuthToken(Base):
    __tablename__ = "google_oauth_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(100), unique=True, default="default")
    access_token: Mapped[str] = mapped_column(Text)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_type: Mapped[str] = mapped_column(String(20), default="Bearer")
    scope: Mapped[str | None] = mapped_column(Text, nullable=True)
    expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
