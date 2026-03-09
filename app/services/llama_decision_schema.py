from __future__ import annotations

from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class LlamaSlotCandidatePayload(BaseModel):
    slot_key: str
    event_date: date | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None
    is_all_day: bool = False
    timezone: str
    title: str = "Meeting"
    proposer_message_id: int | None = None
    supporting_message_ids: list[int] = Field(default_factory=list)
    contradicting_message_ids: list[int] = Field(default_factory=list)
    score: float = 0.0
    recency_score: float = 0.0

    @field_validator("score", "recency_score", mode="before")
    @classmethod
    def _to_float(cls, value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


class LlamaThreadDecisionPayload(BaseModel):
    should_generate: bool
    thread_state: str
    confidence_tier: str
    decision_confidence: float
    action: str
    title: str
    event_date: date | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None
    is_all_day: bool = False
    timezone: str
    recommended_slot_key: str | None = None
    slot_candidates: list[LlamaSlotCandidatePayload] = Field(default_factory=list)
    decision_rationale: str
    slack_summary: str | None = None
    conflict_note: str | None = None
    evidence_message_ids: list[int] = Field(default_factory=list)

    @field_validator("thread_state")
    @classmethod
    def _validate_state(cls, value: str) -> str:
        allowed = {"exploring", "candidate_slots", "likely_consensus", "confirmed", "reschedule_pending", "canceled"}
        v = value.strip().lower()
        if v not in allowed:
            raise ValueError("invalid thread_state")
        return v

    @field_validator("confidence_tier")
    @classmethod
    def _validate_tier(cls, value: str) -> str:
        allowed = {"likely", "ambiguous", "conflicted"}
        v = value.strip().lower()
        if v not in allowed:
            raise ValueError("invalid confidence_tier")
        return v

    @field_validator("action")
    @classmethod
    def _validate_action(cls, value: str) -> str:
        allowed = {"create", "update", "ignore"}
        v = value.strip().lower()
        if v not in allowed:
            raise ValueError("invalid action")
        return v

    @field_validator("decision_confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0
        return max(0.0, min(1.0, confidence))

    @model_validator(mode="after")
    def _validate_schedule_invariants(self) -> "LlamaThreadDecisionPayload":
        if self.is_all_day:
            if self.event_date is None:
                raise ValueError("all-day requires event_date")
            if self.start_at is not None or self.end_at is not None:
                raise ValueError("all-day cannot include start_at/end_at")
        else:
            if self.action in {"create", "update"} and self.should_generate and self.event_date is None:
                raise ValueError("generated invites require event_date")
            if self.start_at is not None and self.end_at is not None and self.end_at <= self.start_at:
                raise ValueError("end_at must be after start_at")
        return self
