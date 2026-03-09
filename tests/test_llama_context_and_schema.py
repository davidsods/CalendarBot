from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import MessageRecord
from app.schemas import IngestMessage
from app.services.extractor import LlamaExtractor
from app.services.ingest import IngestService
from app.services.ollama_adapter import OllamaExtractorClient
from app.services.processor import ProcessorService


def test_build_llama_thread_context_includes_recent_and_anchor(db_session: Session) -> None:
    settings.thread_context_window_messages = 2
    settings.thread_anchor_window_messages = 3

    base = datetime(2026, 3, 9, 18, 0, tzinfo=timezone.utc)
    ingest = IngestService(db_session)
    ingest.ingest_batch(
        "b-ctx",
        [
            IngestMessage(
                external_message_id="ctx-1",
                thread_id="ctx-thread",
                sender_role="other",
                text="Can we do Friday at 2pm?",
                sent_at=base,
                received_at=base,
            ),
            IngestMessage(
                external_message_id="ctx-2",
                thread_id="ctx-thread",
                sender_role="self",
                text="maybe",
                sent_at=base + timedelta(minutes=1),
                received_at=base + timedelta(minutes=1),
            ),
            IngestMessage(
                external_message_id="ctx-3",
                thread_id="ctx-thread",
                sender_role="other",
                text="Move to 3pm instead",
                sent_at=base + timedelta(minutes=2),
                received_at=base + timedelta(minutes=2),
            ),
            IngestMessage(
                external_message_id="ctx-4",
                thread_id="ctx-thread",
                sender_role="self",
                text="works for me",
                sent_at=base + timedelta(minutes=3),
                received_at=base + timedelta(minutes=3),
            ),
        ],
    )
    db_session.commit()
    latest = db_session.scalar(select(MessageRecord).where(MessageRecord.external_message_id == "ctx-4"))
    assert latest is not None

    context = ProcessorService(db_session).build_llama_thread_context(
        thread_id="ctx-thread",
        upto_message=latest,
        has_existing_thread_event=False,
        target_event_ref=None,
        existing_slots=[],
    )

    assert context["context_ready"] is True
    recent = context["thread_messages_recent"]  # type: ignore[index]
    anchors = context["thread_messages_anchor"]  # type: ignore[index]
    assert isinstance(recent, list)
    assert isinstance(anchors, list)
    assert len(recent) == 2
    assert len(anchors) >= 1


def test_extract_thread_uses_llama_structured_decision(monkeypatch) -> None:
    settings.ollama_base_url = "http://ollama.local"
    settings.ollama_model = "llama3.2:1b"

    def _fake_thread_decision(self: OllamaExtractorClient, context: dict[str, object]) -> dict[str, object]:
        assert context["context_ready"] is True
        return {
            "should_generate": True,
            "thread_state": "likely_consensus",
            "confidence_tier": "likely",
            "decision_confidence": 0.84,
            "action": "create",
            "title": "Planning Sync",
            "event_date": "2026-03-12",
            "start_at": "2026-03-12T23:00:00Z",
            "end_at": "2026-03-13T00:00:00Z",
            "is_all_day": False,
            "timezone": "America/Los_Angeles",
            "recommended_slot_key": "2026-03-12|2026-03-12T23:00:00|2026-03-13T00:00:00|America/Los_Angeles",
            "slot_candidates": [
                {
                    "slot_key": "2026-03-12|2026-03-12T23:00:00|2026-03-13T00:00:00|America/Los_Angeles",
                    "event_date": "2026-03-12",
                    "start_at": "2026-03-12T23:00:00Z",
                    "end_at": "2026-03-13T00:00:00Z",
                    "is_all_day": False,
                    "timezone": "America/Los_Angeles",
                    "title": "Planning Sync",
                    "supporting_message_ids": [1, 2],
                    "contradicting_message_ids": [],
                    "score": 1.4,
                    "recency_score": 0.8,
                }
            ],
            "decision_rationale": "Participants converged on Thu 3pm PT.",
            "slack_summary": "Likely consensus reached.",
            "conflict_note": None,
            "evidence_message_ids": [1, 2],
        }

    monkeypatch.setattr(OllamaExtractorClient, "extract_thread_decision", _fake_thread_decision)

    candidate = LlamaExtractor().extract_thread(
        messages=[
            {"id": 1, "sender_role": "other", "text": "Thursday 3pm?", "sent_at": datetime(2026, 3, 9, 18, 0)},
            {"id": 2, "sender_role": "self", "text": "works for me", "sent_at": datetime(2026, 3, 9, 18, 1)},
        ],
        has_existing_thread_event=False,
        reference_utc=datetime(2026, 3, 9, 18, 1),
        existing_slots=[],
        llama_context={"context_ready": True, "context_version": "v1"},
    )

    assert candidate.decision_source == "llama"
    assert candidate.should_generate is True
    assert candidate.title == "Planning Sync"
    assert candidate.confidence_tier == "likely"


def test_extract_thread_falls_back_when_llama_payload_invalid(monkeypatch) -> None:
    settings.ollama_base_url = "http://ollama.local"
    settings.ollama_model = "llama3.2:1b"

    def _bad_thread_decision(self: OllamaExtractorClient, context: dict[str, object]) -> dict[str, object]:
        return {"oops": "invalid"}

    monkeypatch.setattr(OllamaExtractorClient, "extract_thread_decision", _bad_thread_decision)

    candidate = LlamaExtractor().extract_thread(
        messages=[
            {"id": 1, "sender_role": "other", "text": "Friday at 2pm", "sent_at": datetime(2026, 3, 9, 18, 0)},
            {"id": 2, "sender_role": "self", "text": "sounds good", "sent_at": datetime(2026, 3, 9, 18, 1)},
        ],
        has_existing_thread_event=False,
        reference_utc=datetime(2026, 3, 9, 18, 1),
        existing_slots=[],
        llama_context={"context_ready": True, "context_version": "v1"},
    )

    assert candidate.decision_source == "fallback"
    assert candidate.fallback_reason == "llama_invalid_output_or_error"


def test_extract_thread_normalizes_common_llama_aliases(monkeypatch) -> None:
    settings.ollama_base_url = "http://ollama.local"
    settings.ollama_model = "llama3.2:1b"

    def _alias_payload(self: OllamaExtractorClient, context: dict[str, object]) -> dict[str, object]:
        return {
            "should_generate": "true",
            "state": "likely_consensus",
            "confidence_label": "likely",
            "decision_confidence": "0.77",
            "decision_action": "create",
            "title": "Coffee Chat",
            "date": "2026-03-14",
            "start_time": "2026-03-14T22:00:00Z",
            "end_time": "2026-03-14T23:00:00Z",
            "timezone": "America/Los_Angeles",
            "recommendedSlotKey": "slot-1",
            "slotCandidates": [
                {
                    "slot_key": "slot-1",
                    "date": "2026-03-14",
                    "start_time": "2026-03-14T22:00:00Z",
                    "end_time": "2026-03-14T23:00:00Z",
                    "timezone": "America/Los_Angeles",
                    "title": "Coffee Chat",
                    "support_ids": ["1", 2],
                    "contradict_ids": ["9"],
                    "score": "1.2",
                    "recency_score": "0.5",
                }
            ],
            "rationale": "They converged on Saturday afternoon.",
            "summary": "Likely consensus",
            "evidence_ids": ["1", "2"],
        }

    monkeypatch.setattr(OllamaExtractorClient, "extract_thread_decision", _alias_payload)

    candidate = LlamaExtractor().extract_thread(
        messages=[
            {"id": 1, "sender_role": "other", "text": "Sat at 3?", "sent_at": datetime(2026, 3, 9, 18, 0)},
            {"id": 2, "sender_role": "self", "text": "works", "sent_at": datetime(2026, 3, 9, 18, 1)},
        ],
        has_existing_thread_event=False,
        reference_utc=datetime(2026, 3, 9, 18, 1),
        existing_slots=[],
        llama_context={"context_ready": True, "context_version": "v1", "default_timezone": "America/Los_Angeles"},
    )

    assert candidate.decision_source == "llama"
    assert candidate.should_generate is True
    assert candidate.confidence == 0.77
    assert candidate.timezone == "America/Los_Angeles"
    assert candidate.evidence_message_ids == [1, 2]
    assert candidate.slot_candidates
    assert candidate.slot_candidates[0].supporting_message_ids == [1, 2]
    assert candidate.slot_candidates[0].contradicting_message_ids == [9]


def test_extract_thread_llama_missing_date_downgrades_to_ignore(monkeypatch) -> None:
    settings.ollama_base_url = "http://ollama.local"
    settings.ollama_model = "llama3.2:1b"

    def _missing_date(self: OllamaExtractorClient, context: dict[str, object]) -> dict[str, object]:
        return {
            "should_generate": True,
            "thread_state": "candidate_slots",
            "confidence_tier": "ambiguous",
            "decision_confidence": 0.42,
            "action": "create",
            "title": "Catch up",
            "timezone": "America/Los_Angeles",
            "slot_candidates": [],
            "decision_rationale": "No clear date selected yet.",
            "evidence_message_ids": [1],
        }

    monkeypatch.setattr(OllamaExtractorClient, "extract_thread_decision", _missing_date)

    candidate = LlamaExtractor().extract_thread(
        messages=[
            {"id": 1, "sender_role": "other", "text": "we should meet sometime", "sent_at": datetime(2026, 3, 9, 18, 0)},
        ],
        has_existing_thread_event=False,
        reference_utc=datetime(2026, 3, 9, 18, 1),
        existing_slots=[],
        llama_context={"context_ready": True, "context_version": "v1", "default_timezone": "America/Los_Angeles"},
    )

    assert candidate.decision_source == "llama"
    assert candidate.should_generate is False
    assert candidate.action == "ignore"
