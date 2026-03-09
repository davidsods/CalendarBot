from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.config import settings
from app.services.extraction_utils import parse_schedule
from app.services.extractor import LlamaExtractor


def test_parse_schedule_defaults_to_pacific_for_time_when_tz_missing() -> None:
    reference = datetime(2026, 3, 9, 18, 0, tzinfo=timezone.utc)
    parsed = parse_schedule(
        "let's meet tomorrow at 3pm",
        reference_utc=reference,
        default_timezone="America/Los_Angeles",
    )

    assert parsed.timezone == "America/Los_Angeles"
    assert parsed.event_date is not None
    assert parsed.start_at_utc is not None
    start_local = parsed.start_at_utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/Los_Angeles"))
    assert start_local.hour == 15
    assert start_local.minute == 0


def test_parse_schedule_uses_explicit_message_timezone() -> None:
    reference = datetime(2026, 3, 9, 18, 0, tzinfo=timezone.utc)
    parsed = parse_schedule(
        "Mar 12 1:30pm ET works for me",
        reference_utc=reference,
        default_timezone="America/Los_Angeles",
    )

    assert parsed.timezone == "America/New_York"
    assert parsed.start_at_utc is not None
    start_local = parsed.start_at_utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York"))
    assert (start_local.hour, start_local.minute) == (13, 30)


def test_parse_schedule_time_range_and_day_only_cases() -> None:
    reference = datetime(2026, 3, 9, 18, 0, tzinfo=timezone.utc)
    ranged = parse_schedule(
        "3-4pm Friday",
        reference_utc=reference,
        default_timezone="America/Los_Angeles",
    )
    assert ranged.event_date is not None
    assert ranged.start_at_utc is not None
    assert ranged.end_at_utc is not None
    assert ranged.is_all_day is False

    day_only = parse_schedule(
        "Friday works",
        reference_utc=reference,
        default_timezone="America/Los_Angeles",
    )
    assert day_only.event_date is not None
    assert day_only.is_all_day is True
    assert day_only.start_at_utc is None
    assert day_only.end_at_utc is None


def test_parse_schedule_invalid_clock_text_does_not_crash() -> None:
    reference = datetime(2026, 3, 9, 18, 0, tzinfo=timezone.utc)
    parsed = parse_schedule(
        "Friday at 13pm",
        reference_utc=reference,
        default_timezone="America/Los_Angeles",
    )
    assert parsed.event_date is not None
    assert parsed.is_all_day is True
    assert parsed.start_at_utc is None
    assert parsed.end_at_utc is None


def test_extract_thread_includes_both_sides_in_summary() -> None:
    settings.ollama_base_url = None
    candidate = LlamaExtractor().extract_thread(
        messages=[
            {"id": 1, "sender_role": "other", "text": "can we do friday at 3pm?", "sent_at": datetime(2026, 3, 9, 18, 0)},
            {"id": 2, "sender_role": "self", "text": "yes that works for me", "sent_at": datetime(2026, 3, 9, 18, 1)},
        ],
        has_existing_thread_event=False,
        reference_utc=datetime(2026, 3, 9, 18, 1),
    )

    assert candidate.action in {"create", "update"}
    assert candidate.reason_summary is not None
    assert "Me:" in candidate.reason_summary
    assert "Them:" in candidate.reason_summary
    assert candidate.evidence_message_ids == [1, 2]
