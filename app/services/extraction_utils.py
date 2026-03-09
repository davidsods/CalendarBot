from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import re
from zoneinfo import ZoneInfo

INTENT_TOKENS = (
    "meet",
    "meeting",
    "call",
    "appointment",
    "lunch",
    "coffee",
    "sync",
    "review",
    "demo",
    "chat",
    "hang",
)
UPDATE_TOKENS = ("move", "resched", "reschedule", "push", "delay", "instead", "shift")
CANCEL_TOKENS = ("cancel", "canceled", "cancelled", "not happening", "skip")
ACCEPT_STRONG_TOKENS = (
    "works for me",
    "confirmed",
    "locked in",
    "sounds good",
    "yes that works",
    "perfect",
    "see you then",
)
ACCEPT_SOFT_TOKENS = ("should work", "probably works", "i think so", "i can do", "maybe works", "likely")
REJECT_TOKENS = ("can't", "cannot", "wont work", "won't work", "nope", "not free", "busy", "conflict")
UNCERTAIN_TOKENS = ("maybe", "not sure", "might", "possibly", "tentative")

WEEKDAYS = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

TZ_ALIASES = {
    "pt": "America/Los_Angeles",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "mt": "America/Denver",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "ct": "America/Chicago",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "et": "America/New_York",
    "est": "America/New_York",
    "edt": "America/New_York",
    "utc": "UTC",
    "gmt": "UTC",
}


@dataclass
class ParsedSchedule:
    event_date: date | None
    start_at_utc: datetime | None
    end_at_utc: datetime | None
    is_all_day: bool
    timezone: str
    has_explicit_time: bool


@dataclass
class MessageSignal:
    signal_type: str
    strength: float
    schedule: ParsedSchedule | None = None
    reason: str = ""


@dataclass
class SlotCandidate:
    slot_key: str
    event_date: date | None
    start_at: datetime | None
    end_at: datetime | None
    is_all_day: bool
    timezone: str
    title: str
    proposer_message_id: int | None
    supporting_message_ids: list[int] = field(default_factory=list)
    contradicting_message_ids: list[int] = field(default_factory=list)
    score: float = 0.0
    recency_score: float = 0.0
    last_evidence_at: datetime | None = None


@dataclass
class ThreadDecision:
    thread_state: str
    slot_candidates: list[SlotCandidate]
    recommended_slot_key: str | None
    decision_confidence: float
    confidence_tier: str
    decision_rationale: str
    evidence_message_ids: list[int]
    should_generate: bool
    conflict_note: str | None = None


def heuristic_extract(text: str, has_existing_thread_event: bool) -> tuple[str, str, float]:
    lowered = text.lower()
    if has_existing_thread_event and any(token in lowered for token in UPDATE_TOKENS):
        return ("update", "Updated meeting", 0.8)
    if any(token in lowered for token in INTENT_TOKENS) or _contains_date_or_time(lowered):
        return ("create", "Meeting", 0.7)
    return ("ignore", "", 0.1)


def detect_timezone(text: str, default_timezone: str) -> str:
    lowered = text.lower()
    for token, zone in TZ_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            return zone
    return default_timezone


def parse_schedule(
    text: str,
    reference_utc: datetime,
    default_timezone: str,
    default_duration_minutes: int = 60,
) -> ParsedSchedule:
    timezone_name = detect_timezone(text, default_timezone)
    tz = ZoneInfo(timezone_name)
    ref_local = _aware_utc(reference_utc).astimezone(tz)
    lowered = text.lower()

    parsed_date = _parse_date(lowered, ref_local.date())
    if parsed_date is None:
        return ParsedSchedule(
            event_date=None,
            start_at_utc=None,
            end_at_utc=None,
            is_all_day=False,
            timezone=timezone_name,
            has_explicit_time=False,
        )

    range_match = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*(?:-|–|to)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        lowered,
    )
    if range_match:
        try:
            start_local_time = _parse_clock(range_match.group(1), range_match.group(2), range_match.group(3))
            end_meridiem = range_match.group(6) or range_match.group(3)
            end_local_time = _parse_clock(range_match.group(4), range_match.group(5), end_meridiem)
        except ValueError:
            return ParsedSchedule(
                event_date=parsed_date,
                start_at_utc=None,
                end_at_utc=None,
                is_all_day=True,
                timezone=timezone_name,
                has_explicit_time=False,
            )
        start_local = datetime.combine(parsed_date, start_local_time, tzinfo=tz)
        end_local = datetime.combine(parsed_date, end_local_time, tzinfo=tz)
        if end_local <= start_local:
            end_local += timedelta(days=1)
        return ParsedSchedule(
            event_date=parsed_date,
            start_at_utc=start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            end_at_utc=end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            is_all_day=False,
            timezone=timezone_name,
            has_explicit_time=True,
        )

    time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lowered)
    if time_match:
        try:
            start_local_time = _parse_clock(time_match.group(1), time_match.group(2), time_match.group(3))
        except ValueError:
            return ParsedSchedule(
                event_date=parsed_date,
                start_at_utc=None,
                end_at_utc=None,
                is_all_day=True,
                timezone=timezone_name,
                has_explicit_time=False,
            )
        start_local = datetime.combine(parsed_date, start_local_time, tzinfo=tz)
        end_local = start_local + timedelta(minutes=default_duration_minutes)
        return ParsedSchedule(
            event_date=parsed_date,
            start_at_utc=start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            end_at_utc=end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            is_all_day=False,
            timezone=timezone_name,
            has_explicit_time=True,
        )

    return ParsedSchedule(
        event_date=parsed_date,
        start_at_utc=None,
        end_at_utc=None,
        is_all_day=True,
        timezone=timezone_name,
        has_explicit_time=False,
    )


def classify_signal(text: str, schedule: ParsedSchedule) -> MessageSignal:
    lowered = text.lower()
    if any(token in lowered for token in CANCEL_TOKENS):
        return MessageSignal(signal_type="cancel", strength=1.3, schedule=schedule, reason="cancel_token")
    if any(token in lowered for token in UPDATE_TOKENS):
        return MessageSignal(signal_type="reschedule", strength=0.6, schedule=schedule, reason="reschedule_token")
    if any(token in lowered for token in REJECT_TOKENS):
        return MessageSignal(signal_type="reject", strength=0.6, schedule=schedule, reason="reject_token")
    if any(token in lowered for token in ACCEPT_STRONG_TOKENS):
        return MessageSignal(signal_type="accept", strength=0.8, schedule=schedule, reason="accept_strong")
    if any(token in lowered for token in ACCEPT_SOFT_TOKENS):
        return MessageSignal(signal_type="accept", strength=0.35, schedule=schedule, reason="accept_soft")
    if any(token in lowered for token in UNCERTAIN_TOKENS):
        return MessageSignal(signal_type="uncertain", strength=0.25, schedule=schedule, reason="uncertain")
    if schedule.event_date is not None and (any(token in lowered for token in INTENT_TOKENS) or schedule.has_explicit_time):
        return MessageSignal(signal_type="proposal", strength=1.15, schedule=schedule, reason="proposal_with_schedule")
    if schedule.event_date is not None:
        return MessageSignal(signal_type="proposal", strength=0.9, schedule=schedule, reason="date_signal")
    return MessageSignal(signal_type="meta", strength=0.0, schedule=None, reason="meta")


def evaluate_thread_state(
    messages: list[dict[str, object]],
    default_timezone: str,
    existing_slots: list[SlotCandidate] | None = None,
) -> ThreadDecision:
    slots: dict[str, SlotCandidate] = {s.slot_key: s for s in (existing_slots or []) if s.slot_key}
    latest_slot_key: str | None = None
    canceled = False
    saw_reschedule = False
    evidence_ids: list[int] = []

    for idx, msg in enumerate(messages):
        text = str(msg.get("text", "") or "")
        sent_at_raw = msg.get("sent_at")
        sent_at = sent_at_raw if isinstance(sent_at_raw, datetime) else datetime.now()
        message_id = msg.get("id")
        msg_id = int(message_id) if isinstance(message_id, int) else None

        schedule = parse_schedule(text, sent_at, default_timezone)
        signal = classify_signal(text, schedule)
        if msg_id is not None and signal.signal_type != "meta":
            evidence_ids.append(msg_id)

        slot_key = _build_slot_key(schedule)
        target_key = slot_key or latest_slot_key

        if signal.signal_type == "reschedule" and slot_key and latest_slot_key and latest_slot_key in slots:
            if slot_key != latest_slot_key:
                old_slot = slots[latest_slot_key]
                old_slot.score -= signal.strength
                if msg_id is not None:
                    old_slot.contradicting_message_ids.append(msg_id)

                if slot_key not in slots and schedule.event_date is not None:
                    slots[slot_key] = SlotCandidate(
                        slot_key=slot_key,
                        event_date=schedule.event_date,
                        start_at=schedule.start_at_utc,
                        end_at=schedule.end_at_utc,
                        is_all_day=schedule.is_all_day,
                        timezone=schedule.timezone,
                        title="Meeting",
                        proposer_message_id=msg_id,
                    )
                new_slot = slots.get(slot_key)
                if new_slot:
                    new_slot.score += 0.95
                    new_slot.last_evidence_at = sent_at
                    new_slot.recency_score = max(new_slot.recency_score, float(idx + 1) / max(len(messages), 1))
                    if msg_id is not None:
                        new_slot.supporting_message_ids.append(msg_id)
                latest_slot_key = slot_key
                saw_reschedule = True
                continue

        if signal.signal_type == "cancel":
            canceled = True
            if target_key and target_key in slots and msg_id is not None:
                slots[target_key].contradicting_message_ids.append(msg_id)
                slots[target_key].score -= signal.strength
            continue

        if target_key and target_key not in slots and schedule.event_date is not None:
            slots[target_key] = SlotCandidate(
                slot_key=target_key,
                event_date=schedule.event_date,
                start_at=schedule.start_at_utc,
                end_at=schedule.end_at_utc,
                is_all_day=schedule.is_all_day,
                timezone=schedule.timezone,
                title="Meeting",
                proposer_message_id=msg_id,
            )

        if target_key and target_key in slots:
            slot = slots[target_key]
            slot.last_evidence_at = sent_at
            slot.recency_score = max(slot.recency_score, float(idx + 1) / max(len(messages), 1))

            if signal.signal_type == "proposal":
                slot.score += signal.strength
                if msg_id is not None:
                    slot.supporting_message_ids.append(msg_id)
                latest_slot_key = target_key
            elif signal.signal_type == "accept":
                slot.score += signal.strength
                if msg_id is not None:
                    slot.supporting_message_ids.append(msg_id)
            elif signal.signal_type == "uncertain":
                slot.score -= signal.strength
                if msg_id is not None:
                    slot.contradicting_message_ids.append(msg_id)
            elif signal.signal_type == "reject":
                slot.score -= signal.strength
                if msg_id is not None:
                    slot.contradicting_message_ids.append(msg_id)
            elif signal.signal_type == "reschedule":
                saw_reschedule = True
                slot.score -= signal.strength
                if msg_id is not None:
                    slot.contradicting_message_ids.append(msg_id)
                if slot_key and slot_key != target_key:
                    latest_slot_key = slot_key
        elif signal.signal_type == "reschedule":
            saw_reschedule = True

    ranked = sorted(slots.values(), key=lambda s: (s.score + 0.2 * s.recency_score), reverse=True)
    top = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None

    thread_state = "exploring"
    confidence = 0.0
    should_generate = False
    tier = "ambiguous"
    conflict_note: str | None = None
    recommended_slot_key = top.slot_key if top else None

    if canceled:
        thread_state = "canceled"
    elif top:
        top_score = top.score + 0.2 * top.recency_score
        confidence = max(0.0, min(1.0, top_score / 2.3))
        if second and abs((top.score + 0.2 * top.recency_score) - (second.score + 0.2 * second.recency_score)) < 0.2:
            tier = "conflicted"
            conflict_note = "Top slot is close to another candidate."
        elif top_score >= 1.1:
            tier = "likely"
        else:
            tier = "ambiguous"

        if top_score >= 1.7:
            thread_state = "confirmed"
            should_generate = True
        elif top_score >= 1.1:
            thread_state = "likely_consensus"
            should_generate = True
        elif saw_reschedule:
            thread_state = "reschedule_pending"
        else:
            thread_state = "candidate_slots"
    elif saw_reschedule:
        thread_state = "reschedule_pending"

    rationale = summarize_thread(messages)
    if top:
        rationale = f"{rationale} | Recommended slot score={top.score:.2f}, tier={tier}."
    if conflict_note:
        rationale = f"{rationale} {conflict_note}"

    return ThreadDecision(
        thread_state=thread_state,
        slot_candidates=ranked,
        recommended_slot_key=recommended_slot_key,
        decision_confidence=confidence,
        confidence_tier=tier,
        decision_rationale=rationale,
        evidence_message_ids=evidence_ids[-8:],
        should_generate=should_generate,
        conflict_note=conflict_note,
    )


def summarize_thread(messages: list[dict[str, object]]) -> str:
    if not messages:
        return "No thread context available."
    preview = []
    for msg in messages[-5:]:
        role = "Me" if msg.get("sender_role") == "self" else "Them"
        text = " ".join(str(msg.get("text", "")).split())
        if len(text) > 80:
            text = text[:77].rstrip() + "..."
        preview.append(f"{role}: {text}")
    return " | ".join(preview)


def _contains_date_or_time(text: str) -> bool:
    if "tomorrow" in text or "today" in text:
        return True
    if re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", text):
        return True
    if re.search(r"\b(" + "|".join(WEEKDAYS.keys()) + r")\b", text):
        return True
    if re.search(r"\b(" + "|".join(MONTHS.keys()) + r")\s+\d{1,2}\b", text):
        return True
    return False


def _parse_date(text: str, reference_date: date) -> date | None:
    if "tomorrow" in text:
        return reference_date + timedelta(days=1)
    if "today" in text:
        return reference_date

    month_match = re.search(r"\b(" + "|".join(MONTHS.keys()) + r")\s+(\d{1,2})(?:,\s*(\d{4}))?\b", text)
    if month_match:
        month = MONTHS[month_match.group(1)]
        day = int(month_match.group(2))
        year = int(month_match.group(3)) if month_match.group(3) else reference_date.year
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if not month_match.group(3) and candidate < reference_date:
            return date(year + 1, month, day)
        return candidate

    weekday_match = re.search(r"\b(" + "|".join(WEEKDAYS.keys()) + r")\b", text)
    if weekday_match:
        target = WEEKDAYS[weekday_match.group(1)]
        days_ahead = (target - reference_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return reference_date + timedelta(days=days_ahead)

    return None


def _parse_clock(hour_str: str, minute_str: str | None, meridiem: str) -> time:
    hour = int(hour_str)
    minute = int(minute_str or "0")
    if hour < 1 or hour > 12:
        raise ValueError("invalid 12-hour clock hour")
    if minute < 0 or minute > 59:
        raise ValueError("invalid minute")
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return time(hour=hour, minute=minute)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=ZoneInfo("UTC"))
    return value.astimezone(ZoneInfo("UTC"))


def _build_slot_key(schedule: ParsedSchedule) -> str | None:
    if schedule.event_date is None:
        return None
    if schedule.is_all_day:
        return f"{schedule.event_date.isoformat()}|all-day|{schedule.timezone}"
    if schedule.start_at_utc is not None:
        start = schedule.start_at_utc.isoformat()
        end = schedule.end_at_utc.isoformat() if schedule.end_at_utc else ""
        return f"{schedule.event_date.isoformat()}|{start}|{end}|{schedule.timezone}"
    return f"{schedule.event_date.isoformat()}|unknown|{schedule.timezone}"
