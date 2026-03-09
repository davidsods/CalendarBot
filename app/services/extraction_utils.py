from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import re
from zoneinfo import ZoneInfo

_INTENT_TOKENS = (
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
)
_UPDATE_TOKENS = ("move", "resched", "reschedule", "push", "delay", "instead")

_WEEKDAYS = {
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

_MONTHS = {
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

_TZ_ALIASES = {
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


def heuristic_extract(text: str, has_existing_thread_event: bool) -> tuple[str, str, float]:
    lowered = text.lower()
    if has_existing_thread_event and any(token in lowered for token in _UPDATE_TOKENS):
        return ("update", "Updated meeting", 0.8)
    if any(token in lowered for token in _INTENT_TOKENS) or _contains_date_or_time(lowered):
        return ("create", "Meeting", 0.7)
    return ("ignore", "", 0.1)


def detect_timezone(text: str, default_timezone: str) -> str:
    lowered = text.lower()
    for token, zone in _TZ_ALIASES.items():
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
        start_local_time = _parse_clock(range_match.group(1), range_match.group(2), range_match.group(3))
        end_meridiem = range_match.group(6) or range_match.group(3)
        end_local_time = _parse_clock(range_match.group(4), range_match.group(5), end_meridiem)
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
        start_local_time = _parse_clock(time_match.group(1), time_match.group(2), time_match.group(3))
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


def summarize_thread(messages: list[dict[str, object]]) -> str:
    if not messages:
        return "No thread context available."
    preview = []
    for msg in messages[-4:]:
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
    if re.search(r"\b(" + "|".join(_WEEKDAYS.keys()) + r")\b", text):
        return True
    if re.search(r"\b(" + "|".join(_MONTHS.keys()) + r")\s+\d{1,2}\b", text):
        return True
    return False


def _parse_date(text: str, reference_date: date) -> date | None:
    if "tomorrow" in text:
        return reference_date + timedelta(days=1)
    if "today" in text:
        return reference_date

    month_match = re.search(r"\b(" + "|".join(_MONTHS.keys()) + r")\s+(\d{1,2})(?:,\s*(\d{4}))?\b", text)
    if month_match:
        month = _MONTHS[month_match.group(1)]
        day = int(month_match.group(2))
        year = int(month_match.group(3)) if month_match.group(3) else reference_date.year
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if not month_match.group(3) and candidate < reference_date:
            return date(year + 1, month, day)
        return candidate

    weekday_match = re.search(r"\b(" + "|".join(_WEEKDAYS.keys()) + r")\b", text)
    if weekday_match:
        target = _WEEKDAYS[weekday_match.group(1)]
        days_ahead = (target - reference_date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return reference_date + timedelta(days=days_ahead)

    return None


def _parse_clock(hour_str: str, minute_str: str | None, meridiem: str) -> time:
    hour = int(hour_str)
    minute = int(minute_str or "0")
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return time(hour=hour, minute=minute)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=ZoneInfo("UTC"))
    return value.astimezone(ZoneInfo("UTC"))
