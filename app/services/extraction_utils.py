from __future__ import annotations


def heuristic_extract(text: str, has_existing_thread_event: bool) -> tuple[str, str, float]:
    lowered = text.lower()
    if has_existing_thread_event and any(token in lowered for token in ["move", "resched", "push", "delay"]):
        return ("update", "Updated meeting", 0.8)
    if any(token in lowered for token in ["meet", "call", "appointment", "lunch", "tomorrow", "pm", "am"]):
        return ("create", "Meeting", 0.7)
    return ("ignore", "", 0.1)
