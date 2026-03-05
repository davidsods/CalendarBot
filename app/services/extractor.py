from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from app.config import settings


@dataclass
class ExtractedCandidate:
    action: str
    title: str
    confidence: float


class LlamaExtractor:
    def _heuristic(self, text: str, has_existing_thread_event: bool) -> ExtractedCandidate:
        lowered = text.lower()
        if has_existing_thread_event and any(token in lowered for token in ["move", "resched", "push", "delay"]):
            return ExtractedCandidate(action="update", title="Updated meeting", confidence=0.8)
        if any(token in lowered for token in ["meet", "call", "appointment", "lunch", "tomorrow", "pm", "am"]):
            return ExtractedCandidate(action="create", title="Meeting", confidence=0.7)
        return ExtractedCandidate(action="ignore", title="", confidence=0.1)

    def extract(self, text: str, has_existing_thread_event: bool) -> ExtractedCandidate:
        if not settings.llama_extract_url:
            return self._heuristic(text, has_existing_thread_event)

        payload = {
            "message_text": text,
            "has_existing_thread_event": has_existing_thread_event,
            "allowed_actions": ["create", "update", "ignore"],
        }

        headers = {"Content-Type": "application/json"}
        if settings.llama_api_key:
            headers["Authorization"] = f"Bearer {settings.llama_api_key}"

        req = urllib.request.Request(
            settings.llama_extract_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=settings.llama_timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                action = str(body.get("action", "ignore"))
                title = str(body.get("title", "Meeting"))
                confidence = float(body.get("confidence", 0.5))

                if action not in {"create", "update", "ignore"}:
                    return self._heuristic(text, has_existing_thread_event)
                return ExtractedCandidate(action=action, title=title, confidence=confidence)
        except Exception:
            return self._heuristic(text, has_existing_thread_event)
