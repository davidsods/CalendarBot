from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from app.config import settings
from app.services.extraction_utils import heuristic_extract
from app.services.ollama_adapter import OllamaExtractorClient


@dataclass
class ExtractedCandidate:
    action: str
    title: str
    confidence: float


class LlamaExtractor:
    def _heuristic(self, text: str, has_existing_thread_event: bool) -> ExtractedCandidate:
        action, title, confidence = heuristic_extract(text, has_existing_thread_event)
        return ExtractedCandidate(action=action, title=title, confidence=confidence)

    def extract(self, text: str, has_existing_thread_event: bool) -> ExtractedCandidate:
        if not settings.llama_extract_url:
            # Prefer direct Ollama integration when configured (common Railway setup).
            client = OllamaExtractorClient()
            if client.configured():
                try:
                    result = client.extract(
                        text=text,
                        has_existing_thread_event=has_existing_thread_event,
                        allowed_actions=["create", "update", "ignore"],
                    )
                    return ExtractedCandidate(
                        action=str(result["action"]),
                        title=str(result["title"]),
                        confidence=float(result["confidence"]),
                    )
                except Exception:
                    return self._heuristic(text, has_existing_thread_event)

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
