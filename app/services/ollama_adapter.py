from __future__ import annotations

import json
import re
import urllib.request
from typing import Any
from datetime import date, datetime

from app.config import settings

_VALID_ACTIONS = {"create", "update", "ignore"}


class OllamaExtractorClient:
    def __init__(self) -> None:
        self.base_url = (settings.ollama_base_url or "").rstrip("/")
        self.model = settings.ollama_model
        self.api_key = settings.ollama_api_key
        self.timeout_seconds = settings.ollama_timeout_seconds

    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def extract(
        self,
        text: str,
        has_existing_thread_event: bool,
        allowed_actions: list[str] | None = None,
    ) -> dict[str, Any]:
        if not self.configured():
            raise RuntimeError("Ollama is not configured")

        allowed = [action for action in (allowed_actions or list(_VALID_ACTIONS)) if action in _VALID_ACTIONS]
        if not allowed:
            allowed = ["create", "update", "ignore"]

        system_prompt = (
            "You classify whether an iMessage should create, update, or ignore a calendar event. "
            "Return strict JSON with keys: action, title, confidence. "
            f"Action must be one of: {', '.join(allowed)}. "
            "confidence must be a number in [0,1]. "
            "Use an empty title for ignore."
        )
        user_payload = {
            "message_text": text,
            "has_existing_thread_event": has_existing_thread_event,
            "allowed_actions": allowed,
        }

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        parsed = self._parse_model_json(body)
        action = str(parsed.get("action", "ignore")).lower()
        if action not in allowed:
            action = "ignore"

        title = str(parsed.get("title", "")).strip()
        if action != "ignore" and not title:
            title = "Meeting"

        confidence = _coerce_confidence(parsed.get("confidence", 0.5))
        return {"action": action, "title": title, "confidence": confidence}

    def extract_thread_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.configured():
            raise RuntimeError("Ollama is not configured")

        system_prompt = (
            "You analyze conversation context to decide calendar invite readiness. "
            "Return strict JSON only. Required keys: "
            "should_generate, thread_state, confidence_tier, decision_confidence, action, title, "
            "event_date, start_at, end_at, is_all_day, timezone, recommended_slot_key, slot_candidates, "
            "decision_rationale, slack_summary, conflict_note, evidence_message_ids. "
            "thread_state must be one of exploring,candidate_slots,likely_consensus,confirmed,reschedule_pending,canceled. "
            "confidence_tier must be one of likely,ambiguous,conflicted. "
            "action must be create,update,ignore. "
            "Use ISO-8601 for datetimes and YYYY-MM-DD for dates. "
            "If invite should not be generated, set should_generate=false and action=ignore."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, default=_json_default)},
            ],
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return self._parse_model_json(body)

    @staticmethod
    def _parse_model_json(raw_response: dict[str, Any]) -> dict[str, Any]:
        content = ((raw_response.get("message") or {}).get("content") or "").strip()
        if not content:
            return {}
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.5
    return max(0.0, min(1.0, confidence))


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
