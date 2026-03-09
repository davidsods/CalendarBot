from __future__ import annotations

import hashlib
import hmac
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db import Base
from app.models import GoogleOAuthToken
from app.services.integrations import GoogleCalendarClient, SlackActionParser, SlackNotifier
from app.services.ollama_adapter import OllamaExtractorClient
from app.services.extractor import LlamaExtractor


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    return factory()


def test_slack_signature_and_payload_parser() -> None:
    settings.slack_signing_secret = "test-secret"
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    payload = {
        "response_url": "https://hooks.slack.com/actions/T/A/123",
        "actions": [
            {
                "action_id": "approve",
                "value": "42",
            }
        ]
    }
    form_body = urllib.parse.urlencode({"payload": json.dumps(payload)}).encode("utf-8")

    base = f"v0:{timestamp}:{form_body.decode('utf-8')}".encode("utf-8")
    digest = hmac.new(settings.slack_signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    signature = f"v0={digest}"

    assert SlackActionParser.verify_signature(form_body, timestamp, signature)

    parsed = SlackActionParser.parse_form_encoded_payload(form_body)
    assert parsed is not None
    assert parsed.suggestion_id == 42
    assert parsed.action == "edit_then_approve"
    assert parsed.response_url == "https://hooks.slack.com/actions/T/A/123"


def test_slack_notifier_replaces_interactive_message(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_urlopen(req: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))  # type: ignore[union-attr]
        return _FakeHTTPResponse({"ok": True})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    SlackNotifier.replace_interactive_message(
        "https://hooks.slack.com/actions/T/A/123",
        "Suggestion #42: approved.",
    )

    assert captured["url"] == "https://hooks.slack.com/actions/T/A/123"
    assert captured["timeout"] == 10
    assert captured["payload"] == {
        "replace_original": True,
        "text": "Suggestion #42: approved.",
    }


def test_slack_notifier_sends_calendar_preview(monkeypatch) -> None:
    orig_enabled = settings.slack_enabled
    orig_token = settings.slack_bot_token
    orig_channel = settings.slack_channel_id
    try:
        settings.slack_enabled = True
        settings.slack_bot_token = "xoxb-test"
        settings.slack_channel_id = "C123"

        captured: dict[str, object] = {}

        def _fake_urlopen(req: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
            assert req.full_url == "https://slack.com/api/chat.postMessage"
            assert timeout == 15
            captured["payload"] = json.loads(req.data.decode("utf-8"))  # type: ignore[union-attr]
            return _FakeHTTPResponse({"ok": True})

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

        SlackNotifier().send_suggestion(
            suggestion_id=12,
            action="create",
            title="Project Sync",
            thread_id="thread-123",
            source_text="Can we do Friday at 3pm for the launch sync?",
            sent_at=datetime(2026, 3, 4, 18, 30, tzinfo=timezone.utc),
            target_event_ref="evt_abc123",
            confidence=0.91,
        )

        payload = captured["payload"]
        assert isinstance(payload, dict)
        assert payload["channel"] == "C123"
        blocks_json = json.dumps(payload["blocks"])  # type: ignore[index]
        assert "Calendar suggestion ready for approval" in blocks_json
        assert "Proposed title" in blocks_json
        assert "Project Sync" in blocks_json
        assert "Source message" in blocks_json
        assert "thread-123" in blocks_json
        assert "evt_abc123" in blocks_json
        assert "default 1-hour hold" in blocks_json
    finally:
        settings.slack_enabled = orig_enabled
        settings.slack_bot_token = orig_token
        settings.slack_channel_id = orig_channel


def test_google_calendar_client_fallback_without_oauth_token() -> None:
    session = _session()
    client = GoogleCalendarClient(session)

    event_id = client.create_event("Planning Call")
    assert event_id.startswith("gcal_planning_call_")


def test_google_calendar_client_handles_aware_expiry_without_type_error() -> None:
    session = _session()
    token = GoogleOAuthToken(
        user_id="default",
        access_token="token",
        refresh_token=None,
        expiry=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    session.add(token)
    session.commit()

    client = GoogleCalendarClient(session)
    refreshed = client._refresh_if_needed(token)
    assert refreshed is token


def test_ollama_extractor_client_parses_response(monkeypatch) -> None:
    orig_base_url = settings.ollama_base_url
    orig_model = settings.ollama_model
    try:
        settings.ollama_base_url = "http://ollama.railway.internal:11434"
        settings.ollama_model = "llama3.2:1b"

        def _fake_urlopen(req: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
            assert req.full_url.endswith("/api/chat")
            assert timeout == settings.ollama_timeout_seconds
            body = json.loads(req.data.decode("utf-8"))  # type: ignore[union-attr]
            assert body["model"] == "llama3.2:1b"
            return _FakeHTTPResponse(
                {
                    "message": {
                        "content": json.dumps(
                            {"action": "create", "title": "Project sync", "confidence": 0.93}
                        )
                    }
                }
            )

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        result = OllamaExtractorClient().extract("Let's meet tomorrow", False, ["create", "update", "ignore"])
        assert result["action"] == "create"
        assert result["title"] == "Project sync"
        assert result["confidence"] == 0.93
    finally:
        settings.ollama_base_url = orig_base_url
        settings.ollama_model = orig_model


def test_llama_extractor_falls_back_to_heuristic_when_ollama_errors(monkeypatch) -> None:
    orig_llama_extract_url = settings.llama_extract_url
    orig_ollama_base_url = settings.ollama_base_url
    orig_ollama_model = settings.ollama_model
    try:
        settings.llama_extract_url = None
        settings.ollama_base_url = "http://ollama.railway.internal:11434"
        settings.ollama_model = "llama3.2:1b"

        def _raise_urlopen(req: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
            raise OSError("ollama unavailable")

        monkeypatch.setattr(urllib.request, "urlopen", _raise_urlopen)
        result = LlamaExtractor().extract("call tomorrow at 5pm", has_existing_thread_event=False)
        assert result.action == "create"
        assert result.title == "Meeting"
    finally:
        settings.llama_extract_url = orig_llama_extract_url
        settings.ollama_base_url = orig_ollama_base_url
        settings.ollama_model = orig_ollama_model
