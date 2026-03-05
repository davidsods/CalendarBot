from __future__ import annotations

import hashlib
import hmac
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db import Base
from app.services.integrations import GoogleCalendarClient, SlackActionParser
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


def test_google_calendar_client_fallback_without_oauth_token() -> None:
    session = _session()
    client = GoogleCalendarClient(session)

    event_id = client.create_event("Planning Call")
    assert event_id.startswith("gcal_planning_call_")


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
