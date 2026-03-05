from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from app.config import settings
from app.models import GoogleOAuthToken
from app.services.integrations import GoogleCalendarClient, GoogleOAuthService


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


def test_google_auth_url_contains_expected_query(db_session: Session) -> None:
    settings.google_client_id = "client-id"
    settings.google_redirect_uri = "https://example.com/callback"
    settings.google_oauth_scopes = "scope-a scope-b"

    url = GoogleOAuthService(db_session).auth_url(state="state-123")
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert query["client_id"] == ["client-id"]
    assert query["redirect_uri"] == ["https://example.com/callback"]
    assert query["state"] == ["state-123"]


def test_exchange_code_creates_token_record(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.google_client_id = "client-id"
    settings.google_client_secret = "client-secret"
    settings.google_redirect_uri = "https://example.com/callback"

    def _fake_urlopen(req: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
        assert req.full_url == GoogleOAuthService.TOKEN_URL
        assert timeout == 20
        return _FakeHTTPResponse(
            {
                "access_token": "access-1",
                "refresh_token": "refresh-1",
                "token_type": "Bearer",
                "scope": "scope-a",
                "expires_in": 3600,
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    token = GoogleOAuthService(db_session).exchange_code("code-1")

    assert token.access_token == "access-1"
    assert token.refresh_token == "refresh-1"
    assert token.expiry is not None


def test_exchange_code_preserves_existing_refresh_token_when_missing(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.google_client_id = "client-id"
    settings.google_client_secret = "client-secret"
    settings.google_redirect_uri = "https://example.com/callback"

    existing = GoogleOAuthToken(
        user_id="default",
        access_token="old-access",
        refresh_token="keep-me",
        token_type="Bearer",
    )
    db_session.add(existing)
    db_session.flush()

    def _fake_urlopen(req: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
        return _FakeHTTPResponse(
            {
                "access_token": "new-access",
                "token_type": "Bearer",
                "expires_in": 1200,
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    token = GoogleOAuthService(db_session).exchange_code("code-2")

    assert token.access_token == "new-access"
    assert token.refresh_token == "keep-me"


def test_refresh_if_needed_noop_when_token_is_fresh(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = GoogleOAuthToken(
        user_id="default",
        access_token="fresh-token",
        refresh_token="refresh-token",
        token_type="Bearer",
        expiry=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=30),
    )
    db_session.add(token)
    db_session.flush()

    def _fail_urlopen(req: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
        raise AssertionError("refresh should not be attempted for a fresh token")

    monkeypatch.setattr(urllib.request, "urlopen", _fail_urlopen)

    refreshed = GoogleCalendarClient(db_session)._refresh_if_needed(token)

    assert refreshed.access_token == "fresh-token"


def test_refresh_if_needed_updates_access_token_when_expired(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.google_client_id = "client-id"
    settings.google_client_secret = "client-secret"

    token = GoogleOAuthToken(
        user_id="default",
        access_token="old-access",
        refresh_token="refresh-token",
        token_type="Bearer",
        expiry=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=1),
    )
    db_session.add(token)
    db_session.flush()

    def _fake_urlopen(req: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
        return _FakeHTTPResponse(
            {
                "access_token": "new-access",
                "expires_in": 60,
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    refreshed = GoogleCalendarClient(db_session)._refresh_if_needed(token)

    assert refreshed.access_token == "new-access"
    assert refreshed.expiry is not None


def test_authorized_request_returns_none_when_no_token(db_session: Session) -> None:
    result = GoogleCalendarClient(db_session)._authorized_request("POST", "https://example.com", {"x": "y"})
    assert result is None


def test_create_event_falls_back_when_api_returns_no_id(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_authorized_request(
        self: GoogleCalendarClient,
        method: str,
        url: str,
        body: dict[str, object],
    ) -> dict[str, object] | None:
        return {}

    monkeypatch.setattr(GoogleCalendarClient, "_authorized_request", _fake_authorized_request)

    event_id = GoogleCalendarClient(db_session).create_event("Planning Call")

    assert event_id.startswith("gcal_planning_call_")


def test_update_event_returns_id_and_uses_patch(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def _fake_authorized_request(
        self: GoogleCalendarClient,
        method: str,
        url: str,
        body: dict[str, object],
    ) -> dict[str, object] | None:
        calls["method"] = method
        calls["url"] = url
        calls["body"] = body
        return {"id": "event-1"}

    monkeypatch.setattr(GoogleCalendarClient, "_authorized_request", _fake_authorized_request)

    result = GoogleCalendarClient(db_session).update_event("event-1", "Renamed")

    assert result == "event-1"
    assert calls["method"] == "PATCH"
    assert "event-1" in str(calls["url"])
    assert calls["body"] == {"summary": "Renamed"}
