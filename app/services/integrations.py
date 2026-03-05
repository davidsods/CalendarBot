from __future__ import annotations

import hashlib
import hmac
import json
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.models import GoogleOAuthToken


@dataclass
class SlackParsedAction:
    suggestion_id: int
    action: str
    edited_title: str | None = None


class SlackNotifier:
    def send_suggestion(self, suggestion_id: int, action: str) -> None:
        if not settings.slack_enabled or not settings.slack_bot_token or not settings.slack_channel_id:
            return

        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Suggestion #{suggestion_id} is ready for review. Proposed action: *{action}*.",
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": "approve",
                        "value": str(suggestion_id),
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "action_id": "reject",
                        "value": str(suggestion_id),
                    },
                ],
            },
        ]

        payload = {
            "channel": settings.slack_channel_id,
            "text": f"Suggestion #{suggestion_id}",
            "blocks": blocks,
        }

        req = urllib.request.Request(
            "https://slack.com/api/chat.postMessage",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.slack_bot_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if resp.status >= 300 or not body.get("ok", False):
                    err = body.get("error", "unknown_error")
                    raise RuntimeError(
                        f"failed to send Slack suggestion (status={resp.status}, error={err})"
                    )
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                detail = str(payload.get("error", "http_error"))
            except Exception:
                detail = "http_error"
            raise RuntimeError(
                f"failed to send Slack suggestion (status={exc.code}, error={detail})"
            ) from exc


class SlackActionParser:
    @staticmethod
    def verify_signature(raw_body: bytes, slack_timestamp: str, slack_signature: str) -> bool:
        if not settings.slack_signing_secret:
            return False

        now_ts = int(datetime.now(timezone.utc).timestamp())
        if abs(now_ts - int(slack_timestamp)) > 60 * 5:
            return False

        sig_basestring = f"v0:{slack_timestamp}:{raw_body.decode('utf-8')}".encode("utf-8")
        digest = hmac.new(
            settings.slack_signing_secret.encode("utf-8"),
            sig_basestring,
            hashlib.sha256,
        ).hexdigest()
        expected = f"v0={digest}"
        return hmac.compare_digest(expected, slack_signature)

    @staticmethod
    def parse_form_encoded_payload(raw_body: bytes) -> SlackParsedAction | None:
        parsed = urllib.parse.parse_qs(raw_body.decode("utf-8"))
        payload_raw = parsed.get("payload", [None])[0]
        if not payload_raw:
            return None

        payload = json.loads(payload_raw)
        action_obj = (payload.get("actions") or [{}])[0]
        action_id = action_obj.get("action_id")
        value = action_obj.get("value")
        if not value:
            return None

        if action_id == "reject":
            mapped = "reject"
        elif action_id == "approve":
            # Let approval service decide create vs update from suggestion state.
            mapped = "edit_then_approve"
        else:
            mapped = "edit_then_approve"

        return SlackParsedAction(suggestion_id=int(value), action=mapped)


class GoogleOAuthService:
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    def __init__(self, session: Session) -> None:
        self.session = session

    def _scopes(self) -> str:
        return settings.google_oauth_scopes

    def auth_url(self, state: str = "scheduler") -> str:
        if not settings.google_client_id or not settings.google_redirect_uri:
            raise RuntimeError("Google OAuth is not configured")

        query = urllib.parse.urlencode(
            {
                "client_id": settings.google_client_id,
                "redirect_uri": settings.google_redirect_uri,
                "response_type": "code",
                "scope": self._scopes(),
                "access_type": "offline",
                "prompt": "consent",
                "state": state,
            }
        )
        return f"{self.AUTH_URL}?{query}"

    def exchange_code(self, code: str, user_id: str = "default") -> GoogleOAuthToken:
        if not settings.google_client_id or not settings.google_client_secret or not settings.google_redirect_uri:
            raise RuntimeError("Google OAuth is not configured")

        data = urllib.parse.urlencode(
            {
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        token = self.session.query(GoogleOAuthToken).filter(GoogleOAuthToken.user_id == user_id).one_or_none()
        if token is None:
            token = GoogleOAuthToken(user_id=user_id, access_token="")
            self.session.add(token)

        token.access_token = payload["access_token"]
        token.refresh_token = payload.get("refresh_token", token.refresh_token)
        token.token_type = payload.get("token_type", "Bearer")
        token.scope = payload.get("scope", self._scopes())
        expires_in = payload.get("expires_in")
        token.expiry = (
            datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=int(expires_in))
            if expires_in
            else None
        )
        self.session.flush()
        return token

    def status(self, user_id: str = "default") -> dict[str, object]:
        token = self.session.query(GoogleOAuthToken).filter(GoogleOAuthToken.user_id == user_id).one_or_none()
        if not token:
            return {"connected": False, "has_refresh_token": False, "expiry": None}
        return {
            "connected": True,
            "has_refresh_token": bool(token.refresh_token),
            "expiry": token.expiry,
        }


class GoogleCalendarClient:
    BASE_URL = "https://www.googleapis.com/calendar/v3"

    def __init__(self, session: Session, user_id: str = "default") -> None:
        self.session = session
        self.user_id = user_id

    def _token_record(self) -> GoogleOAuthToken | None:
        return self.session.query(GoogleOAuthToken).filter(GoogleOAuthToken.user_id == self.user_id).one_or_none()

    def _refresh_if_needed(self, token: GoogleOAuthToken) -> GoogleOAuthToken:
        if token.expiry and token.expiry > datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=2):
            return token
        if not token.refresh_token:
            return token
        if not settings.google_client_id or not settings.google_client_secret:
            return token

        data = urllib.parse.urlencode(
            {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "refresh_token": token.refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            GoogleOAuthService.TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        token.access_token = payload.get("access_token", token.access_token)
        expires_in = payload.get("expires_in")
        if expires_in:
            token.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=int(expires_in))
        self.session.flush()
        return token

    def _authorized_request(self, method: str, url: str, body: dict[str, object]) -> dict[str, object] | None:
        token = self._token_record()
        if not token:
            return None
        token = self._refresh_if_needed(token)

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method=method,
            headers={
                "Authorization": f"Bearer {token.access_token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def create_event(self, title: str) -> str:
        url = f"{self.BASE_URL}/calendars/{urllib.parse.quote(settings.google_calendar_id, safe='')}/events"
        start_at = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(minutes=5)
        end_at = start_at + timedelta(hours=1)
        payload = {
            "summary": title,
            "start": {
                "dateTime": start_at.isoformat(),
                "timeZone": "UTC",
            },
            "end": {
                "dateTime": end_at.isoformat(),
                "timeZone": "UTC",
            },
        }
        response = self._authorized_request("POST", url, payload)
        if response and response.get("id"):
            return str(response["id"])

        # Fallback in local/dev when OAuth is not configured yet.
        ts = int(datetime.now(timezone.utc).timestamp())
        return f"gcal_{title.lower().replace(' ', '_')}_{ts}"

    def update_event(self, google_event_id: str, title: str) -> str:
        url = (
            f"{self.BASE_URL}/calendars/{urllib.parse.quote(settings.google_calendar_id, safe='')}/events/"
            f"{urllib.parse.quote(google_event_id, safe='')}"
        )
        payload = {"summary": title}
        self._authorized_request("PATCH", url, payload)
        return google_event_id
