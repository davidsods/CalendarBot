from __future__ import annotations

import hashlib
import hmac
import json
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.config import settings
from app.models import GoogleOAuthToken


@dataclass
class SlackParsedAction:
    suggestion_id: int
    action: str
    edited_title: str | None = None
    response_url: str | None = None


class SlackNotifier:
    @staticmethod
    def _escape_mrkdwn(value: str) -> str:
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _single_line(value: str) -> str:
        return " ".join(value.split())

    @staticmethod
    def _truncate(value: str, limit: int = 180) -> str:
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    @staticmethod
    def _format_utc(value: datetime | None) -> str:
        if value is None:
            return "Not extracted"
        ts = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _time_preview(self, action: str, start_at: datetime | None, end_at: datetime | None) -> str:
        return self._time_preview_with_date(action, start_at, end_at, None, False, None)

    def _time_preview_with_date(
        self,
        action: str,
        start_at: datetime | None,
        end_at: datetime | None,
        event_date: date | None,
        is_all_day: bool,
        timezone_name: str | None,
    ) -> str:
        tz_label = timezone_name or "UTC"
        if is_all_day and event_date:
            return f"All-day on {event_date.isoformat()} ({tz_label})"
        if start_at and end_at:
            start = self._format_in_tz(start_at, timezone_name)
            end = self._format_in_tz(end_at, timezone_name)
            return f"{start} to {end}"
        if start_at:
            return f"Starts {self._format_in_tz(start_at, timezone_name)}"
        if event_date:
            return f"Date extracted ({event_date.isoformat()}) but time missing. Approval creates all-day."
        return "Not extracted"

    @staticmethod
    def _format_in_tz(value: datetime, timezone_name: str | None) -> str:
        tz = ZoneInfo(timezone_name or "UTC")
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        local = dt.astimezone(tz)
        return local.strftime("%Y-%m-%d %I:%M %p %Z")

    def send_suggestion(
        self,
        suggestion_id: int,
        action: str,
        title: str | None = None,
        thread_id: str | None = None,
        source_text: str | None = None,
        sent_at: datetime | None = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        event_date: date | None = None,
        is_all_day: bool = False,
        timezone_name: str | None = None,
        target_event_ref: str | None = None,
        confidence: float | None = None,
        thread_summary: str | None = None,
        evidence_messages: list[dict[str, object]] | None = None,
        confidence_tier: str | None = None,
        decision_rationale: str | None = None,
        conflict_note: str | None = None,
        decision_source: str | None = None,
    ) -> None:
        if not settings.slack_enabled or not settings.slack_bot_token or not settings.slack_channel_id:
            return

        safe_title = self._escape_mrkdwn(title or "Meeting")
        time_preview = self._escape_mrkdwn(
            self._time_preview_with_date(
                action=action,
                start_at=start_at,
                end_at=end_at,
                event_date=event_date,
                is_all_day=is_all_day,
                timezone_name=timezone_name,
            )
        )
        action_label = action.upper()

        fields: list[dict[str, str]] = [
            {"type": "mrkdwn", "text": f"*Suggestion ID:*\n#{suggestion_id}"},
            {"type": "mrkdwn", "text": f"*Action:*\n`{action_label}`"},
            {"type": "mrkdwn", "text": f"*Proposed title:*\n{safe_title}"},
            {"type": "mrkdwn", "text": f"*Proposed time:*\n{time_preview}"},
        ]
        if thread_id:
            fields.append({"type": "mrkdwn", "text": f"*Thread:*\n`{self._escape_mrkdwn(thread_id)}`"})
        if target_event_ref:
            fields.append({"type": "mrkdwn", "text": f"*Target event:*\n`{self._escape_mrkdwn(target_event_ref)}`"})
        if confidence is not None:
            fields.append({"type": "mrkdwn", "text": f"*Confidence:*\n{confidence:.2f}"})
        if confidence_tier:
            fields.append({"type": "mrkdwn", "text": f"*Confidence tier:*\n`{self._escape_mrkdwn(confidence_tier)}`"})
        if decision_source:
            fields.append({"type": "mrkdwn", "text": f"*Decision source:*\n`{self._escape_mrkdwn(decision_source)}`"})
        if sent_at is not None:
            fields.append({"type": "mrkdwn", "text": f"*Message sent:*\n{self._format_utc(sent_at)}"})
        if timezone_name:
            fields.append({"type": "mrkdwn", "text": f"*Timezone:*\n`{self._escape_mrkdwn(timezone_name)}`"})
        if event_date:
            fields.append({"type": "mrkdwn", "text": f"*Event date:*\n{event_date.isoformat()}"})
        fields.append({"type": "mrkdwn", "text": f"*All-day:*\n`{'yes' if is_all_day else 'no'}`"})

        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Calendar suggestion ready for approval*",
                },
            },
            {"type": "section", "fields": fields},
        ]
        if thread_summary:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Thread summary:*\n{self._escape_mrkdwn(thread_summary)}",
                    },
                }
            )
        if decision_rationale:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Why now:*\n{self._escape_mrkdwn(decision_rationale)}",
                    },
                }
            )
        if conflict_note:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Conflict note:*\n{self._escape_mrkdwn(conflict_note)}",
                    },
                }
            )
        if source_text:
            snippet = self._truncate(self._single_line(source_text))
            blocks.insert(
                3,
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Source message:*\n>{self._escape_mrkdwn(snippet)}",
                    },
                },
            )
        if evidence_messages:
            evidence_lines: list[str] = []
            for msg in evidence_messages[:6]:
                role = "Me" if msg.get("sender_role") == "self" else "Them"
                text = self._truncate(self._single_line(str(msg.get("text", ""))), limit=120)
                evidence_lines.append(f"*{role}:* {self._escape_mrkdwn(text)}")
            if evidence_lines:
                blocks.append(
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Evidence snippets:*\n" + "\n".join(evidence_lines),
                        },
                    }
                )

        blocks.append(
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
            }
        )

        payload = {
            "channel": settings.slack_channel_id,
            "text": f"Suggestion #{suggestion_id}: {action_label} {title or 'Meeting'}",
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

    @staticmethod
    def replace_interactive_message(response_url: str, text: str) -> None:
        payload = {"replace_original": True, "text": text}
        req = urllib.request.Request(
            response_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            return


class SlackActionParser:
    @staticmethod
    def verify_signature(raw_body: bytes, slack_timestamp: str, slack_signature: str) -> bool:
        if not settings.slack_signing_secret:
            return False

        try:
            request_ts = int(slack_timestamp)
        except (TypeError, ValueError):
            return False

        now_ts = int(datetime.now(timezone.utc).timestamp())
        if abs(now_ts - request_ts) > 60 * 5:
            return False

        try:
            body_text = raw_body.decode("utf-8")
        except UnicodeDecodeError:
            return False

        sig_basestring = f"v0:{slack_timestamp}:{body_text}".encode("utf-8")
        digest = hmac.new(
            settings.slack_signing_secret.encode("utf-8"),
            sig_basestring,
            hashlib.sha256,
        ).hexdigest()
        expected = f"v0={digest}"
        return hmac.compare_digest(expected, slack_signature)

    @staticmethod
    def parse_form_encoded_payload(raw_body: bytes) -> SlackParsedAction | None:
        try:
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

            return SlackParsedAction(
                suggestion_id=int(value),
                action=mapped,
                response_url=payload.get("response_url"),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return None


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

    @staticmethod
    def _as_aware_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _refresh_if_needed(self, token: GoogleOAuthToken) -> GoogleOAuthToken:
        expiry_utc = self._as_aware_utc(token.expiry)
        if expiry_utc and expiry_utc > datetime.now(timezone.utc) + timedelta(minutes=2):
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
            token.expiry = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
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

    def create_event(
        self,
        title: str,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        event_date: date | None = None,
        is_all_day: bool = False,
        timezone_name: str | None = None,
    ) -> str:
        url = f"{self.BASE_URL}/calendars/{urllib.parse.quote(settings.google_calendar_id, safe='')}/events"
        payload: dict[str, object] = {"summary": title}
        if is_all_day and event_date is not None:
            payload["start"] = {"date": event_date.isoformat()}
            payload["end"] = {"date": (event_date + timedelta(days=1)).isoformat()}
        elif start_at is not None:
            tz_name = timezone_name or settings.default_timezone
            tz = ZoneInfo(tz_name)
            start_aware = start_at.replace(tzinfo=timezone.utc) if start_at.tzinfo is None else start_at
            if end_at is None:
                end_at = (start_aware.astimezone(timezone.utc).replace(tzinfo=None) + timedelta(hours=1))
            end_aware = end_at.replace(tzinfo=timezone.utc) if end_at.tzinfo is None else end_at
            payload["start"] = {
                "dateTime": start_aware.astimezone(tz).isoformat(),
                "timeZone": tz_name,
            }
            payload["end"] = {
                "dateTime": end_aware.astimezone(tz).isoformat(),
                "timeZone": tz_name,
            }
        else:
            raise RuntimeError("missing_schedule")

        response = self._authorized_request("POST", url, payload)
        if response and response.get("id"):
            return str(response["id"])

        # Fallback in local/dev when OAuth is not configured yet.
        ts = int(datetime.now(timezone.utc).timestamp())
        return f"gcal_{title.lower().replace(' ', '_')}_{ts}"

    def update_event(
        self,
        google_event_id: str,
        title: str,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        event_date: date | None = None,
        is_all_day: bool = False,
        timezone_name: str | None = None,
    ) -> str:
        url = (
            f"{self.BASE_URL}/calendars/{urllib.parse.quote(settings.google_calendar_id, safe='')}/events/"
            f"{urllib.parse.quote(google_event_id, safe='')}"
        )
        payload: dict[str, object] = {"summary": title}
        if is_all_day and event_date is not None:
            payload["start"] = {"date": event_date.isoformat()}
            payload["end"] = {"date": (event_date + timedelta(days=1)).isoformat()}
        elif start_at is not None:
            tz_name = timezone_name or settings.default_timezone
            tz = ZoneInfo(tz_name)
            start_aware = start_at.replace(tzinfo=timezone.utc) if start_at.tzinfo is None else start_at
            if end_at is None:
                end_at = (start_aware.astimezone(timezone.utc).replace(tzinfo=None) + timedelta(hours=1))
            end_aware = end_at.replace(tzinfo=timezone.utc) if end_at.tzinfo is None else end_at
            payload["start"] = {
                "dateTime": start_aware.astimezone(tz).isoformat(),
                "timeZone": tz_name,
            }
            payload["end"] = {
                "dateTime": end_aware.astimezone(tz).isoformat(),
                "timeZone": tz_name,
            }
        self._authorized_request("PATCH", url, payload)
        return google_event_id
