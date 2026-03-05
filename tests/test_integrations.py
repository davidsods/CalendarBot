from __future__ import annotations

import hashlib
import hmac
import json
import urllib.parse
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db import Base
from app.services.integrations import GoogleCalendarClient, SlackActionParser


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
