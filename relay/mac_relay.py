#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

API_URL = os.getenv("INGEST_URL", "http://localhost:8000/v1/ingest/messages")
DEVICE_ID = os.getenv("DEVICE_ID", "mac-relay-local")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "300"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
CHECKPOINT_FILE = Path(os.getenv("CHECKPOINT_FILE", str(Path.home() / ".scheduler_relay_checkpoint.json")))
MESSAGES_DB = Path(os.getenv("MESSAGES_DB", str(Path.home() / "Library/Messages/chat.db")))


@dataclass
class RelayMessage:
    external_message_id: str
    thread_id: str
    sender_role: str
    text: str
    sent_at: str
    received_at: str


def apple_ns_to_iso(nanos: int | None) -> str:
    # Apple epoch starts 2001-01-01.
    if not nanos:
        return datetime.now(timezone.utc).isoformat()
    seconds = nanos / 1_000_000_000
    unix_seconds = seconds + 978307200
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat()


def load_checkpoint() -> int:
    if not CHECKPOINT_FILE.exists():
        return 0
    try:
        data = json.loads(CHECKPOINT_FILE.read_text())
        return int(data.get("last_message_rowid", 0))
    except (ValueError, json.JSONDecodeError):
        return 0


def save_checkpoint(rowid: int) -> None:
    CHECKPOINT_FILE.write_text(json.dumps({"last_message_rowid": rowid}))


def fetch_new_messages(last_rowid: int) -> tuple[list[RelayMessage], int]:
    if not MESSAGES_DB.exists():
        return [], last_rowid

    conn = sqlite3.connect(f"file:{MESSAGES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            m.ROWID as rowid,
            m.guid as guid,
            COALESCE(h.id, 'unknown-thread') as thread_id,
            COALESCE(m.is_from_me, 0) as is_from_me,
            COALESCE(m.text, '') as text,
            m.date as msg_date
        FROM message m
        LEFT JOIN handle h ON h.ROWID = m.handle_id
        WHERE m.ROWID > ?
          AND m.text IS NOT NULL
          AND LENGTH(TRIM(m.text)) > 0
        ORDER BY m.ROWID ASC
        LIMIT ?
    """

    rows = conn.execute(query, (last_rowid, BATCH_SIZE)).fetchall()
    conn.close()

    messages: list[RelayMessage] = []
    max_rowid = last_rowid

    for row in rows:
        rowid = int(row["rowid"])
        max_rowid = max(max_rowid, rowid)
        iso_time = apple_ns_to_iso(row["msg_date"])
        messages.append(
            RelayMessage(
                external_message_id=row["guid"] or f"rowid-{rowid}",
                thread_id=row["thread_id"],
                sender_role="self" if int(row["is_from_me"] or 0) == 1 else "other",
                text=row["text"],
                sent_at=iso_time,
                received_at=iso_time,
            )
        )

    return messages, max_rowid


def post_batch(messages: list[RelayMessage]) -> None:
    payload: dict[str, Any] = {
        "device_id": DEVICE_ID,
        "batch_id": f"{DEVICE_ID}-{int(time.time())}",
        "messages": [m.__dict__ for m in messages],
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"ingest request failed with status {resp.status}")


def run_forever() -> None:
    while True:
        last = load_checkpoint()
        messages, max_rowid = fetch_new_messages(last)
        if messages:
            post_batch(messages)
            save_checkpoint(max_rowid)
            print(f"sent {len(messages)} messages, checkpoint={max_rowid}")
        else:
            print("no new messages")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_forever()
