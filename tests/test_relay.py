from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "relay" / "mac_relay.py"
SPEC = importlib.util.spec_from_file_location("mac_relay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mac_relay = importlib.util.module_from_spec(SPEC)
sys.modules["mac_relay"] = mac_relay
SPEC.loader.exec_module(mac_relay)


def test_apple_ns_to_iso_known_conversion() -> None:
    # 1 second after Apple epoch (2001-01-01T00:00:00Z).
    iso_time = mac_relay.apple_ns_to_iso(1_000_000_000)
    assert iso_time.startswith("2001-01-01T00:00:01")


def test_load_checkpoint_handles_missing_and_corrupt_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    monkeypatch.setattr(mac_relay, "CHECKPOINT_FILE", checkpoint)

    assert mac_relay.load_checkpoint() == 0

    checkpoint.write_text("{not-json")
    assert mac_relay.load_checkpoint() == 0


def test_save_and_load_checkpoint_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    monkeypatch.setattr(mac_relay, "CHECKPOINT_FILE", checkpoint)

    mac_relay.save_checkpoint(42)

    assert checkpoint.exists()
    assert json.loads(checkpoint.read_text())["last_message_rowid"] == 42
    assert mac_relay.load_checkpoint() == 42


def test_fetch_new_messages_reads_and_maps_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chat.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
    conn.execute(
        """
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            guid TEXT,
            handle_id INTEGER,
            is_from_me INTEGER,
            text TEXT,
            date INTEGER
        )
        """
    )
    conn.execute("INSERT INTO handle(ROWID, id) VALUES(1, 'thread-user')")
    conn.execute(
        "INSERT INTO message(ROWID, guid, handle_id, is_from_me, text, date) VALUES(10, 'guid-10', 1, 0, 'hello', 1000000000)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(mac_relay, "MESSAGES_DB", db_path)
    monkeypatch.setattr(mac_relay, "BATCH_SIZE", 100)

    messages, max_rowid = mac_relay.fetch_new_messages(0)

    assert max_rowid == 10
    assert len(messages) == 1
    assert messages[0].external_message_id == "guid-10"
    assert messages[0].thread_id == "thread-user"
    assert messages[0].sender_role == "other"
    assert messages[0].text == "hello"


def test_fetch_new_messages_returns_empty_when_db_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.db"
    monkeypatch.setattr(mac_relay, "MESSAGES_DB", missing)

    messages, max_rowid = mac_relay.fetch_new_messages(15)

    assert messages == []
    assert max_rowid == 15


def test_post_batch_raises_on_non_2xx_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        status = 500

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            return None

    def _fake_urlopen(req: urllib.request.Request, timeout: int):  # type: ignore[no-untyped-def]
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError, match="ingest request failed"):
        mac_relay.post_batch(
            [
                mac_relay.RelayMessage(
                    external_message_id="m1",
                    thread_id="t1",
                    sender_role="other",
                    text="hello",
                    sent_at="2026-03-01T00:00:00+00:00",
                    received_at="2026-03-01T00:00:00+00:00",
                )
            ]
        )
