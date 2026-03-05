from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db import Base
from app.models import MessageRecord
from app.services.maintenance import MaintenanceService


def _session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    return factory()


def test_raw_message_retention_purge() -> None:
    settings.raw_retention_hours = 24
    session = _session()

    old = MessageRecord(
        external_message_id="old",
        batch_id="b1",
        thread_id="t1",
        sender_role="other",
        text="old",
        sent_at=datetime.now(timezone.utc),
        received_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=30),
    )
    new = MessageRecord(
        external_message_id="new",
        batch_id="b1",
        thread_id="t1",
        sender_role="other",
        text="new",
        sent_at=datetime.now(timezone.utc),
        received_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1),
    )
    session.add_all([old, new])
    session.commit()

    deleted = MaintenanceService(session).purge_raw_messages()
    session.commit()

    assert deleted == 1
    assert session.get(MessageRecord, new.id) is not None
