from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.models import MessageRecord


class MaintenanceService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def purge_raw_messages(self) -> int:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=settings.raw_retention_hours)
        deleted = self.session.query(MessageRecord).filter(MessageRecord.created_at < cutoff).delete()
        self.session.flush()
        return int(deleted)
