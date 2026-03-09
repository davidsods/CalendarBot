from __future__ import annotations

from app.db import SessionLocal
from app.services.processor import ProcessorService


def run_once() -> None:
    with SessionLocal() as session:
        service = ProcessorService(session)
        service.run_once()
        session.commit()


if __name__ == "__main__":
    run_once()
