from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import settings
from app.db import SessionLocal
from app.services.processor import ProcessorService

scheduler = BackgroundScheduler()


def process_job() -> None:
    with SessionLocal() as session:
        service = ProcessorService(session)
        service.run_once()
        session.commit()


def start_scheduler() -> None:
    if scheduler.running:
        return

    scheduler.add_job(
        process_job,
        "interval",
        hours=settings.processor_interval_hours,
        id="periodic_processor",
        replace_existing=True,
    )
    scheduler.start()


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
