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

    trigger_kwargs: dict[str, int] = {"hours": settings.processor_interval_hours}
    if settings.processor_interval_seconds and settings.processor_interval_seconds > 0:
        trigger_kwargs = {"seconds": settings.processor_interval_seconds}

    scheduler.add_job(
        process_job,
        "interval",
        **trigger_kwargs,
        id="periodic_processor",
        replace_existing=True,
    )
    scheduler.start()


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
