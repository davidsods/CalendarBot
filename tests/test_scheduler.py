from __future__ import annotations

import pytest

from app.config import settings
import app.scheduler as scheduler_module


class _FakeScheduler:
    def __init__(self, running: bool = False) -> None:
        self.running = running
        self.add_job_calls: list[dict[str, object]] = []
        self.start_calls = 0
        self.shutdown_calls: list[bool] = []

    def add_job(self, func, trigger: str, hours: int, id: str, replace_existing: bool) -> None:  # type: ignore[no-untyped-def]
        self.add_job_calls.append(
            {
                "func": func,
                "trigger": trigger,
                "hours": hours,
                "id": id,
                "replace_existing": replace_existing,
            }
        )

    def start(self) -> None:
        self.start_calls += 1
        self.running = True

    def shutdown(self, wait: bool) -> None:
        self.shutdown_calls.append(wait)
        self.running = False


def test_start_scheduler_adds_expected_job_and_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeScheduler(running=False)
    settings.processor_interval_hours = 7
    monkeypatch.setattr(scheduler_module, "scheduler", fake)

    scheduler_module.start_scheduler()

    assert len(fake.add_job_calls) == 1
    call = fake.add_job_calls[0]
    assert call["func"] == scheduler_module.process_job
    assert call["trigger"] == "interval"
    assert call["hours"] == 7
    assert call["id"] == "periodic_processor"
    assert call["replace_existing"] is True
    assert fake.start_calls == 1


def test_start_scheduler_is_idempotent_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeScheduler(running=True)
    monkeypatch.setattr(scheduler_module, "scheduler", fake)

    scheduler_module.start_scheduler()

    assert fake.add_job_calls == []
    assert fake.start_calls == 0


def test_stop_scheduler_only_shuts_down_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = _FakeScheduler(running=True)
    monkeypatch.setattr(scheduler_module, "scheduler", running)

    scheduler_module.stop_scheduler()

    assert running.shutdown_calls == [False]

    stopped = _FakeScheduler(running=False)
    monkeypatch.setattr(scheduler_module, "scheduler", stopped)

    scheduler_module.stop_scheduler()

    assert stopped.shutdown_calls == []


def test_process_job_runs_processor_and_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"run_once": 0, "commits": 0}

    class _FakeSession:
        def __enter__(self) -> "_FakeSession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            return None

        def commit(self) -> None:
            state["commits"] += 1

    class _FakeProcessorService:
        def __init__(self, session: _FakeSession) -> None:
            self.session = session

        def run_once(self) -> None:
            state["run_once"] += 1

    monkeypatch.setattr(scheduler_module, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(scheduler_module, "ProcessorService", _FakeProcessorService)

    scheduler_module.process_job()

    assert state["run_once"] == 1
    assert state["commits"] == 1
