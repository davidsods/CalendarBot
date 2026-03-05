from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.main as app_main
from app.config import settings
from app.db import Base, get_session


@pytest.fixture(autouse=True)
def reset_settings() -> Iterator[None]:
    field_names = list(settings.__class__.model_fields.keys())
    original_values = {name: getattr(settings, name) for name in field_names}
    yield
    for name, value in original_values.items():
        setattr(settings, name, value)


@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(app_main, "start_scheduler", lambda: None)
    monkeypatch.setattr(app_main, "stop_scheduler", lambda: None)

    def _override_get_session() -> Iterator[Session]:
        yield db_session

    app_main.app.dependency_overrides[get_session] = _override_get_session
    try:
        with TestClient(app_main.app) as test_client:
            yield test_client
    finally:
        app_main.app.dependency_overrides.clear()
