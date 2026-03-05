from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from app.config import settings
from app.models import BudgetState, BudgetStatus, MessageRecord, QueueItem, QueueStatus
from app.services.budget import BudgetService


def _deferred_queue_item(session: Session, external_id: str = "msg-1") -> QueueItem:
    now = datetime.now(timezone.utc)
    msg = MessageRecord(
        external_message_id=external_id,
        batch_id="b1",
        thread_id="thread-1",
        sender_role="other",
        text="hello",
        sent_at=now,
        received_at=now,
    )
    session.add(msg)
    session.flush()

    item = QueueItem(
        message_id=msg.id,
        status=QueueStatus.deferred_budget_cap,
        deferred_reason="monthly_budget_cap",
    )
    session.add(item)
    session.flush()
    return item


def test_refresh_month_requeues_deferred_items(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_session.add(BudgetState(month="2026-02", spend_estimate_usd=10.0, cap_usd=15.0, status=BudgetStatus.capped))
    item = _deferred_queue_item(db_session)

    monkeypatch.setattr(BudgetService, "_current_month", staticmethod(lambda: "2026-03"))

    state = BudgetService(db_session).refresh_month_and_requeue_if_needed()
    db_session.expire_all()
    refreshed_item = db_session.get(QueueItem, item.id)

    assert state.month == "2026-03"
    assert state.status == BudgetStatus.ok
    assert refreshed_item is not None
    assert refreshed_item.status == QueueStatus.pending
    assert refreshed_item.deferred_reason is None


def test_manual_reset_current_month_resets_state_and_requeues(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _deferred_queue_item(db_session)
    monkeypatch.setattr(BudgetService, "_current_month", staticmethod(lambda: "2026-03"))

    svc = BudgetService(db_session)
    state = svc.current_state()
    state.spend_estimate_usd = 9.0
    state.status = BudgetStatus.capped
    db_session.flush()

    reset = svc.manual_reset_current_month()
    db_session.expire_all()
    refreshed_item = db_session.get(QueueItem, item.id)

    assert reset.spend_estimate_usd == 0.0
    assert reset.status == BudgetStatus.ok
    assert refreshed_item is not None
    assert refreshed_item.status == QueueStatus.pending
    assert refreshed_item.deferred_reason is None


def test_can_claim_next_unit_threshold_behavior(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.monthly_budget_cap_usd = 10.0
    settings.budget_safety_buffer_usd = 1.0
    monkeypatch.setattr(BudgetService, "_current_month", staticmethod(lambda: "2026-03"))

    svc = BudgetService(db_session)
    state = svc.current_state()
    state.spend_estimate_usd = 6.0
    db_session.flush()

    exact = svc.can_claim_next_unit(3.0)
    over = svc.can_claim_next_unit(3.01)

    assert exact.allowed is True
    assert exact.will_be_capped is False
    assert over.allowed is False
    assert over.will_be_capped is True


def test_record_usage_transitions_to_capped_at_cap_boundary(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.monthly_budget_cap_usd = 10.0
    settings.budget_safety_buffer_usd = 1.0
    monkeypatch.setattr(BudgetService, "_current_month", staticmethod(lambda: "2026-03"))

    svc = BudgetService(db_session)

    almost = svc.record_usage(8.9)
    assert almost.status == BudgetStatus.ok

    capped = svc.record_usage(0.1)
    assert capped.status == BudgetStatus.capped
