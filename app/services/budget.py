from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import BudgetState, BudgetStatus, QueueItem, QueueStatus


@dataclass
class BudgetDecision:
    allowed: bool
    will_be_capped: bool


class BudgetService:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _current_month() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _state_for_month(self, month: str) -> BudgetState | None:
        return self.session.scalar(select(BudgetState).where(BudgetState.month == month))

    def current_state(self) -> BudgetState:
        month = self._current_month()
        state = self._state_for_month(month)
        if state:
            return state

        state = BudgetState(
            month=month,
            spend_estimate_usd=0.0,
            cap_usd=settings.monthly_budget_cap_usd,
            status=BudgetStatus.ok,
        )
        self.session.add(state)
        self.session.flush()
        return state

    def refresh_month_and_requeue_if_needed(self) -> BudgetState:
        month = self._current_month()
        existing = self._state_for_month(month)
        if existing:
            return existing

        # Month rollover creates a fresh budget and re-opens previously deferred work.
        state = BudgetState(
            month=month,
            spend_estimate_usd=0.0,
            cap_usd=settings.monthly_budget_cap_usd,
            status=BudgetStatus.ok,
        )
        self.session.add(state)
        self.session.flush()

        self.session.query(QueueItem).filter(QueueItem.status == QueueStatus.deferred_budget_cap).update(
            {
                QueueItem.status: QueueStatus.pending,
                QueueItem.deferred_reason: None,
            },
            synchronize_session=False,
        )
        return state

    def can_claim_next_unit(self, estimated_unit_cost: float) -> BudgetDecision:
        state = self.current_state()
        if state.status == BudgetStatus.capped:
            return BudgetDecision(allowed=False, will_be_capped=True)

        projected = state.spend_estimate_usd + estimated_unit_cost + settings.budget_safety_buffer_usd
        will_cap = projected > state.cap_usd
        return BudgetDecision(allowed=not will_cap, will_be_capped=will_cap)

    def record_usage(self, amount_usd: float) -> BudgetState:
        state = self.current_state()
        state.spend_estimate_usd += amount_usd
        if state.spend_estimate_usd + settings.budget_safety_buffer_usd >= state.cap_usd:
            state.status = BudgetStatus.capped
        else:
            state.status = BudgetStatus.ok
        self.session.flush()
        return state

    def enforce_cap(self) -> int:
        """Defer all pending work when cap is reached."""
        state = self.current_state()
        state.status = BudgetStatus.capped

        updated = self.session.query(QueueItem).filter(QueueItem.status == QueueStatus.pending).update(
            {
                QueueItem.status: QueueStatus.deferred_budget_cap,
                QueueItem.deferred_reason: "monthly_budget_cap",
            },
            synchronize_session=False,
        )
        self.session.flush()
        return int(updated)

    def manual_reset_current_month(self) -> BudgetState:
        state = self.current_state()
        state.spend_estimate_usd = 0.0
        state.status = BudgetStatus.ok
        self.session.query(QueueItem).filter(QueueItem.status == QueueStatus.deferred_budget_cap).update(
            {
                QueueItem.status: QueueStatus.pending,
                QueueItem.deferred_reason: None,
            },
            synchronize_session=False,
        )
        self.session.flush()
        return state
