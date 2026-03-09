from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import AuditLog


@dataclass
class DailyCostMetrics:
    day: str
    processed_threads: int = 0
    model_invocations: int = 0
    gated_skips: int = 0
    deferred_by_budget: int = 0
    estimated_model_spend_usd: float = 0.0


class CostSummaryService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def summarize(self, lookback_days: int = 30) -> tuple[list[DailyCostMetrics], DailyCostMetrics]:
        days = max(1, int(lookback_days))
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

        rows = self.session.scalars(
            select(AuditLog).where(AuditLog.created_at >= start, AuditLog.created_at <= end).order_by(AuditLog.created_at)
        ).all()

        by_day: dict[str, DailyCostMetrics] = {}
        for row in rows:
            day = row.created_at.date().isoformat()
            point = by_day.setdefault(day, DailyCostMetrics(day=day))
            payload = _parse_payload(row.payload)
            if row.event_type == "thread_processed":
                point.processed_threads += 1
            elif row.event_type == "llama_invoked":
                point.model_invocations += 1
            elif row.event_type == "llama_gate_skipped":
                point.gated_skips += 1
            elif row.event_type == "budget_cap_enforced":
                point.deferred_by_budget += int(payload.get("deferred", 0))

        points = [by_day[day] for day in sorted(by_day.keys())]
        for point in points:
            point.estimated_model_spend_usd = round(
                point.model_invocations * settings.estimated_llama_cost_per_message_usd,
                6,
            )

        totals = DailyCostMetrics(day="total")
        for point in points:
            totals.processed_threads += point.processed_threads
            totals.model_invocations += point.model_invocations
            totals.gated_skips += point.gated_skips
            totals.deferred_by_budget += point.deferred_by_budget
            totals.estimated_model_spend_usd += point.estimated_model_spend_usd
        totals.estimated_model_spend_usd = round(totals.estimated_model_spend_usd, 6)
        return points, totals


def _parse_payload(raw: str) -> dict[str, object]:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
