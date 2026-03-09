from __future__ import annotations

from sqlalchemy import Engine, inspect, text


def run_startup_migrations(engine: Engine) -> None:
    insp = inspect(engine)
    if "event_suggestions" not in insp.get_table_names():
        return

    existing_cols = {col["name"] for col in insp.get_columns("event_suggestions")}
    ddl: list[str] = []
    if "event_date" not in existing_cols:
        ddl.append("ALTER TABLE event_suggestions ADD COLUMN event_date DATE")
    if "is_all_day" not in existing_cols:
        ddl.append("ALTER TABLE event_suggestions ADD COLUMN is_all_day BOOLEAN")
    if "timezone" not in existing_cols:
        ddl.append("ALTER TABLE event_suggestions ADD COLUMN timezone VARCHAR(64)")
    if "reason_summary" not in existing_cols:
        ddl.append("ALTER TABLE event_suggestions ADD COLUMN reason_summary TEXT")
    if "evidence_message_ids_json" not in existing_cols:
        ddl.append("ALTER TABLE event_suggestions ADD COLUMN evidence_message_ids_json TEXT")
    if "context_window_size" not in existing_cols:
        ddl.append("ALTER TABLE event_suggestions ADD COLUMN context_window_size INTEGER")

    if not ddl:
        return

    with engine.begin() as conn:
        for statement in ddl:
            conn.execute(text(statement))
