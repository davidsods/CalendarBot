from __future__ import annotations

from sqlalchemy import Engine, inspect, text


def run_startup_migrations(engine: Engine) -> None:
    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    if "event_suggestions" not in table_names:
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
    if "thread_state" not in existing_cols:
        ddl.append("ALTER TABLE event_suggestions ADD COLUMN thread_state VARCHAR(32)")
    if "confidence_tier" not in existing_cols:
        ddl.append("ALTER TABLE event_suggestions ADD COLUMN confidence_tier VARCHAR(20)")
    if "slot_candidate_id" not in existing_cols:
        ddl.append("ALTER TABLE event_suggestions ADD COLUMN slot_candidate_id INTEGER")

    with engine.begin() as conn:
        for statement in ddl:
            conn.execute(text(statement))

        if "thread_planning_states" not in table_names:
            conn.execute(
                text(
                    """
                    CREATE TABLE thread_planning_states (
                        id INTEGER PRIMARY KEY,
                        thread_id VARCHAR(255),
                        state VARCHAR(32),
                        summary TEXT,
                        recommended_slot_key VARCHAR(128),
                        decision_confidence FLOAT,
                        decision_rationale TEXT,
                        updated_at DATETIME,
                        created_at DATETIME
                    )
                    """
                )
            )
            conn.execute(text("CREATE UNIQUE INDEX uq_thread_planning_state_thread ON thread_planning_states(thread_id)"))
            conn.execute(text("CREATE INDEX ix_thread_planning_states_thread_id ON thread_planning_states(thread_id)"))
            conn.execute(text("CREATE INDEX ix_thread_planning_states_state ON thread_planning_states(state)"))

        if "thread_slot_candidates" not in table_names:
            conn.execute(
                text(
                    """
                    CREATE TABLE thread_slot_candidates (
                        id INTEGER PRIMARY KEY,
                        thread_id VARCHAR(255),
                        slot_key VARCHAR(128),
                        event_date DATE,
                        start_at DATETIME,
                        end_at DATETIME,
                        is_all_day BOOLEAN,
                        timezone VARCHAR(64),
                        title VARCHAR(255),
                        proposer_message_id INTEGER,
                        supporting_message_ids_json TEXT,
                        contradicting_message_ids_json TEXT,
                        score FLOAT,
                        recency_score FLOAT,
                        status VARCHAR(20),
                        version INTEGER,
                        last_evidence_at DATETIME,
                        created_at DATETIME,
                        updated_at DATETIME
                    )
                    """
                )
            )
            conn.execute(text("CREATE INDEX ix_thread_slot_candidates_thread_id ON thread_slot_candidates(thread_id)"))
            conn.execute(text("CREATE INDEX ix_thread_slot_candidates_slot_key ON thread_slot_candidates(slot_key)"))
            conn.execute(text("CREATE INDEX ix_thread_slot_candidates_status ON thread_slot_candidates(status)"))
            conn.execute(text("CREATE INDEX ix_thread_slot_candidates_score ON thread_slot_candidates(score)"))
