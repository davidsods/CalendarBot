# Scheduler Backend (V1)

Low-cost backend scaffold for iMessage -> Slack approval -> Google Calendar event creation.

## What is implemented

- `POST /v1/ingest/messages`
  - Deduplicates on `external_message_id`.
  - Enqueues only messages newer than the last successful processing checkpoint.
- Periodic processor (APScheduler)
  - Configurable interval via `PROCESSOR_INTERVAL_HOURS` (default `3`).
  - Processes all pending queue items until drained.
  - No fixed run timeout.
- Budget guardrail
  - Uses monthly estimated spend with cap (`MONTHLY_BUDGET_CAP_USD`, default `15`).
  - Safety buffer (`BUDGET_SAFETY_BUFFER_USD`, default `1`).
  - When cap risk is hit: completes in-flight item, defers remaining pending work to `deferred_budget_cap`.
- Suggestions and approvals
  - Creates `EventSuggestion` records with `create|update|ignore` action.
  - `POST /v1/slack/actions` supports approve/edit/reject flow and verifies Slack signatures for interactive callbacks.
  - Approved create writes/updates a `CalendarLink`.
- Real Llama extraction integration
  - `LLAMA_EXTRACT_URL` endpoint is called for structured extraction.
  - Automatic fallback to local heuristic extractor when unavailable.
- Real Google OAuth + Calendar integration
  - `GET /v1/google/oauth/start` generates consent URL.
  - `POST /v1/google/oauth/callback` exchanges code and stores tokens.
  - `GET /v1/google/oauth/status` returns connection status.
  - Calendar writes use stored OAuth tokens (with refresh flow).
- Retention
  - `POST /v1/maintenance/purge-raw` purges message records older than `RAW_RETENTION_HOURS` (default `24`).
- macOS relay agent
  - `relay/mac_relay.py` polls `~/Library/Messages/chat.db` and sends incremental batches.
  - Uses local checkpoint file to process only new messages.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload
```

In a second terminal, run the relay:

```bash
source .venv/bin/activate
python relay/mac_relay.py
```

## Key configuration

- `DATABASE_URL` (default `sqlite:///./scheduler.db`)
- `PROCESSOR_INTERVAL_HOURS` (default `3`)
- `MONTHLY_BUDGET_CAP_USD` (default `15`)
- `BUDGET_SAFETY_BUFFER_USD` (default `1`)
- `ESTIMATED_LLAMA_COST_PER_MESSAGE_USD` (default `0.02`)
- `RAW_RETENTION_HOURS` (default `24`)
- `LLAMA_EXTRACT_URL`, `LLAMA_API_KEY`, `LLAMA_TIMEOUT_SECONDS`
- `SLACK_ENABLED`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `SLACK_CHANNEL_ID`
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `GOOGLE_CALENDAR_ID`

## Tests

```bash
source .venv/bin/activate
pytest -q
```

Current test coverage validates:

- Budget cap defers remaining queue items.
- In-flight task completes before halt.
- Next cycles do not resume while capped.
- Checkpoint advances only after full drain.
- Reschedule messages generate update suggestions.
- Raw message purge works independently.
