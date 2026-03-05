from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    database_url: str = Field(default="sqlite:///./scheduler.db", alias="DATABASE_URL")

    processor_interval_hours: int = Field(default=3, alias="PROCESSOR_INTERVAL_HOURS")
    monthly_budget_cap_usd: float = Field(default=15.0, alias="MONTHLY_BUDGET_CAP_USD")
    budget_safety_buffer_usd: float = Field(default=1.0, alias="BUDGET_SAFETY_BUFFER_USD")
    estimated_llama_cost_per_message_usd: float = Field(
        default=0.02,
        alias="ESTIMATED_LLAMA_COST_PER_MESSAGE_USD",
    )

    raw_retention_hours: int = Field(default=24, alias="RAW_RETENTION_HOURS")

    llama_extract_url: str | None = Field(default=None, alias="LLAMA_EXTRACT_URL")
    llama_api_key: str | None = Field(default=None, alias="LLAMA_API_KEY")
    llama_timeout_seconds: int = Field(default=30, alias="LLAMA_TIMEOUT_SECONDS")
    ollama_base_url: str | None = Field(default=None, alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2:1b", alias="OLLAMA_MODEL")
    ollama_api_key: str | None = Field(default=None, alias="OLLAMA_API_KEY")
    ollama_timeout_seconds: int = Field(default=30, alias="OLLAMA_TIMEOUT_SECONDS")

    slack_bot_token: str | None = Field(default=None, alias="SLACK_BOT_TOKEN")
    slack_signing_secret: str | None = Field(default=None, alias="SLACK_SIGNING_SECRET")
    slack_channel_id: str | None = Field(default=None, alias="SLACK_CHANNEL_ID")
    slack_enabled: bool = Field(default=False, alias="SLACK_ENABLED")

    google_client_id: str | None = Field(default=None, alias="GOOGLE_CLIENT_ID")
    google_client_secret: str | None = Field(default=None, alias="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str | None = Field(default=None, alias="GOOGLE_REDIRECT_URI")
    google_calendar_id: str = Field(default="primary", alias="GOOGLE_CALENDAR_ID")
    google_oauth_scopes: str = Field(
        default="https://www.googleapis.com/auth/calendar.events",
        alias="GOOGLE_OAUTH_SCOPES",
    )


settings = Settings()
