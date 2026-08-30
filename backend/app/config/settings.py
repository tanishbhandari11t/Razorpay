from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_URL = f"sqlite:///{(BACKEND_ROOT / 'data' / 'recoverai.db').as_posix()}"


class Settings(BaseSettings):
    razorpay_key_id: str = Field(default="", alias="RAZORPAY_KEY_ID")
    razorpay_key_secret: str = Field(default="", alias="RAZORPAY_KEY_SECRET")
    razorpay_webhook_secret: str = Field(default="", alias="RAZORPAY_WEBHOOK_SECRET")
    database_url: str = Field(default=DEFAULT_SQLITE_URL, alias="DATABASE_URL")
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
    )
    celery_task_always_eager: bool = Field(
        default=False,
        alias="CELERY_TASK_ALWAYS_EAGER",
    )
    execution_mode: str = Field(default="shadow", alias="EXECUTION_MODE")

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def default_blank_database_url(cls, value: object) -> object:
        return DEFAULT_SQLITE_URL if not str(value or "").strip() else value

    @field_validator("execution_mode")
    @classmethod
    def validate_execution_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"shadow", "dry_run", "controlled"}:
            raise ValueError("EXECUTION_MODE must be shadow, dry_run, or controlled")
        return normalized


def get_settings() -> Settings:
    """Read settings on demand so local .env edits are reflected immediately."""
    return Settings()
