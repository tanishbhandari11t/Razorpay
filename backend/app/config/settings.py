from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_SQLITE_URL = f"sqlite:///{(BACKEND_ROOT / 'data' / 'recoverai.db').as_posix()}"
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:8010",
    "http://127.0.0.1:8010",
)
DEFAULT_CORS_ORIGIN_REGEX = (
    r"https?://(localhost|127\.0\.0\.1|100\.\d+\.\d+\.\d+)(:\d+)?"
)


def normalize_database_url(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return DEFAULT_SQLITE_URL
    if raw.startswith("sqlite:"):
        return raw
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://") :]
    if raw.startswith("postgresql://"):
        raw = "postgresql+psycopg2://" + raw[len("postgresql://") :]
    return raw


def parse_cors_origins(value: str) -> list[str]:
    extras = [
        part.strip().rstrip("/")
        for part in value.replace(";", ",").split(",")
        if part.strip()
    ]
    return list(dict.fromkeys([*DEFAULT_CORS_ORIGINS, *extras]))


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
    cors_origins: str = Field(default="", alias="CORS_ORIGINS")
    cors_origin_regex: str = Field(
        default=DEFAULT_CORS_ORIGIN_REGEX,
        alias="CORS_ORIGIN_REGEX",
    )

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def default_blank_database_url(cls, value: object) -> object:
        return normalize_database_url(value)

    @field_validator("execution_mode")
    @classmethod
    def validate_execution_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"shadow", "dry_run", "controlled"}:
            raise ValueError("EXECUTION_MODE must be shadow, dry_run, or controlled")
        return normalized

    @property
    def cors_origin_list(self) -> list[str]:
        return parse_cors_origins(self.cors_origins)


def get_settings() -> Settings:
    """Read settings on demand so local .env edits are reflected immediately."""
    return Settings()
