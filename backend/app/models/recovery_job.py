from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class RecoveryJob(Base):
    __tablename__ = "recovery_jobs"
    __table_args__ = (
        UniqueConstraint(
            "recovery_case_id",
            "task_name",
            "policy_version",
            "execution_mode",
            name="uq_recovery_job_identity",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    recovery_case_id: Mapped[str] = mapped_column(
        ForeignKey("recovery_cases.id"),
        index=True,
    )
    job_key: Mapped[str] = mapped_column(
        String(200),
        unique=True,
        index=True,
    )
    task_name: Mapped[str] = mapped_column(
        String(80),
        default="shadow_inference",
    )
    model_version: Mapped[str] = mapped_column(
        String(64),
        default="recovery_model_v1",
    )
    policy_version: Mapped[str] = mapped_column(
        String(64),
        default="recovery_policy_v3",
    )
    execution_mode: Mapped[str] = mapped_column(
        String(32),
        default="shadow",
    )
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    celery_task_id: Mapped[str | None] = mapped_column(
        String(80),
        nullable=True,
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_class: Mapped[str | None] = mapped_column(
        String(160),
        nullable=True,
    )
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
