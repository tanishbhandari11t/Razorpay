from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class AgentDecision(Base):
    __tablename__ = "agent_decisions"
    __table_args__ = (
        UniqueConstraint(
            "recovery_case_id",
            "policy_version",
            "execution_mode",
            name="uq_agent_decision_case_policy_mode",
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
    payment_id: Mapped[str] = mapped_column(
        ForeignKey("payments.id"),
        index=True,
    )
    decision_key: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(64))
    features_version: Mapped[str] = mapped_column(String(64))
    policy_manifest_sha256: Mapped[str] = mapped_column(String(64))
    execution_mode: Mapped[str] = mapped_column(
        String(32),
        default="shadow",
    )
    inference_status: Mapped[str] = mapped_column(
        String(32),
        default="completed",
    )
    decision_type: Mapped[str] = mapped_column(String(32))
    selected_action: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    candidate_actions: Mapped[dict] = mapped_column(JSON)
    predicted_probabilities: Mapped[dict] = mapped_column(JSON)
    expected_values: Mapped[dict] = mapped_column(JSON)
    features_snapshot: Mapped[dict] = mapped_column(JSON)
    decision_reasons: Mapped[list] = mapped_column(JSON)
    decision_margin: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    failure_class: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_checks: Mapped[dict] = mapped_column(JSON)
    risk_checks_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    dry_run: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
