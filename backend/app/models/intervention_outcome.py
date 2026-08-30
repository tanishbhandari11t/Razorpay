from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class InterventionOutcome(Base):
    __tablename__ = "intervention_outcomes"
    __table_args__ = (
        Index(
            "ix_intervention_outcomes_state_window",
            "outcome_state",
            "observation_window_ends_at",
        ),
        Index(
            "ix_intervention_outcomes_payment_state",
            "payment_id",
            "outcome_state",
        ),
        Index(
            "ix_intervention_outcomes_evidence_mode",
            "data_source",
            "execution_mode",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    outcome_key: Mapped[str] = mapped_column(
        String(200),
        unique=True,
        index=True,
    )
    agent_decision_id: Mapped[str] = mapped_column(
        ForeignKey("agent_decisions.id"),
        unique=True,
        index=True,
    )
    intervention_id: Mapped[str | None] = mapped_column(
        ForeignKey("interventions.id"),
        nullable=True,
        index=True,
    )
    payment_id: Mapped[str] = mapped_column(
        ForeignKey("payments.id"),
        index=True,
    )
    recovery_case_id: Mapped[str] = mapped_column(
        ForeignKey("recovery_cases.id"),
        index=True,
    )
    action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_probability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    decision_margin: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    model_version: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(64))
    execution_mode: Mapped[str] = mapped_column(String(32))
    attempted: Mapped[bool] = mapped_column(Boolean, default=False)
    attempted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    failure_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    payment_status_after_24h: Mapped[str | None] = mapped_column(
        String(40),
        nullable=True,
    )
    payment_status_after_48h: Mapped[str | None] = mapped_column(
        String(40),
        nullable=True,
    )
    outcome_state: Mapped[str] = mapped_column(
        String(40),
        default="decided",
        index=True,
    )
    outcome_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    payment_recovered: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )
    recovered_amount_minor: Mapped[int] = mapped_column(Integer, default=0)
    recovery_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    time_to_recovery_seconds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    observation_window_starts_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True)
    )
    observation_window_ends_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True)
    )
    outcome_source: Mapped[str] = mapped_column(
        String(32),
        default="database",
    )
    data_source: Mapped[str] = mapped_column(String(32))
    natural_recovery_observed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    last_observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    state_history: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
