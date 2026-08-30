from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class OutcomeObservation(Base):
    __tablename__ = "outcome_observations"
    __table_args__ = (
        UniqueConstraint(
            "intervention_outcome_id",
            "observation_source",
            "external_ref",
            name="uq_outcome_observation_evidence",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    intervention_outcome_id: Mapped[str] = mapped_column(
        ForeignKey("intervention_outcomes.id"),
        index=True,
    )
    observation_source: Mapped[str] = mapped_column(String(32))
    external_ref: Mapped[str] = mapped_column(String(160))
    payment_status: Mapped[str] = mapped_column(String(40))
    recovered_signal: Mapped[bool] = mapped_column(Boolean, default=False)
    attribution_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
