from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class Intervention(Base):
    __tablename__ = "interventions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    payment_id: Mapped[str] = mapped_column(ForeignKey("payments.id"), index=True)
    agent_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("agent_decisions.id"),
        nullable=True,
        index=True,
    )
    type: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32))
    cost: Mapped[Decimal] = mapped_column(Numeric(12, 2), default=Decimal("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
