from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class Promise(Base):
    """Customer promise-to-pay — Churnkey/dunning-inspired lifecycle."""

    __tablename__ = "promises"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.id"),
        nullable=True,
        index=True,
    )
    payment_id: Mapped[str] = mapped_column(
        ForeignKey("payments.id"),
        index=True,
    )
    recovery_case_id: Mapped[str | None] = mapped_column(
        ForeignKey("recovery_cases.id"),
        nullable=True,
        index=True,
    )
    amount: Mapped[int] = mapped_column(Integer)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="merchant")
    language: Mapped[str] = mapped_column(String(32), default="hinglish")
    promised_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deadline: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
        index=True,
    )
    reminder_count: Mapped[int] = mapped_column(Integer, default=0)
    last_reminded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    fulfilled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
