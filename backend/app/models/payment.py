from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    razorpay_payment_id: Mapped[str | None] = mapped_column(
        String(64),
        unique=True,
        nullable=True,
    )
    razorpay_order_id: Mapped[str | None] = mapped_column(
        String(64),
        unique=True,
        nullable=True,
    )
    razorpay_subscription_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.id"),
        nullable=True,
    )
    amount: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(8), default="INR")
    status: Mapped[str] = mapped_column(String(32), index=True)
    method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
