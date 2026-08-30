from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.connection import Base


class PaymentFeatureContext(Base):
    __tablename__ = "payment_feature_contexts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    payment_id: Mapped[str] = mapped_column(
        ForeignKey("payments.id"),
        unique=True,
        index=True,
    )
    transaction_type: Mapped[str] = mapped_column(
        String(64),
        default="UNKNOWN",
    )
    merchant_category: Mapped[str] = mapped_column(
        String(120),
        default="UNKNOWN",
    )
    device_type: Mapped[str] = mapped_column(
        String(64),
        default="UNKNOWN",
    )
    network_type: Mapped[str] = mapped_column(
        String(64),
        default="UNKNOWN",
    )
    sender_age_group: Mapped[str] = mapped_column(
        String(64),
        default="UNKNOWN",
    )
    sender_state: Mapped[str] = mapped_column(
        String(120),
        default="UNKNOWN",
    )
    sender_bank: Mapped[str] = mapped_column(
        String(120),
        default="UNKNOWN",
    )
    fraud_flag: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(64), default="razorpay_webhook")
    unknown_fields: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
