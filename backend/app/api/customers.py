from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.database.connection import get_session
from app.models.customer import Customer


router = APIRouter(prefix="/api/customers", tags=["customers"])


@router.get("")
def customers(limit: int = 25) -> list[dict]:
    with get_session() as session:
        rows = session.scalars(
            select(Customer).order_by(Customer.created_at.desc()).limit(
                max(1, min(limit, 100))
            )
        ).all()
        return [
            {
                "id": customer.id,
                "name": customer.name,
                "email": customer.email,
                "phone": customer.phone,
                "preferred_language": customer.preferred_language,
                "razorpay_customer_id": customer.razorpay_customer_id,
                "lifetime_value": customer.lifetime_value,
                "created_at": customer.created_at.isoformat(),
            }
            for customer in rows
        ]
