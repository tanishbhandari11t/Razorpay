from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.customer import Customer


def find_or_create_customer(
    session: Session,
    payment_entity: dict[str, Any],
) -> Customer | None:
    razorpay_customer_id = payment_entity.get("customer_id")
    email = payment_entity.get("email")
    phone = payment_entity.get("contact")

    customer = None
    if razorpay_customer_id:
        customer = session.scalar(
            select(Customer).where(
                Customer.razorpay_customer_id == str(razorpay_customer_id)
            )
        )
    if customer is None and email:
        customer = session.scalar(select(Customer).where(Customer.email == str(email)))
    if customer is not None:
        if phone and not customer.phone:
            customer.phone = str(phone)
        return customer
    if not any((razorpay_customer_id, email, phone)):
        return None

    customer = Customer(
        name=str(payment_entity.get("notes", {}).get("customer") or "Razorpay Customer"),
        email=str(email) if email else None,
        phone=str(phone) if phone else None,
        razorpay_customer_id=(
            str(razorpay_customer_id) if razorpay_customer_id else None
        ),
    )
    session.add(customer)
    session.flush()
    return customer
