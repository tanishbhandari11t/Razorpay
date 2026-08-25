from __future__ import annotations

from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.database.connection import get_session
from app.services.payment_service import recent_payments, save_created_order
from app.services.razorpay_service import (
    RazorpayNotConfigured,
    create_order,
    create_plan,
    create_subscription,
    is_configured,
    key_id,
    test_connection,
    webhook_secret,
)


router = APIRouter(prefix="/api", tags=["payments"])


class CreateOrderRequest(BaseModel):
    amount: int = Field(default=249_900, ge=100, le=10_000_000)
    currency: Literal["INR"] = "INR"
    customer_name: str = Field(default="Amit", min_length=1, max_length=80)
    purpose: str = Field(
        default="subscription_recovery_test",
        min_length=1,
        max_length=120,
    )


class CreateTestSubscriptionRequest(BaseModel):
    amount: int = Field(default=249_900, ge=100, le=10_000_000)
    plan_name: str = Field(default="RecoverAI Pro", min_length=1, max_length=80)
    total_count: int = Field(default=6, ge=1, le=100)


def _gateway_error(error: Exception, message: str) -> HTTPException:
    if isinstance(error, RazorpayNotConfigured):
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(status_code=502, detail=message)


@router.get("/razorpay/status")
def razorpay_status() -> dict[str, str | bool | None]:
    configured = is_configured()
    public_key = key_id()
    return {
        "configured": configured,
        "webhook_configured": bool(webhook_secret()),
        "mode": "test" if public_key and public_key.startswith("rzp_test_") else None,
        "key_id": public_key if configured else None,
    }


@router.get("/razorpay/test")
def razorpay_test() -> dict[str, str | int | bool]:
    try:
        result = test_connection()
    except Exception as error:
        raise _gateway_error(
            error,
            "Razorpay Test API rejected the connection. Verify the test keys.",
        ) from error
    return {
        "success": True,
        "message": "Razorpay Test Mode connection working",
        "orders_returned": len(result.get("items", [])),
    }


@router.post("/payments/create-order")
def create_payment_order(request: CreateOrderRequest) -> dict[str, str | int]:
    receipt = f"recoverai_{uuid4().hex[:20]}"
    try:
        order = create_order(
            amount=request.amount,
            currency=request.currency,
            receipt=receipt,
            notes={
                "customer": request.customer_name,
                "purpose": request.purpose,
                "source": "recoverai_test_mode",
            },
        )
    except Exception as error:
        raise _gateway_error(
            error,
            "Razorpay could not create the Test Mode order.",
        ) from error

    with get_session() as session:
        save_created_order(session, order)
        session.commit()

    return {
        "id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "status": order.get("status", "created"),
        "receipt": receipt,
        "key_id": key_id() or "",
    }


@router.post("/subscriptions/create-test")
def create_test_subscription(
    request: CreateTestSubscriptionRequest,
) -> dict[str, str | int]:
    try:
        plan = create_plan(name=request.plan_name, amount=request.amount)
        subscription = create_subscription(
            plan_id=plan["id"],
            total_count=request.total_count,
        )
    except Exception as error:
        raise _gateway_error(
            error,
            "Razorpay could not create the Test Mode subscription.",
        ) from error

    return {
        "plan_id": plan["id"],
        "subscription_id": subscription["id"],
        "status": subscription.get("status", "created"),
        "amount": request.amount,
        "currency": "INR",
        "key_id": key_id() or "",
    }


@router.get("/payments/provider-states")
def provider_states(limit: int = 25) -> list[dict]:
    with get_session() as session:
        return recent_payments(session, max(1, min(limit, 100)))
