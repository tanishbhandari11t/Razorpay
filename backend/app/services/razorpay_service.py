from __future__ import annotations

from typing import Any

import razorpay

from app.config.settings import get_settings


class RazorpayNotConfigured(RuntimeError):
    """Raised when local Test Mode credentials are missing."""


def _setting(name: str) -> str:
    settings = get_settings()
    values = {
        "RAZORPAY_KEY_ID": settings.razorpay_key_id,
        "RAZORPAY_KEY_SECRET": settings.razorpay_key_secret,
        "RAZORPAY_WEBHOOK_SECRET": settings.razorpay_webhook_secret,
    }
    return values[name].strip()


def key_id() -> str | None:
    value = _setting("RAZORPAY_KEY_ID")
    return value or None


def webhook_secret() -> str | None:
    value = _setting("RAZORPAY_WEBHOOK_SECRET")
    return value or None


def is_configured() -> bool:
    return bool(key_id() and _setting("RAZORPAY_KEY_SECRET"))


def get_client() -> razorpay.Client:
    public_key = key_id()
    secret = _setting("RAZORPAY_KEY_SECRET")
    if not public_key or not secret:
        raise RazorpayNotConfigured(
            "Razorpay Test Mode credentials are not configured in backend/.env."
        )
    if not public_key.startswith("rzp_test_"):
        raise RazorpayNotConfigured(
            "RecoverAI refuses non-test Razorpay keys during development."
        )
    return razorpay.Client(auth=(public_key, secret))


def test_connection() -> dict[str, Any]:
    return get_client().order.all({"count": 1})


def create_order(
    *,
    amount: int,
    currency: str,
    receipt: str,
    notes: dict[str, str],
) -> dict[str, Any]:
    return get_client().order.create(
        {
            "amount": amount,
            "currency": currency,
            "receipt": receipt,
            "notes": notes,
        }
    )


def create_plan(
    *,
    name: str,
    amount: int,
    currency: str = "INR",
) -> dict[str, Any]:
    return get_client().plan.create(
        {
            "period": "monthly",
            "interval": 1,
            "item": {
                "name": name,
                "amount": amount,
                "currency": currency,
                "description": "RecoverAI Test Mode subscription",
            },
        }
    )


def create_subscription(
    *,
    plan_id: str,
    total_count: int = 6,
) -> dict[str, Any]:
    return get_client().subscription.create(
        {
            "plan_id": plan_id,
            "total_count": total_count,
            "customer_notify": 1,
            "notes": {"source": "recoverai_test_mode"},
        }
    )


def verify_webhook(body: bytes, signature: str) -> None:
    secret = webhook_secret()
    if not secret:
        raise RazorpayNotConfigured(
            "RAZORPAY_WEBHOOK_SECRET is not configured in backend/.env."
        )
    get_client().utility.verify_webhook_signature(
        body.decode("utf-8"),
        signature,
        secret,
    )
