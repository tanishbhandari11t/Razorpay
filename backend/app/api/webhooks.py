from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from app.services.razorpay_service import RazorpayNotConfigured, verify_webhook
from app.services.recovery_jobs import enqueue_recovery_case
from app.services.recovery_jobs_v2_online import (
    enqueue_v2_online_case,
    v2_online_automatic_enqueue_enabled,
)
from app.services.webhook_service import process_webhook


router = APIRouter(tags=["webhooks"])


@router.post("/api/webhooks/razorpay")
@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request) -> dict[str, object]:
    # Razorpay requires verification against the exact raw bytes. Do not call
    # request.json() or transform the payload before this check.
    body = await request.body()
    signature = request.headers.get("x-razorpay-signature", "").strip()
    event_id = request.headers.get("x-razorpay-event-id", "").strip()

    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Razorpay-Signature header.")
    if not event_id:
        raise HTTPException(status_code=400, detail="Missing X-Razorpay-Event-Id header.")

    try:
        verify_webhook(body, signature)
    except RazorpayNotConfigured as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=401, detail="Invalid webhook signature.") from error

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="Webhook body is not valid JSON.") from error

    try:
        result = process_webhook(event_id=event_id, payload=payload)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Webhook was verified but could not be processed.",
        ) from error

    recovery_case_id = result.get("recovery_case_id")
    queued = bool(recovery_case_id and not result.get("duplicate"))
    job: dict[str, object] | None = None
    v2_online_job: dict[str, object] | None = None
    if queued:
        job = enqueue_recovery_case(str(recovery_case_id))
        if v2_online_automatic_enqueue_enabled():
            v2_online_job = enqueue_v2_online_case(str(recovery_case_id))
    return {
        "status": "received",
        "shadow_inference_queued": queued,
        "recovery_job": job,
        "v2_online_shadow_job": v2_online_job,
        **result,
    }
