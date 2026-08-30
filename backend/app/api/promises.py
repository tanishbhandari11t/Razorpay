from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.database.connection import get_session
from app.services.automation_state import automation_status, set_recover_revenue
from app.services.campaign_engine import campaign_overview
from app.services.promise_service import (
    create_promise,
    list_promises,
    promise_summary,
    seed_demo_promises,
)


router = APIRouter(prefix="/api/recovery", tags=["promises-automation"])


class PromiseCreateRequest(BaseModel):
    case_id: str | None = None
    payment_id: str | None = None
    days: int = Field(default=1, ge=1, le=14)
    note: str | None = None
    language: str = "hinglish"


class RecoverRevenueRequest(BaseModel):
    enabled: bool = True


@router.get("/promises")
def get_promises(limit: int = 50) -> dict[str, Any]:
    with get_session() as session:
        seed_demo_promises(session)
        items = list_promises(session, limit=limit)
        summary = promise_summary(session)
        session.commit()
        return {"summary": summary, "items": items}


@router.post("/promises")
def post_promise(request: PromiseCreateRequest) -> dict[str, Any]:
    with get_session() as session:
        result = create_promise(
            session,
            recovery_case_id=request.case_id,
            payment_id=request.payment_id,
            days=request.days,
            note=request.note,
            language=request.language,
            source="merchant",
        )
        session.commit()
        return result


@router.get("/promises/summary")
def get_promise_summary() -> dict[str, Any]:
    with get_session() as session:
        return promise_summary(session)


@router.get("/automation")
def get_automation() -> dict[str, Any]:
    return automation_status()


@router.post("/automation/recover-revenue")
def toggle_recover_revenue(request: RecoverRevenueRequest) -> dict[str, Any]:
    return set_recover_revenue(request.enabled)


@router.get("/campaigns")
def get_campaigns() -> dict[str, Any]:
    return campaign_overview()
