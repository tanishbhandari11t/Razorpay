from __future__ import annotations

from fastapi import APIRouter

from app.database.connection import get_session
from app.services.payment_service import recent_recovery_cases


router = APIRouter(prefix="/api/recovery", tags=["recovery"])


@router.get("/cases")
def recovery_cases(limit: int = 25) -> list[dict]:
    with get_session() as session:
        return recent_recovery_cases(session, max(1, min(limit, 100)))
