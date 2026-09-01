from __future__ import annotations

import math
import random
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config.settings import REPO_ROOT, get_settings

from app.api.customers import router as customers_router
from app.api.outcomes import router as outcomes_router
from app.api.payments import router as payments_router
from app.api.promises import router as promises_router
from app.api.recovery import router as recovery_router
from app.api.webhooks import router as webhooks_router
from app.database import initialize_database
from app.ml.model_loader import load_model_bundle
from app.services.execution_gate import load_execution_gate
from app.services.runtime_health import runtime_health


FailureReason = Literal[
    "insufficient_funds",
    "bank_timeout",
    "expired_method",
    "mandate_failure",
    "unknown",
]
Strategy = Literal["smart_retry", "payment_link", "message", "voice_call", "escalate"]
CaseStatus = Literal["recovered", "active", "stopped", "escalated"]


class SimulationRequest(BaseModel):
    transactions: int = Field(default=500, ge=10, le=50_000)
    seed: int = 42


class AuditEvent(BaseModel):
    timestamp: str
    event: str
    detail: str
    tone: Literal["neutral", "positive", "warning"] = "neutral"


class RecoveryCase(BaseModel):
    id: str
    customer_name: str
    amount: int
    language: str
    payment_method: str
    failure_reason: FailureReason
    previous_payments: int
    successful_payments: int
    customer_ltv: int
    recovery_probability: float
    strategy: Strategy
    status: CaseStatus
    interventions: int
    intervention_cost: float
    recovered_amount: int
    created_at: str
    audit: list[AuditEvent]


class DashboardResponse(BaseModel):
    transactions: int
    failed_transactions: int
    revenue_at_risk: int
    revenue_recovered: int
    net_recovered: float
    recovery_rate: float
    intervention_cost: float
    interventions: int
    recovered_cases: int
    stopped_cases: int
    escalated_cases: int
    policy_violations: int
    duplicate_actions: int
    strategy_mix: dict[str, int]


app = FastAPI(
    title="RecoverAI",
    version="0.1.0",
    description="A bounded revenue-recovery workflow for failed subscription payments.",
)
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_origin_regex=_settings.cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(payments_router)
app.include_router(customers_router)
app.include_router(outcomes_router)
app.include_router(promises_router)
app.include_router(recovery_router)
app.include_router(webhooks_router)

CASES: list[RecoveryCase] = []
TOTAL_TRANSACTIONS = 0
POLICY = {
    "max_payment_retries": 2,
    "max_customer_contacts": 2,
    "max_interventions": 3,
    "min_contact_interval_hours": 24,
    "max_recovery_window_days": 7,
}

NAMES = [
    ("Amit Sharma", "Hinglish"),
    ("Priya Nair", "English"),
    ("Arjun Rao", "Kannada"),
    ("Meera Iyer", "Tamil"),
    ("Rohan Gupta", "Hindi"),
    ("Ananya Singh", "English"),
]
FAILURES: list[tuple[FailureReason, float]] = [
    ("insufficient_funds", 0.40),
    ("bank_timeout", 0.25),
    ("expired_method", 0.15),
    ("mandate_failure", 0.12),
    ("unknown", 0.08),
]
ACTION_COST = {
    "smart_retry": 0.25,
    "payment_link": 0.50,
    "message": 0.35,
    "voice_call": 3.50,
    "escalate": 0.0,
}


def now(offset_minutes: int = 0) -> str:
    value = datetime.now(UTC) + timedelta(minutes=offset_minutes)
    return value.isoformat(timespec="seconds")


def event(
    offset: int,
    name: str,
    detail: str,
    tone: Literal["neutral", "positive", "warning"] = "neutral",
) -> AuditEvent:
    return AuditEvent(timestamp=now(offset), event=name, detail=detail, tone=tone)


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def score_recovery(
    amount: int,
    failure_reason: FailureReason,
    previous_payments: int,
    successful_payments: int,
    previous_interventions: int,
) -> float:
    """Transparent MVP scorer; replace with a calibrated trained model later."""
    reliability = successful_payments / max(previous_payments, 1)
    reason_weight = {
        "bank_timeout": 0.55,
        "insufficient_funds": 0.20,
        "mandate_failure": -0.15,
        "expired_method": -0.35,
        "unknown": -0.65,
    }[failure_reason]
    amount_weight = min(amount / 25_000, 1) * -0.25
    raw = -1.65 + (reliability * 1.6) + reason_weight + amount_weight
    raw -= previous_interventions * 0.75
    return round(min(max(sigmoid(raw), 0.05), 0.95), 2)


def choose_strategy(
    probability: float,
    amount: int,
    reason: FailureReason,
    previous_interventions: int,
) -> Strategy:
    if previous_interventions >= POLICY["max_interventions"]:
        return "escalate"
    if reason == "bank_timeout":
        return "smart_retry"
    if reason == "expired_method":
        return "payment_link"
    if reason == "insufficient_funds" and amount >= 2_000 and probability >= 0.52:
        return "voice_call"
    if probability >= 0.45:
        return "message"
    return "escalate"


def policy_allows(strategy: Strategy, previous_interventions: int) -> tuple[bool, str]:
    if strategy == "escalate":
        return False, "No automated action has positive expected value."
    if previous_interventions >= POLICY["max_interventions"]:
        return False, "Maximum intervention policy reached."
    return True, "Within retry, contact, and recovery-window limits."


def weighted_choice(rng: random.Random) -> FailureReason:
    marker = rng.random()
    cumulative = 0.0
    for reason, weight in FAILURES:
        cumulative += weight
        if marker <= cumulative:
            return reason
    return "unknown"


def build_case(rng: random.Random, index: int) -> RecoveryCase:
    customer_name, language = rng.choice(NAMES)
    amount = rng.choices(
        [499, 999, 1499, 2499, 4999, 8999, 14999, 50000],
        weights=[20, 18, 16, 14, 12, 9, 6, 1],
        k=1,
    )[0]
    previous = rng.randint(1, 24)
    successful = rng.randint(max(0, previous - 5), previous)
    reason = weighted_choice(rng)
    probability = score_recovery(amount, reason, previous, successful, 0)
    strategy = choose_strategy(probability, amount, reason, 0)
    allowed, policy_reason = policy_allows(strategy, 0)
    case_id = f"REC-{uuid4().hex[:8].upper()}"
    created = now(index)
    audit = [
        event(index, "Payment failed", f"₹{amount:,} · {reason.replace('_', ' ')}"),
        event(
            index + 1,
            "Customer context loaded",
            f"{successful}/{previous} successful payments · preferred language: {language}",
        ),
        event(
            index + 2,
            "Recovery score calculated",
            f"Estimated recovery probability: {probability:.0%}",
        ),
        event(
            index + 3,
            "Strategy selected",
            strategy.replace("_", " ").title(),
        ),
    ]

    if not allowed:
        audit.append(event(index + 4, "Policy blocked automation", policy_reason, "warning"))
        audit.append(
            event(index + 5, "Escalated to merchant", "Manual review requested.", "warning")
        )
        return RecoveryCase(
            id=case_id,
            customer_name=customer_name,
            amount=amount,
            language=language,
            payment_method=rng.choice(["UPI", "Card", "eMandate"]),
            failure_reason=reason,
            previous_payments=previous,
            successful_payments=successful,
            customer_ltv=amount * successful,
            recovery_probability=probability,
            strategy="escalate",
            status="escalated",
            interventions=0,
            intervention_cost=0,
            recovered_amount=0,
            created_at=created,
            audit=audit,
        )

    audit.append(event(index + 4, "Policy approved action", policy_reason, "positive"))
    audit.append(
        event(
            index + 5,
            "Intervention executed",
            f"{strategy.replace('_', ' ').title()} initiated in {language}.",
        )
    )

    # Ground truth for the simulator: interventions succeed according to the
    # score, with a small strategy-specific uplift/penalty.
    strategy_adjustment = {
        "smart_retry": 0.08,
        "payment_link": 0.03,
        "message": 0.00,
        "voice_call": 0.10,
        "escalate": 0.00,
    }[strategy]
    recovered = rng.random() < min(probability + strategy_adjustment, 0.96)
    if strategy == "voice_call":
        audit.append(
            event(
                index + 7,
                "Promise to pay recorded",
                "Customer committed to complete payment today by 8:00 PM.",
                "positive",
            )
        )

    if recovered:
        audit.append(
            event(index + 12, "Payment verified", "Payment provider status: captured.", "positive")
        )
        audit.append(
            event(index + 12, "Revenue recovered", f"₹{amount:,} recovered.", "positive")
        )
        status: CaseStatus = "recovered"
        recovered_amount = amount
    else:
        audit.append(
            event(index + 12, "Payment not recovered", "No successful payment detected.", "warning")
        )
        audit.append(
            event(
                index + 13,
                "Workflow stopped",
                "One bounded intervention completed; merchant review requested.",
                "warning",
            )
        )
        status = "stopped"
        recovered_amount = 0

    return RecoveryCase(
        id=case_id,
        customer_name=customer_name,
        amount=amount,
        language=language,
        payment_method=rng.choice(["UPI", "Card", "eMandate"]),
        failure_reason=reason,
        previous_payments=previous,
        successful_payments=successful,
        customer_ltv=amount * successful,
        recovery_probability=probability,
        strategy=strategy,
        status=status,
        interventions=1,
        intervention_cost=ACTION_COST[strategy],
        recovered_amount=recovered_amount,
        created_at=created,
        audit=audit,
    )


def run_simulation(transactions: int, seed: int) -> list[RecoveryCase]:
    rng = random.Random(seed)
    failed_count = max(1, round(transactions * 0.12))
    return [build_case(rng, index) for index in range(failed_count)]


def dashboard() -> DashboardResponse:
    at_risk = sum(case.amount for case in CASES)
    recovered = sum(case.recovered_amount for case in CASES)
    costs = round(sum(case.intervention_cost for case in CASES), 2)
    return DashboardResponse(
        transactions=TOTAL_TRANSACTIONS,
        failed_transactions=len(CASES),
        revenue_at_risk=at_risk,
        revenue_recovered=recovered,
        net_recovered=round(recovered - costs, 2),
        recovery_rate=round(recovered / at_risk, 4) if at_risk else 0,
        intervention_cost=costs,
        interventions=sum(case.interventions for case in CASES),
        recovered_cases=sum(case.status == "recovered" for case in CASES),
        stopped_cases=sum(case.status == "stopped" for case in CASES),
        escalated_cases=sum(case.status == "escalated" for case in CASES),
        policy_violations=0,
        duplicate_actions=0,
        strategy_mix=dict(Counter(case.strategy for case in CASES)),
    )


def seed_demo() -> None:
    global CASES, TOTAL_TRANSACTIONS
    if not CASES:
        TOTAL_TRANSACTIONS = 500
        CASES = run_simulation(TOTAL_TRANSACTIONS, 42)


seed_demo()
initialize_database()
load_model_bundle()
load_execution_gate()


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", **runtime_health()}


@app.get("/api/dashboard", response_model=DashboardResponse)
def get_dashboard() -> DashboardResponse:
    return dashboard()


@app.get("/api/cases", response_model=list[RecoveryCase])
def get_cases(limit: int = 25) -> list[RecoveryCase]:
    return CASES[: max(1, min(limit, 100))]


@app.get("/api/cases/{case_id}", response_model=RecoveryCase)
def get_case(case_id: str) -> RecoveryCase:
    match = next((case for case in CASES if case.id == case_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Recovery case not found")
    return match


@app.post("/api/simulations", response_model=DashboardResponse)
def create_simulation(request: SimulationRequest) -> DashboardResponse:
    global CASES, TOTAL_TRANSACTIONS
    TOTAL_TRANSACTIONS = request.transactions
    CASES = run_simulation(request.transactions, request.seed)
    return dashboard()


@app.get("/api/policy")
def get_policy() -> dict[str, int]:
    return POLICY


_dashboard_dist = REPO_ROOT / "dashboard" / "dist"
if (_dashboard_dist / "index.html").is_file():
    app.mount(
        "/",
        StaticFiles(directory=_dashboard_dist, html=True),
        name="dashboard",
    )
