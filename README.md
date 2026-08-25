# RecoverAI

RecoverAI is a bounded revenue-recovery workflow for failed subscription
payments. It detects revenue at risk, estimates recoverability, selects an
intervention, checks deterministic policy limits, executes a simulated action,
and verifies whether money was recovered.

This repository currently contains the first runnable MVP:

- a FastAPI recovery simulator and JSON API;
- a transparent recovery-probability scorer;
- deterministic strategy and policy decisions;
- simulated retry, payment-link, message, and voice-call outcomes;
- Razorpay Test Mode order creation and Standard Checkout;
- raw-body webhook verification and event-ID idempotency;
- SQLAlchemy persistence for SQLite development or hosted PostgreSQL;
- recovery metrics in rupees;
- a React merchant dashboard and per-case audit trail;
- API tests for bounded actions and deterministic batch evaluation.

Razorpay actions are restricted to Test Mode. No real money moves.

## Run locally on Windows

Open two PowerShell terminals.

### 1. Backend

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8010
```

The API is available at `http://localhost:8010`. Interactive API documentation
is at `http://localhost:8010/docs`.

### 2. Dashboard

```powershell
cd dashboard
npm install
npm run dev
```

Open `http://localhost:5173`, then select **Run 1,000 transactions**.

## Configure Razorpay Test Mode

If a test key has ever been posted in chat, a screenshot, an issue, or a
commit, revoke it first. Generate a fresh key pair while the Razorpay Dashboard
is in Test Mode.

Copy `backend/.env.example` to `backend/.env` and fill in the replacement
values:

```dotenv
RAZORPAY_KEY_ID=rzp_test_replace_me
RAZORPAY_KEY_SECRET=replace_me
RAZORPAY_WEBHOOK_SECRET=replace_me_with_a_long_random_secret
```

`backend/.env` is ignored by Git. The Key Secret is loaded only by FastAPI. The
public Test Key ID is returned with a server-created order because Razorpay
Checkout requires it.

Restart FastAPI after changing `.env`, then verify:

```text
GET  http://localhost:8010/api/razorpay/status
GET  http://localhost:8010/api/razorpay/test
POST http://localhost:8010/api/payments/create-order
```

The dashboard provides two Test Mode paths:

- **Test order** performs server-created Order → Checkout for ₹2,499.
- **Start subscription** creates a monthly Plan and six-cycle Subscription,
  then opens Checkout for authentication.

After authenticating the subscription, trigger a failed subsequent charge from
the Razorpay Test Mode Dashboard. Razorpay should send `subscription.pending`
to the configured webhook.

## Receive Test Mode webhooks

Razorpay cannot reach localhost directly. Expose the local API using a tunnel
that forwards to port 8010, for example:

```powershell
cloudflared tunnel --url http://127.0.0.1:8010 --protocol http2
```

In the Razorpay Dashboard's Test Mode webhook settings, configure:

```text
https://<your-tunnel-host>/api/webhooks/razorpay
```

Use the same webhook secret in the Dashboard and `backend/.env`. Enable:

- `payment.failed`
- `payment.captured`
- `subscription.pending`
- `subscription.charged`
- `subscription.activated`

Webhook bodies are verified against the raw request bytes before parsing.
Duplicate deliveries are ignored using Razorpay's `X-Razorpay-Event-Id`, backed
by a unique database constraint. Failed payment state can be inspected at:

```text
GET http://localhost:8010/api/payments/provider-states
```

Every new `payment.failed` event also creates exactly one recovery case:

```text
GET http://localhost:8010/api/recovery/cases
```

## PostgreSQL

Without `DATABASE_URL`, local development uses
`backend/data/recoverai.db`. To use Supabase PostgreSQL, add its pooler
connection string to `backend/.env`:

```dotenv
DATABASE_URL=postgresql+psycopg2://user:password@host:6543/postgres?sslmode=require
```

Restart FastAPI. `GET /health` reports the active database dialect. Tables for
customers, payments, interventions, promises, webhook events, and recovery
cases are created automatically.

## ML data foundation

The Kaggle UPI data provides transaction behavior, not customer identities or
recovery outcomes. The current pipeline deliberately stops before labels and
model training:

```powershell
.\ml\.venv\Scripts\python.exe -m ml.src.clean
.\ml\.venv\Scripts\python.exe -m ml.src.assign_customers
.\ml\.venv\Scripts\python.exe -m ml.src.build_features
.\ml\.venv\Scripts\python.exe -m ml.src.simulate_recovery
.\ml\.venv\Scripts\python.exe -m ml.src.create_logging_policy
```

Phase 1 validates and cleans all 250,000 transactions. Phase 2 assigns them to
10,000 synthetic customers using a fixed seed. Sender age group, state, and bank
are stable profile strata; activity archetypes and device/network preferences
are documented synthetic assumptions.

Generated files under `ml/data/processed/` include `customers.csv`,
`transactions_with_customers.csv`, and a validation summary with SHA-256
hashes. They are ignored by Git. No recovery label has been generated and the
synthetic archetype must not be used as a model feature.

Phase 3 produces one leakage-safe feature row for each failed payment. History
uses only transactions with timestamps strictly earlier than the prediction
time; same-timestamp transactions and the current failure are excluded. The
pipeline uses explicit no-history indicators, `-1` for unavailable recency, and
`UNKNOWN` for unavailable historical modes. See
`ml/notebooks/03_temporal_features.ipynb` for the audit workflow.

Phase 4 creates four deterministic potential outcomes per failed payment using
the versioned assumptions in `ml/config/recovery_simulation.yaml`. Observable
history, intervention/scenario interactions, hidden customer responsiveness,
and stable hash-based random draws create a probabilistic experimental world.
Fraud blocks automated actions as a policy rule.

`simulated_recovery_probability` exists only for environment calibration and is
forbidden as a model input. The four counterfactual outcomes also cannot be
treated as four observed real-world outcomes; the next data step must simulate
a historical logging policy that reveals one intervention and one outcome per
payment.

Phase 5 applies the versioned policy in `ml/config/logging_policy.yaml`. Ninety
percent of eligible payments follow a simple ordered ruleset; ten percent
explore uniformly among the other three actions. The resulting propensity is
therefore `0.90` for the baseline action and `0.10 / 3` for an explored
alternative. Fraud cases deterministically receive `no_action` with propensity
`1.0`.

`logging_policy_dataset.csv` contains exactly one chosen intervention and one
observed outcome per failed payment. It excludes simulator probabilities,
synthetic scenarios, and all unchosen outcomes. Its train, validation, and test
partitions are strictly chronological.

## Test

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q

cd ..\dashboard
npm run lint
npm run build
```

## Current workflow

```text
failed subscription
  -> load customer context
  -> estimate P(recovery)
  -> select recovery strategy
  -> deterministic policy check
  -> execute one simulated intervention
  -> verify simulated payment outcome
  -> recovered, stopped, or escalated
  -> immutable-style audit events
```

The scorer in `backend/app/main.py` is deliberately transparent and
rule-based. It is an MVP baseline, not a trained production model. The next ML
milestone is to generate a labelled recovery dataset, train and calibrate an
XGBoost/LightGBM model, and compare it against this baseline on a held-out set.

## Next milestones

1. Train XGBoost on the predefined temporal split and calibrate probabilities.
2. Evaluate discrimination, calibration, and support by intervention.
3. Compare learned policy value against always-retry and historical baselines.
4. Connect model inference to PostgreSQL recovery cases.
5. Add payment-link recovery, outbound idempotency keys, and retry queues.
6. Add multilingual message generation with deterministic templates as a
   fallback.
7. Add voice and promise-to-pay only after the payment recovery loop is stable.

## Safety boundary

AI may recommend an intervention. It cannot authorize money movement. Retry
limits, contact limits, idempotency, escalation, and payment verification remain
deterministic.
