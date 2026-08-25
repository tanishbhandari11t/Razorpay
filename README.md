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

1. Connect a hosted PostgreSQL `DATABASE_URL` and run the live event pipeline.
2. Add a dataset generator and held-out evaluation report.
3. Replace the baseline score with a calibrated recovery model.
4. Add payment-link recovery, outbound idempotency keys, and retry queues.
5. Add multilingual message generation with deterministic templates as a
   fallback.
6. Add voice and promise-to-pay only after the payment recovery loop is stable.

## Safety boundary

AI may recommend an intervention. It cannot authorize money movement. Retry
limits, contact limits, idempotency, escalation, and payment verification remain
deterministic.
