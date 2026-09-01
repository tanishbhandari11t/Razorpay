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

## Deploy

Hosted RecoverAI is three processes from one Docker image: the API (which also
serves the dashboard), a Celery worker, and Celery beat. Keep Razorpay in Test
Mode and `EXECUTION_MODE=shadow`. Do not use live keys.

Frozen model files must be on disk before the image will boot:

```text
ml/artifacts/recovery_model_v1.json
ml/artifacts/preprocessing_v1.joblib
ml/artifacts/calibration_v1.joblib
ml/artifacts/model_metadata.json
```

They were gitignored during training. The ignore rules now allow those four
files so you can commit them:

```powershell
git add ml/artifacts/recovery_model_v1.json ml/artifacts/preprocessing_v1.joblib ml/artifacts/calibration_v1.joblib ml/artifacts/model_metadata.json
```

Confirm locally:

```powershell
py deploy/check_runtime_files.py
```

### Render (recommended)

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and select the repo. `render.yaml`
   creates `recoverai-api`, `recoverai-worker`, `recoverai-beat`, Redis, and
   Postgres.
3. Set the Blueprint secrets: `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`,
   `RAZORPAY_WEBHOOK_SECRET`. Leave `CORS_ORIGINS` blank unless the dashboard
   is hosted on a different domain.
4. Open the API URL. The dashboard is served from the same host. Check
   `https://<api-host>/health` — `database` should be `postgresql`, `redis`
   `available`, `worker` `ready`, and `execution` `shadow`. If the web
   service is killed while loading the model, raise it above 512 MB RAM.
5. In Razorpay Test Mode, set the webhook URL to
   `https://<api-host>/api/webhooks/razorpay` with the same secret. Enable
   `payment.failed`, `payment.captured`, `subscription.pending`,
   `subscription.charged`, and `subscription.activated`.

`DATABASE_URL` values from Render/Railway (`postgres://` or `postgresql://`)
are rewritten to `postgresql+psycopg2://` automatically.

### Railway

1. New Project → Deploy from GitHub → this repo. Railway uses `railway.toml`
   and the Dockerfile for the web process.
2. Add a **PostgreSQL** plugin and a **Redis** plugin.
3. Duplicate the service twice. Set start commands:

   ```text
   api:    sh /app/scripts/start.sh api
   worker: sh /app/scripts/start.sh worker
   beat:   sh /app/scripts/start.sh beat
   ```

4. Give all three services the same variables: `DATABASE_URL`, `REDIS_URL`,
   Razorpay Test Mode keys, `EXECUTION_MODE=shadow`,
   `CELERY_TASK_ALWAYS_EAGER=false`.
5. Generate a public domain on the API service and point the Razorpay webhook
   at `/api/webhooks/razorpay`.

### Docker Compose (local production-shaped stack)

```powershell
docker compose up --build
```

API and dashboard: `http://localhost:8010`. Optional `backend/.env` supplies
Razorpay keys. Postgres and Redis run in the compose network.

### Separate dashboard host

If you deploy `dashboard/` to Vercel instead of serving `dist` from FastAPI:

1. Set `VITE_API_URL` to the public API origin and rebuild.
2. Set `CORS_ORIGINS` on the API to that dashboard origin, for example
   `https://your-app.vercel.app`.

Local Vite (`npm run dev`) still talks to `http://localhost:8010`. Extra
origins in `CORS_ORIGINS` are merged with those localhost defaults.

## ML data foundation

The Kaggle UPI data provides transaction behavior, not customer identities or
recovery outcomes. The pipeline keeps observed transaction data, synthetic
environment assumptions, logged outcomes, and model artifacts as separate
layers:

```powershell
.\ml\.venv\Scripts\python.exe -m ml.src.clean
.\ml\.venv\Scripts\python.exe -m ml.src.assign_customers
.\ml\.venv\Scripts\python.exe -m ml.src.build_features
.\ml\.venv\Scripts\python.exe -m ml.src.simulate_recovery
.\ml\.venv\Scripts\python.exe -m ml.src.create_logging_policy
.\ml\.venv\Scripts\python.exe -m ml.src.train_recovery_model
.\ml\.venv\Scripts\python.exe -m ml.src.evaluate_policy
.\ml\.venv\Scripts\python.exe -m ml.src.analyze_support
.\ml\.venv\Scripts\python.exe -m ml.src.offline_policy_eval
.\ml\.venv\Scripts\python.exe -m ml.src.evaluate_policy_sensitivity
.\ml\.venv\Scripts\python.exe -m ml.src.policy_engine_simulator
```

Phase 1 validates and cleans all 250,000 transactions. Phase 2 assigns them to
10,000 synthetic customers using a fixed seed. Sender age group, state, and bank
are stable profile strata; activity archetypes and device/network preferences
are documented synthetic assumptions.

Generated files under `ml/data/processed/` include `customers.csv`,
`transactions_with_customers.csv`, and a validation summary with SHA-256
hashes. They are ignored by Git. Synthetic customer archetypes must not be used
as model features.

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

Phase 6 freezes that dataset and split in `ml/config/dataset_manifest.yaml`.
XGBoost V1 uses 49 prediction-time features, one-hot encoding fitted only on
training data, and Platt scaling fitted only on validation data. Current fraud
is excluded from the feature matrix and remains a hard policy constraint.

On the frozen test set, the calibrated model has ROC-AUC `0.6249`, PR-AUC
`0.6526`, log loss `0.6610`, and Brier score `0.2341`. In the synthetic
counterfactual environment, its selected interventions recover `65.00%` versus
`58.91%` for always-retry on the same test payments, an uplift of `6.09`
percentage points. This is a simulation result, not real-world causal evidence.
Observed-outcome model evaluation and synthetic policy evaluation are reported
separately.

Phase 7 freezes every V1 artifact hash in
`ml/config/policy_evaluation.yaml`. Contextual support uses training-only
history, amount, and failure-frequency buckets. The minimum local action count
(`20`), inverse-propensity effective sample size (`8`), and probability margin
(`0.0170`) are derived from frozen training/validation diagnostics before V2
evaluation.

Support-safe V2 removes unsupported model candidates, applies the fraud block,
and falls back to retry or manual merchant escalation when evidence is
insufficient. It selects no unsupported model action. Offline evaluation uses
only logged actions, logged outcomes, propensities, and frozen calibrated V1
predictions. IPS, self-normalized IPS, doubly robust estimates, clipping
sensitivity, and deterministic customer-cluster bootstrap intervals are
reported under `ml/reports/`.

The evidence is mixed and is intentionally preserved. V2's doubly robust point
estimate is `62.44%` with a wide `53.25%–71.48%` interval. In the original
synthetic environment V2 reaches `58.54%` versus `58.91%` for always-retry. It
beats always-retry in only 6 of 16 intervention-probability perturbations.
Therefore V1's `+6.09` percentage-point synthetic result is not robust enough
for a production claim or execution integration.

Phase 8 freezes those findings in `ml/config/policy_v2_manifest.yaml` and wraps
the frozen model in Recovery Policy V3. The policy classifies failures using
deterministic rules, applies the versioned action matrix, contextual support,
attempt budgets, cooldowns, opt-out/fraud controls, and stopping rules, then
maximizes risk-adjusted expected monetary value. All intervention costs and
contact-availability assumptions are explicitly synthetic.

On the frozen synthetic test cases, V3 recovers `59.18%` versus `58.91%` for
always-retry. It recovers ₹1,526,714 with 1,579 interventions, compared with
₹1,411,362 from 1,853 always-retry interventions. V3 makes zero fraud automated
actions and zero policy violations. The uplift is deliberately reported as
small; the main Phase 8 gain is bounded behavior and intervention efficiency.

Phase 9 freezes V3 in `ml/config/policy_v3_manifest.yaml` and defines the
training/production feature contract in `ml/config/feature_schema.yaml`. The
online builder queries only the current customer's terminal payments before
the failed payment timestamp. Transactions at the same timestamp are excluded,
and 7/30-day windows preserve the Phase 3 open-left boundary exactly.

The parity suite compares all 52 temporal fields against
`ml/src/build_features.py`, including no-history sentinels and categorical
values. The cached startup loader verifies the frozen model hash, 49 raw model
features, 103 encoded features, calibrator, dataset version, and Policy V3.

Signed failed-payment webhooks now persist feature context, acknowledge the
event, and schedule shadow inference after persistence. Shadow decisions are
idempotent per case, policy, and execution mode and always record
`executed=false`; no Razorpay recovery action is available. Provider fields
that do not exist in a webhook are stored as `UNKNOWN` and surfaced by drift
monitoring rather than replaced with invented values.

Shadow APIs:

- `POST /api/recovery/{case_id}/evaluate`
- `GET /api/recovery/shadow/metrics`
- `GET /api/recovery/cases/{case_id}/decisions`

`backend/migrations/phase9_shadow_mode.sql` contains the initial
PostgreSQL/Supabase schema changes. Drift gates remain in `collecting` state
until at least 20 real Test Mode shadow cases.

Phase 10 replaces the in-process task with a transactional database outbox,
Redis, Celery worker, and Celery beat dispatcher:

```text
signed webhook -> persist payment/case/job -> return 200
                                          -> beat dispatches queued job
                                          -> Redis
                                          -> Celery worker
                                          -> shadow inference
                                          -> idempotent agent_decision
                                          -> execution gate blocks
```

The database job is committed before queue publication, so a Redis outage does
not lose the recovery request or delay the webhook while a broker connection
times out. Celery uses late acknowledgement, worker-loss rejection, one-message
prefetch, bounded retries, and permanent/retryable error classification.
`backend/migrations/phase10_durable_jobs.sql` adds the durable job records.

Configure `REDIS_URL`, then run these in separate terminals:

```powershell
cd backend
.\.venv\Scripts\celery.exe -A app.workers.celery_app:celery_app worker --loglevel=INFO --pool=solo
.\.venv\Scripts\celery.exe -A app.workers.celery_app:celery_app beat --loglevel=INFO
```

The global gate in `ml/config/execution_gate.yaml` is frozen to `shadow`;
provider actions and Qwen tools are disabled. Readiness is exposed through
`GET /api/recovery/shadow/gate`, and job failures through
`GET /api/recovery/jobs`.

Policy V4 is a separate economic-veto experiment and does not modify V3. On the
frozen one-step comparison, V4 reduced synthetic cost from ₹9,247 to ₹9,122 and
improved ROI from 131.06 to 132.80, but net recovered value fell by ₹343.
Therefore `ml/config/policy_v4_manifest.yaml` explicitly rejects V4 for
promotion. Synthetic costs are not presented as real provider pricing.

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

The simulation scorer in `backend/app/main.py` remains transparent and
rule-based. The frozen XGBoost model is connected only to the separate shadow
path; it has no execution capability.

## Next milestones

1. Collect at least 20 real Razorpay Test Mode shadow cases.
2. Review feature drift, unknown categories, confidence margins, and taxonomy
   coverage; stop if the dashboard drift gate reports `stop`.
3. Configure a real Redis endpoint and pass the worker crash/restart test.
4. Keep all payment, link, WhatsApp, voice, and Qwen execution disabled until a
   separately approved controlled-execution phase.
5. Improve V4 on validation data before reconsidering promotion.
6. Add signed internal policy-decision submission and audit access controls.
7. Add payment-link recovery, outbound idempotency keys, and retry queues.
8. Add multilingual message generation with deterministic templates as a
   fallback.
9. Add voice and promise-to-pay only after the payment recovery loop is stable.

## Safety boundary

AI may recommend an intervention. It cannot authorize money movement. Retry
limits, contact limits, idempotency, escalation, and payment verification remain
deterministic.
