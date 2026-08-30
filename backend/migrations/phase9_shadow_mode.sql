CREATE TABLE IF NOT EXISTS payment_feature_contexts (
    id VARCHAR(36) PRIMARY KEY,
    payment_id VARCHAR(36) NOT NULL UNIQUE REFERENCES payments(id),
    transaction_type VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN',
    merchant_category VARCHAR(120) NOT NULL DEFAULT 'UNKNOWN',
    device_type VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN',
    network_type VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN',
    sender_age_group VARCHAR(64) NOT NULL DEFAULT 'UNKNOWN',
    sender_state VARCHAR(120) NOT NULL DEFAULT 'UNKNOWN',
    sender_bank VARCHAR(120) NOT NULL DEFAULT 'UNKNOWN',
    fraud_flag INTEGER NOT NULL DEFAULT 0,
    source VARCHAR(64) NOT NULL DEFAULT 'razorpay_webhook',
    unknown_fields JSON NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE agent_decisions
    ADD COLUMN IF NOT EXISTS payment_id VARCHAR(36) REFERENCES payments(id),
    ADD COLUMN IF NOT EXISTS features_version VARCHAR(64)
        NOT NULL DEFAULT 'offline_phase8_features',
    ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(32)
        NOT NULL DEFAULT 'dry_run',
    ADD COLUMN IF NOT EXISTS inference_status VARCHAR(32)
        NOT NULL DEFAULT 'completed',
    ADD COLUMN IF NOT EXISTS features_snapshot JSON NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS decision_margin DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS failure_class VARCHAR(64),
    ADD COLUMN IF NOT EXISTS risk_checks_passed BOOLEAN NOT NULL DEFAULT FALSE;

UPDATE agent_decisions AS decision
SET payment_id = recovery.payment_id
FROM recovery_cases AS recovery
WHERE decision.recovery_case_id = recovery.id
  AND decision.payment_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_decision_case_policy_mode
ON agent_decisions (recovery_case_id, policy_version, execution_mode);
