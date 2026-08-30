CREATE TABLE IF NOT EXISTS recovery_jobs (
    id VARCHAR(36) PRIMARY KEY,
    recovery_case_id VARCHAR(36) NOT NULL REFERENCES recovery_cases(id),
    job_key VARCHAR(200) NOT NULL UNIQUE,
    task_name VARCHAR(80) NOT NULL DEFAULT 'shadow_inference',
    model_version VARCHAR(64) NOT NULL DEFAULT 'recovery_model_v1',
    policy_version VARCHAR(64) NOT NULL DEFAULT 'recovery_policy_v3',
    execution_mode VARCHAR(32) NOT NULL DEFAULT 'shadow',
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    celery_task_id VARCHAR(80),
    last_error TEXT,
    error_class VARCHAR(160),
    queued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_recovery_job_identity UNIQUE (
        recovery_case_id,
        task_name,
        policy_version,
        execution_mode
    )
);

CREATE INDEX IF NOT EXISTS ix_recovery_jobs_recovery_case_id
ON recovery_jobs (recovery_case_id);

CREATE INDEX IF NOT EXISTS ix_recovery_jobs_status
ON recovery_jobs (status);

ALTER TABLE webhook_events
    ADD COLUMN IF NOT EXISTS delivery_count INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS last_received_at TIMESTAMPTZ;

UPDATE webhook_events
SET last_received_at = created_at
WHERE last_received_at IS NULL;
