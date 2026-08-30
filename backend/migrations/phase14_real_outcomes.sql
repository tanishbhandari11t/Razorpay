ALTER TABLE interventions
    ADD COLUMN IF NOT EXISTS agent_decision_id VARCHAR(36)
    REFERENCES agent_decisions(id);

CREATE INDEX IF NOT EXISTS ix_interventions_agent_decision_id
ON interventions (agent_decision_id);

CREATE TABLE IF NOT EXISTS intervention_outcomes (
    id VARCHAR(36) PRIMARY KEY,
    outcome_key VARCHAR(200) NOT NULL UNIQUE,
    agent_decision_id VARCHAR(36) NOT NULL UNIQUE REFERENCES agent_decisions(id),
    intervention_id VARCHAR(36) REFERENCES interventions(id),
    payment_id VARCHAR(36) NOT NULL REFERENCES payments(id),
    recovery_case_id VARCHAR(36) NOT NULL REFERENCES recovery_cases(id),
    action VARCHAR(64),
    decision_probability DOUBLE PRECISION,
    decision_margin DOUBLE PRECISION,
    model_version VARCHAR(64) NOT NULL,
    policy_version VARCHAR(64) NOT NULL,
    execution_mode VARCHAR(32) NOT NULL,
    attempted BOOLEAN NOT NULL DEFAULT FALSE,
    attempted_at TIMESTAMPTZ,
    failure_timestamp TIMESTAMPTZ,
    payment_status_after_24h VARCHAR(40),
    payment_status_after_48h VARCHAR(40),
    outcome_state VARCHAR(40) NOT NULL DEFAULT 'decided',
    outcome_at TIMESTAMPTZ,
    payment_recovered BOOLEAN,
    recovered_amount_minor BIGINT NOT NULL DEFAULT 0,
    recovery_timestamp TIMESTAMPTZ,
    time_to_recovery_seconds BIGINT,
    observation_window_starts_at TIMESTAMPTZ NOT NULL,
    observation_window_ends_at TIMESTAMPTZ NOT NULL,
    outcome_source VARCHAR(32) NOT NULL DEFAULT 'database',
    data_source VARCHAR(32) NOT NULL,
    natural_recovery_observed BOOLEAN NOT NULL DEFAULT FALSE,
    last_observed_at TIMESTAMPTZ,
    state_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_intervention_outcome_state CHECK (
        outcome_state IN (
            'decided',
            'executed',
            'waiting_for_outcome',
            'recovered',
            'no_recovery_observed',
            'unknown'
        )
    ),
    CONSTRAINT ck_intervention_outcome_source CHECK (
        outcome_source IN ('provider', 'webhook', 'database', 'simulator')
    ),
    CONSTRAINT ck_intervention_outcome_data_source CHECK (
        data_source IN ('synthetic', 'real_shadow', 'real_controlled')
    )
);

CREATE INDEX IF NOT EXISTS ix_intervention_outcomes_state_window
ON intervention_outcomes (outcome_state, observation_window_ends_at);

CREATE INDEX IF NOT EXISTS ix_intervention_outcomes_payment_state
ON intervention_outcomes (payment_id, outcome_state);

CREATE INDEX IF NOT EXISTS ix_intervention_outcomes_evidence_mode
ON intervention_outcomes (data_source, execution_mode);

CREATE TABLE IF NOT EXISTS outcome_observations (
    id VARCHAR(36) PRIMARY KEY,
    intervention_outcome_id VARCHAR(36) NOT NULL
        REFERENCES intervention_outcomes(id),
    observation_source VARCHAR(32) NOT NULL,
    external_ref VARCHAR(160) NOT NULL,
    payment_status VARCHAR(40) NOT NULL,
    recovered_signal BOOLEAN NOT NULL DEFAULT FALSE,
    attribution_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_outcome_observation_evidence UNIQUE (
        intervention_outcome_id,
        observation_source,
        external_ref
    )
);

CREATE INDEX IF NOT EXISTS ix_outcome_observations_outcome_id
ON outcome_observations (intervention_outcome_id);

ALTER TABLE intervention_outcomes
    ADD COLUMN IF NOT EXISTS failure_timestamp TIMESTAMPTZ;
ALTER TABLE intervention_outcomes
    ADD COLUMN IF NOT EXISTS payment_status_after_24h VARCHAR(40);
ALTER TABLE intervention_outcomes
    ADD COLUMN IF NOT EXISTS payment_status_after_48h VARCHAR(40);
