from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config.settings import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=8)
def _engine_for_url(database_url: str) -> Engine:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    if database_url.startswith("sqlite:///"):
        Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


def get_engine() -> Engine:
    return _engine_for_url(get_settings().database_url)


@contextmanager
def get_session() -> Iterator[Session]:
    session_factory = sessionmaker(
        bind=get_engine(),
        autoflush=False,
        expire_on_commit=False,
    )
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def initialize_database() -> None:
    # Importing registers every model with Base.metadata.
    import app.models  # noqa: F401
    import app.services.outcome_events  # noqa: F401

    engine = get_engine()
    if engine.dialect.name == "sqlite":
        inspector = inspect(engine)
        if inspector.has_table("webhook_events"):
            columns = {column["name"] for column in inspector.get_columns("webhook_events")}
            if "razorpay_event_id" not in columns:
                with engine.begin() as connection:
                    connection.execute(
                        text(
                            "ALTER TABLE webhook_events "
                            "RENAME TO webhook_events_legacy"
                        )
                    )
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    if inspector.has_table("interventions"):
        intervention_columns = {
            column["name"]
            for column in inspector.get_columns("interventions")
        }
        with engine.begin() as connection:
            if "agent_decision_id" not in intervention_columns:
                connection.execute(
                    text(
                        "ALTER TABLE interventions ADD COLUMN "
                        "agent_decision_id VARCHAR(36)"
                    )
                )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS "
                    "ix_interventions_agent_decision_id "
                    "ON interventions (agent_decision_id)"
                )
            )
    inspector = inspect(engine)
    if inspector.has_table("agent_decisions"):
        columns = {
            column["name"]
            for column in inspector.get_columns("agent_decisions")
        }
        additions = {
            "payment_id": "VARCHAR(36)",
            "features_version": (
                "VARCHAR(64) NOT NULL DEFAULT 'offline_phase8_features'"
            ),
            "execution_mode": "VARCHAR(32) NOT NULL DEFAULT 'dry_run'",
            "inference_status": "VARCHAR(32) NOT NULL DEFAULT 'completed'",
            "features_snapshot": (
                "JSON NOT NULL DEFAULT '{}'"
                if engine.dialect.name == "sqlite"
                else "JSON NOT NULL DEFAULT '{}'"
            ),
            "decision_margin": "FLOAT",
            "failure_class": "VARCHAR(64)",
            "risk_checks_passed": "BOOLEAN NOT NULL DEFAULT FALSE",
        }
        with engine.begin() as connection:
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(
                        text(
                            f"ALTER TABLE agent_decisions "
                            f"ADD COLUMN {name} {definition}"
                        )
                    )
            connection.execute(
                text(
                    "UPDATE agent_decisions SET payment_id = ("
                    "SELECT payment_id FROM recovery_cases "
                    "WHERE recovery_cases.id = "
                    "agent_decisions.recovery_case_id"
                    ") WHERE payment_id IS NULL"
                )
            )
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "uq_agent_decision_case_policy_mode "
                    "ON agent_decisions "
                    "(recovery_case_id, policy_version, execution_mode)"
                )
            )
    inspector = inspect(engine)
    if inspector.has_table("webhook_events"):
        webhook_columns = {
            column["name"]
            for column in inspector.get_columns("webhook_events")
        }
        with engine.begin() as connection:
            if "delivery_count" not in webhook_columns:
                connection.execute(
                    text(
                        "ALTER TABLE webhook_events ADD COLUMN "
                        "delivery_count INTEGER NOT NULL DEFAULT 1"
                    )
                )
            if "last_received_at" not in webhook_columns:
                connection.execute(
                    text(
                        "ALTER TABLE webhook_events ADD COLUMN "
                        "last_received_at TIMESTAMP"
                    )
                )
                connection.execute(
                    text(
                        "UPDATE webhook_events SET last_received_at = created_at "
                        "WHERE last_received_at IS NULL"
                    )
                )
    inspector = inspect(engine)
    if inspector.has_table("intervention_outcomes"):
        outcome_columns = {
            column["name"]
            for column in inspector.get_columns("intervention_outcomes")
        }
        outcome_additions = {
            "failure_timestamp": "TIMESTAMP",
            "payment_status_after_24h": "VARCHAR(40)",
            "payment_status_after_48h": "VARCHAR(40)",
        }
        with engine.begin() as connection:
            for name, definition in outcome_additions.items():
                if name not in outcome_columns:
                    connection.execute(
                        text(
                            "ALTER TABLE intervention_outcomes "
                            f"ADD COLUMN {name} {definition}"
                        )
                    )
    inspector = inspect(engine)
    if inspector.has_table("promises"):
        promise_columns = {
            column["name"] for column in inspector.get_columns("promises")
        }
        promise_additions = {
            "recovery_case_id": "VARCHAR(36)",
            "note": "TEXT",
            "source": "VARCHAR(40) DEFAULT 'merchant'",
            "language": "VARCHAR(32) DEFAULT 'hinglish'",
            "reminder_count": "INTEGER DEFAULT 0",
            "last_reminded_at": "TIMESTAMP",
            "fulfilled_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        }
        with engine.begin() as connection:
            for name, definition in promise_additions.items():
                if name not in promise_columns:
                    connection.execute(
                        text(f"ALTER TABLE promises ADD COLUMN {name} {definition}")
                    )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_promises_deadline "
                    "ON promises (deadline)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_promises_status "
                    "ON promises (status)"
                )
            )


def database_backend() -> str:
    return get_engine().dialect.name
