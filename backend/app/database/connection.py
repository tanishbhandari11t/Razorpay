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


def database_backend() -> str:
    return get_engine().dialect.name
