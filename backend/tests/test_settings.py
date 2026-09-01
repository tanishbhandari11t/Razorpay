from app.config.settings import (
    DEFAULT_CORS_ORIGINS,
    Settings,
    normalize_database_url,
    parse_cors_origins,
)


def test_database_url_normalizes_render_and_heroku_schemes() -> None:
    assert (
        normalize_database_url("postgresql://user:pass@host:5432/db")
        == "postgresql+psycopg2://user:pass@host:5432/db"
    )
    assert (
        normalize_database_url("postgres://user:pass@host:5432/db?sslmode=require")
        == "postgresql+psycopg2://user:pass@host:5432/db?sslmode=require"
    )
    already = "postgresql+psycopg2://user:pass@host:5432/db"
    assert normalize_database_url(already) == already
    sqlite = "sqlite:///C:/tmp/recoverai.db"
    assert normalize_database_url(sqlite) == sqlite
    assert normalize_database_url("") == normalize_database_url(None)


def test_cors_origins_merge_localhost_with_hosted_dashboard() -> None:
    origins = parse_cors_origins("https://recoverai.vercel.app, https://revback.onrender.com/")
    assert "https://recoverai.vercel.app" in origins
    assert "https://revback.onrender.com" in origins
    for origin in DEFAULT_CORS_ORIGINS:
        assert origin in origins


def test_settings_read_cors_and_database_from_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://recoverai:secret@db.internal:5432/recoverai",
    )
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql+psycopg2://")
    assert "https://app.example.com" in settings.cors_origin_list
    assert "http://localhost:5173" in settings.cors_origin_list
