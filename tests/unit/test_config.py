import os
import pytest
from unittest.mock import patch


def test_settings_load_from_env():
    env_vars = {
        "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/testdb",
        "REDIS_URL": "redis://localhost:6379",
        "SUPABASE_URL": "https://fake.supabase.co",
        "SUPABASE_ANON_KEY": "fake_anon",
        "SUPABASE_SERVICE_KEY": "fake_service",
        "GOOGLE_API_KEY": "fake_google_key",
    }
    with patch.dict(os.environ, env_vars):
        from api.config import Settings
        s = Settings()
        assert s.DATABASE_URL == "postgresql+asyncpg://test:test@localhost/testdb"
        assert s.LLM_PROVIDER == "gemini"  # default
        assert s.ENV == "development"  # default


def test_get_settings_is_cached():
    env_vars = {
        "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/testdb",
        "REDIS_URL": "redis://localhost:6379",
        "SUPABASE_URL": "https://fake.supabase.co",
        "SUPABASE_ANON_KEY": "fake_anon",
        "SUPABASE_SERVICE_KEY": "fake_service",
        "GOOGLE_API_KEY": "fake_google_key",
    }
    with patch.dict(os.environ, env_vars):
        from api.config import get_settings
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # same object returned
