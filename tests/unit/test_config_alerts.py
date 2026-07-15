import os
from unittest.mock import patch

_TEST_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/testdb",
    "REDIS_URL": "redis://localhost:6379",
    "SUPABASE_URL": "https://fake.supabase.co",
    "SUPABASE_ANON_KEY": "fake_anon",
    "SUPABASE_SERVICE_KEY": "fake_service",
    "SUPABASE_JWT_SECRET": "test_secret",
    "GOOGLE_API_KEY": "fake_google_key",
}


def test_alert_settings_defaults():
    from api.config import get_settings
    with patch.dict(os.environ, _TEST_ENV):
        get_settings.cache_clear()
        s = get_settings()
        assert s.ALERT_DISCORD_WEBHOOK_URL == ""
        assert s.ALERT_COOLDOWN_SECONDS == 600
    get_settings.cache_clear()
