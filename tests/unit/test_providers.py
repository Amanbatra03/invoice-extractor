import pytest
from unittest.mock import MagicMock, patch


def test_llm_provider_protocol():
    from agents.base import LLMProvider
    import typing
    # Protocol should be importable and have required methods
    assert hasattr(LLMProvider, 'embed_text')
    assert hasattr(LLMProvider, 'embed_image')
    assert hasattr(LLMProvider, 'generate')
    assert hasattr(LLMProvider, 'generate_structured')


def test_get_provider_returns_gemini_by_default():
    import os
    with patch.dict(os.environ, {
        "DATABASE_URL": "postgresql+asyncpg://x:y@localhost/test",
        "SUPABASE_URL": "https://fake.supabase.co",
        "SUPABASE_ANON_KEY": "fake",
        "SUPABASE_SERVICE_KEY": "fake",
        "GOOGLE_API_KEY": "fake_key",
        "LLM_PROVIDER": "gemini",
    }):
        from api.config import get_settings
        get_settings.cache_clear()
        with patch('google.genai.Client'):
            from agents.base import get_provider
            provider = get_provider()
            from agents.providers.gemini import GeminiProvider
            assert isinstance(provider, GeminiProvider)


def test_get_provider_returns_ollama_for_local():
    import os
    with patch.dict(os.environ, {
        "DATABASE_URL": "postgresql+asyncpg://x:y@localhost/test",
        "SUPABASE_URL": "https://fake.supabase.co",
        "SUPABASE_ANON_KEY": "fake",
        "SUPABASE_SERVICE_KEY": "fake",
        "GOOGLE_API_KEY": "fake_key",
        "LLM_PROVIDER": "ollama_gemma",
    }):
        from api.config import get_settings
        get_settings.cache_clear()
        from agents.base import get_provider
        provider = get_provider()
        from agents.providers.ollama_gemma import OllamaGemmaProvider
        assert isinstance(provider, OllamaGemmaProvider)
