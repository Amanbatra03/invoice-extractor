import structlog
from jwt import PyJWKClient
from supabase import create_client, Client

log = structlog.get_logger()

_jwks_client: PyJWKClient | None = None


def get_service_client() -> Client:
    """Create a fresh Supabase client with the service role key (bypasses RLS)."""
    from api.config import get_settings
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


def get_jwks_client() -> PyJWKClient:
    """Return a cached JWKS client for ES256/RS256 JWT verification."""
    global _jwks_client
    if _jwks_client is None:
        from api.config import get_settings
        settings = get_settings()
        _jwks_client = PyJWKClient(
            f"{settings.SUPABASE_URL}/auth/v1/.well-known/jwks.json",
            cache_jwk_set=True,
            lifespan=3600,
        )
        log.info("jwks.client_initialized")
    return _jwks_client
