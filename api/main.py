# Render free tier cannot route IPv6. asyncpg and httpx both reach DNS via
# socket.getaddrinfo (asyncio runs it in a thread executor). Filtering IPv6
# results here prevents ENETUNREACH on DB and Supabase storage connections.
import socket as _socket
_orig_getaddrinfo = _socket.getaddrinfo
def _ipv4_preferred(host, port, *a, **kw):
    results = _orig_getaddrinfo(host, port, *a, **kw)
    ipv4 = [r for r in results if r[0] == _socket.AF_INET]
    return ipv4 if ipv4 else results
_socket.getaddrinfo = _ipv4_preferred

import structlog
import structlog.contextvars
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from api.config import get_settings
from api.supabase_client import get_jwks_client
from api.middleware.rate_limiter import limiter
from api.middleware.request_context import RequestContextMiddleware
from api.routers import health as health_router
from api.routers import invoices as invoices_router
from api.routers import extraction as extraction_router
from api.routers import qa as qa_router
from api.routers import compare as compare_router
from api.routers import batch as batch_router
from api.routers import jobs as jobs_router
from api.routers import webhooks as webhooks_router
from api.routers import users as users_router
from api.routers import api_keys as api_keys_router
from api.routers import audit as audit_router
from api.routers import chat as chat_router
from api.routers import alerts as alerts_router

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if False else structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger()


def create_app() -> FastAPI:
    settings = get_settings()

    if settings.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.2,
            environment=settings.ENV,
        )

    app = FastAPI(
        title="Invoice Analyst API",
        version="2.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        lambda req, exc: JSONResponse(
            status_code=429,
            content={"data": None, "error": "Rate limit exceeded", "request_id": None},
        ),
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        log.error(
            "api.unhandled_exception",
            error=str(exc),
            method=request.method,
            path=request.url.path,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content={"data": None, "error": "Internal server error", "request_id": request_id},
        )

    @app.on_event("startup")
    async def _warmup():
        try:
            import asyncio
            await asyncio.to_thread(get_jwks_client)
            log.info("api.jwks_warmed")
        except Exception as exc:
            log.warning("api.jwks_warmup_failed", error=str(exc))

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router.router, prefix="/api/v1")
    app.include_router(invoices_router.router, prefix="/api/v1")
    app.include_router(extraction_router.router, prefix="/api/v1")
    app.include_router(qa_router.router, prefix="/api/v1")
    app.include_router(compare_router.router, prefix="/api/v1")
    app.include_router(batch_router.router, prefix="/api/v1")
    app.include_router(jobs_router.router, prefix="/api/v1")
    app.include_router(webhooks_router.router, prefix="/api/v1")
    app.include_router(users_router.router, prefix="/api/v1")
    app.include_router(api_keys_router.router, prefix="/api/v1")
    app.include_router(audit_router.router, prefix="/api/v1")
    app.include_router(chat_router.router, prefix="/api/v1")
    app.include_router(alerts_router.router, prefix="/api/v1")

    from prometheus_fastapi_instrumentator import Instrumentator
    from prometheus_client import Counter, Histogram

    try:
        extractions_total = Counter(  # noqa: F841
            "invoice_extractions_total", "Total extractions", ["status"]
        )
        extraction_duration = Histogram(  # noqa: F841
            "invoice_extraction_duration_seconds", "Extraction latency",
            buckets=[0.5, 1, 2, 5, 10, 30],
        )
        tokens_used_total = Counter(  # noqa: F841
            "llm_tokens_used_total", "LLM tokens", ["model", "direction"]
        )
    except ValueError:
        pass

    Instrumentator().instrument(app).expose(app, endpoint="/api/v1/metrics")

    return app


app = create_app()
