import uuid
from datetime import datetime, timezone

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = structlog.get_logger()

_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if request.method not in _MUTATING_METHODS:
            return response
        if response.status_code >= 400:
            return response

        try:
            from db.models import AuditLog
            from db.session import get_session_factory

            user_id_str = getattr(request.state, "user_id", None)
            tenant_id_str = getattr(request.state, "tenant_id", None)
            if not tenant_id_str:
                return response

            action = f"{request.method} {request.url.path}"
            session_factory = get_session_factory()
            async with session_factory() as db:
                db.add(AuditLog(
                    tenant_id=uuid.UUID(tenant_id_str),
                    user_id=uuid.UUID(user_id_str) if user_id_str and user_id_str != "api_key" else None,
                    action=action,
                    resource_type=None,
                    resource_id=None,
                    meta=None,
                    created_at=datetime.now(timezone.utc),
                ))
                await db.commit()
        except Exception as exc:
            log.warning("audit_write_failed", error=str(exc))

        return response
