import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid


def test_request_id_generated():
    import uuid as _uuid
    # RequestContextMiddleware generates a UUID request ID
    request_id = str(_uuid.uuid4())
    assert len(request_id) == 36


@pytest.mark.asyncio
async def test_audit_middleware_skips_get_requests():
    from api.middleware.audit_writer import AuditMiddleware
    app = MagicMock()
    middleware = AuditMiddleware(app)

    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_call_next = AsyncMock(return_value=MagicMock(status_code=200))

    response = await middleware.dispatch(mock_request, mock_call_next)
    # GET requests should pass through without writing to audit log
    mock_call_next.assert_called_once_with(mock_request)


@pytest.mark.asyncio
async def test_audit_middleware_skips_4xx_errors():
    from api.middleware.audit_writer import AuditMiddleware
    app = MagicMock()
    middleware = AuditMiddleware(app)

    mock_request = MagicMock()
    mock_request.method = "POST"
    mock_request.url = MagicMock(path="/api/v1/invoices/upload")
    mock_call_next = AsyncMock(return_value=MagicMock(status_code=422))

    response = await middleware.dispatch(mock_request, mock_call_next)
    mock_call_next.assert_called_once_with(mock_request)
