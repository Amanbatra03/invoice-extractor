import pytest
import respx
import httpx
from frontend.api_client import APIClient

@pytest.mark.asyncio
async def test_client_list_invoices():
    client = APIClient(base_url="http://fake-api", token="jwt-token")
    with respx.mock:
        respx.get("http://fake-api/api/v1/invoices").mock(
            return_value=httpx.Response(200, json={"data": {"items": [], "total": 0, "page": 1, "limit": 20}, "error": None, "request_id": "x"})
        )
        result = await client.list_invoices()
    assert result["items"] == []

@pytest.mark.asyncio
async def test_client_raises_on_auth_error():
    client = APIClient(base_url="http://fake-api", token="bad-token")
    with respx.mock:
        respx.get("http://fake-api/api/v1/invoices").mock(
            return_value=httpx.Response(401, json={"data": None, "error": "Unauthorized", "request_id": "x"})
        )
        with pytest.raises(Exception, match="401"):
            await client.list_invoices()
