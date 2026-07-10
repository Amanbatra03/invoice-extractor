from unittest.mock import AsyncMock

import pytest

from frontend.api_client import APIClient


@pytest.mark.asyncio
async def test_chat_client_methods_hit_expected_paths():
    client = APIClient("http://x", "tok")
    client._post = AsyncMock(return_value={"id": "c1"})
    client._get = AsyncMock(return_value=[])
    client._delete = AsyncMock(return_value={"deleted": "c1"})

    await client.create_conversation("June")
    client._post.assert_awaited_with("/api/v1/chat/conversations", json={"title": "June"})

    await client.list_conversations()
    client._get.assert_awaited_with("/api/v1/chat/conversations")

    await client.get_conversation("c1")
    client._get.assert_awaited_with("/api/v1/chat/conversations/c1")

    await client.send_message("c1", "highest total?")
    client._post.assert_awaited_with(
        "/api/v1/chat/conversations/c1/messages", json={"content": "highest total?"}
    )

    await client.delete_conversation("c1")
    client._delete.assert_awaited_with("/api/v1/chat/conversations/c1")
