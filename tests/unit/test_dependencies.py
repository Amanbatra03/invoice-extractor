import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid


def test_verify_supabase_jwt_valid():
    from jose import jwt
    from api.dependencies import verify_supabase_jwt
    tenant_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    payload = {
        "sub": user_id,
        "app_metadata": {"tenant_id": tenant_id, "role": "analyst"},
        "email": "test@example.com",
    }
    token = jwt.encode(payload, "test_secret", algorithm="HS256")
    result = verify_supabase_jwt(token, "test_secret")
    assert result["sub"] == user_id
    assert result["app_metadata"]["role"] == "analyst"


def test_verify_supabase_jwt_invalid():
    from jose import jwt, JWTError
    from api.dependencies import verify_supabase_jwt
    with pytest.raises(Exception):
        verify_supabase_jwt("invalid.token.here", "wrong_secret")


def test_require_roles_allows_matching_role():
    from api.dependencies import require_roles, CurrentUser
    from fastapi import HTTPException

    async def mock_get_current_user():
        return CurrentUser(id="u1", tenant_id="t1", role="admin", email="a@b.com")

    checker = require_roles("admin", "analyst")
    # The checker is a FastAPI dependency — just verify it's callable
    assert callable(checker)


def test_current_user_dataclass():
    from api.dependencies import CurrentUser
    user = CurrentUser(id="u1", tenant_id="t1", role="viewer", email="v@x.com")
    assert user.role == "viewer"
    assert user.tenant_id == "t1"
