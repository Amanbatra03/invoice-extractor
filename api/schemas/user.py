from pydantic import BaseModel
import uuid


class UserOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    role: str


class RoleUpdateIn(BaseModel):
    role: str
