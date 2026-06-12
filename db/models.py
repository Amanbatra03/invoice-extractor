import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer,
    Numeric, String, Text, ARRAY, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    plan = Column(String(50), nullable=False, server_default="free")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    users = relationship("User", back_populates="tenant")
    invoices = relationship("Invoice", back_populates="tenant")


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(320), nullable=False)
    role = Column(String(20), nullable=False, server_default="analyst")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    tenant = relationship("Tenant", back_populates="users")
    __table_args__ = (UniqueConstraint("tenant_id", "email"),)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    role = Column(String(20), nullable=False, server_default="api_user")
    active = Column(Boolean, nullable=False, server_default="true")
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


class Invoice(Base):
    __tablename__ = "invoices"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    file_name = Column(String(512), nullable=False)
    file_type = Column(String(10), nullable=False)
    storage_path = Column(Text, nullable=False)
    sha256 = Column(String(64), nullable=False)
    status = Column(String(20), nullable=False, server_default="pending")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    tenant = relationship("Tenant", back_populates="invoices")
    chunks = relationship("InvoiceChunk", back_populates="invoice", cascade="all, delete-orphan")
    extraction = relationship("Extraction", back_populates="invoice", uselist=False, cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint("tenant_id", "sha256"),)


class InvoiceChunk(Base):
    __tablename__ = "invoice_chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    page_num = Column(Integer, nullable=False)
    embedding = Column(Vector(768), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    invoice = relationship("Invoice", back_populates="chunks")


class Extraction(Base):
    __tablename__ = "extractions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    invoice_id = Column(UUID(as_uuid=True), ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False, unique=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    schema_json = Column(JSONB, nullable=False)
    model_used = Column(String(100), nullable=False)
    validated = Column(Boolean, nullable=False, server_default="false")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    invoice = relationship("Invoice", back_populates="extraction")


class Job(Base):
    __tablename__ = "jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, server_default="queued")
    payload = Column(JSONB, nullable=True)
    result = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class Webhook(Base):
    __tablename__ = "webhooks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    url = Column(Text, nullable=False)
    events = Column(ARRAY(String), nullable=False)
    secret = Column(Text, nullable=False)
    active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    deliveries = relationship("WebhookDelivery", back_populates="webhook", cascade="all, delete-orphan")


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id = Column(UUID(as_uuid=True), ForeignKey("webhooks.id", ondelete="CASCADE"), nullable=False)
    event = Column(String(100), nullable=False)
    payload = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False, server_default="pending")
    attempts = Column(Integer, nullable=False, server_default="0")
    last_error = Column(Text, nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    webhook = relationship("Webhook", back_populates="deliveries")


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    meta = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)


class LlmUsage(Base):
    __tablename__ = "llm_usage"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    invoice_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    model = Column(String(100), nullable=False)
    agent = Column(String(50), nullable=False)
    input_tokens = Column(Integer, nullable=False, server_default="0")
    output_tokens = Column(Integer, nullable=False, server_default="0")
    latency_ms = Column(Integer, nullable=False, server_default="0")
    cost_usd = Column(Numeric(10, 6), nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
