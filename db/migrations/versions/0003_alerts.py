"""alerts table for developer alerting

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-15
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("severity", sa.String(10), nullable=False),
        sa.Column("source", sa.String(50), nullable=False, index=True),
        sa.Column("event", sa.String(100), nullable=False),
        sa.Column("detail", sa.Text, nullable=False),
        sa.Column("context", postgresql.JSONB, nullable=True),
        sa.Column("fingerprint", sa.String(64), nullable=False, index=True),
        sa.Column("delivery_status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("delivery_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text, nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("alerts")
