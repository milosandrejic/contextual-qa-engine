"""add latency_ms to messages table

Revision ID: 01a3f7c9a5d2
Revises: 399db2ff3d3c
Create Date: 2026-04-29 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "01a3f7c9a5d2"
down_revision: Union[str, None] = "399db2ff3d3c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("latency_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "latency_ms")
