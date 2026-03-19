"""add_google_api_key_to_service_name_enum

Adds GOOGLE_API_KEY to the service_name_enum PostgreSQL type so that
Google Gemini credentials can be stored in the credentials table.

Revision ID: b1e4f9c2d307
Revises: a93cfe2aa6a8
Create Date: 2026-03-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'b1e4f9c2d307'
down_revision: Union[str, Sequence[str], None] = 'a93cfe2aa6a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # PostgreSQL requires ALTER TYPE ... ADD VALUE to extend an enum.
    # IF NOT EXISTS guards against re-running on a DB that already has the value.
    op.execute("ALTER TYPE service_name_enum ADD VALUE IF NOT EXISTS 'GOOGLE_API_KEY'")


def downgrade() -> None:
    # PostgreSQL does not support removing enum values without recreating the type.
    # A full recreate is intentionally omitted here; removing this value is a
    # destructive, rarely-needed operation that should be handled manually.
    pass
