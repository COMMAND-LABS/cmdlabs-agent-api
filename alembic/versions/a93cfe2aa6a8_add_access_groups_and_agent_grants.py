"""add_access_groups_and_agent_grants

Create the three tables that support group-based agent sharing:
  - access_groups          (named groups owned by an account)
  - access_group_members   (accounts belonging to a group)
  - agent_access_grants    (agents shared with a group)

Revision ID: a93cfe2aa6a8
Revises:
Create Date: 2026-02-15 04:30:37.175310

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a93cfe2aa6a8'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create access_groups, access_group_members, agent_access_grants."""

    # ── access_groups ─────────────────────────────────────────────
    op.create_table(
        'access_groups',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column(
            'owner_account_id',
            sa.Integer,
            sa.ForeignKey('accounts.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index('ix_access_groups_id', 'access_groups', ['id'])
    op.create_index('ix_access_groups_owner_account_id', 'access_groups', ['owner_account_id'])

    # ── access_group_members ──────────────────────────────────────
    op.create_table(
        'access_group_members',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            'access_group_id',
            sa.Integer,
            sa.ForeignKey('access_groups.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column(
            'account_id',
            sa.Integer,
            sa.ForeignKey('accounts.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            'access_group_id', 'account_id',
            name='uq_access_group_member',
        ),
    )
    op.create_index('ix_access_group_members_id', 'access_group_members', ['id'])
    op.create_index('ix_access_group_members_access_group_id', 'access_group_members', ['access_group_id'])
    op.create_index('ix_access_group_members_account_id', 'access_group_members', ['account_id'])

    # ── agent_access_grants ───────────────────────────────────────
    op.create_table(
        'agent_access_grants',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            'agent_id',
            sa.Integer,
            sa.ForeignKey('agents.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column(
            'access_group_id',
            sa.Integer,
            sa.ForeignKey('access_groups.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            'agent_id', 'access_group_id',
            name='uq_agent_access_grant',
        ),
    )
    op.create_index('ix_agent_access_grants_id', 'agent_access_grants', ['id'])
    op.create_index('ix_agent_access_grants_agent_id', 'agent_access_grants', ['agent_id'])
    op.create_index('ix_agent_access_grants_access_group_id', 'agent_access_grants', ['access_group_id'])


def downgrade() -> None:
    """Drop agent_access_grants, access_group_members, access_groups (reverse order)."""
    op.drop_table('agent_access_grants')
    op.drop_table('access_group_members')
    op.drop_table('access_groups')
