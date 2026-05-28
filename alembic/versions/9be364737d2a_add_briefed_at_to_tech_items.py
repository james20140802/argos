"""add briefed_at to tech_items

Revision ID: 9be364737d2a
Revises: 337debaadf4a
Create Date: 2026-05-29 00:36:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9be364737d2a'
down_revision: Union[str, Sequence[str], None] = '337debaadf4a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('tech_items', sa.Column('briefed_at', sa.DateTime(timezone=True), nullable=True))
    op.create_index(op.f('ix_tech_items_briefed_at'), 'tech_items', ['briefed_at'], unique=False)
    # Backfill existing rows so the "briefed_at IS NULL" filter in briefing_query
    # doesn't re-send all pre-migration items on the first run after upgrade.
    op.execute("UPDATE tech_items SET briefed_at = NOW() WHERE briefed_at IS NULL")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_tech_items_briefed_at'), table_name='tech_items')
    op.drop_column('tech_items', 'briefed_at')
