"""add tech_items feed_score

Revision ID: 4ac05a69769b
Revises: 64ffd26e3c14
Create Date: 2026-07-13 03:33:02.881229

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4ac05a69769b'
down_revision: Union[str, Sequence[str], None] = '64ffd26e3c14'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('tech_items', sa.Column('feed_score', sa.Float(), nullable=True))
    op.create_index(op.f('ix_tech_items_feed_score'), 'tech_items', ['feed_score'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_tech_items_feed_score'), table_name='tech_items')
    op.drop_column('tech_items', 'feed_score')
