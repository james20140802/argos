"""feed_score index desc nulls last

Revision ID: 16778301d264
Revises: 4ac05a69769b
Create Date: 2026-07-16 19:20:08.044861

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '16778301d264'
down_revision: Union[str, Sequence[str], None] = '4ac05a69769b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    The recommended feed sort is ``feed_score DESC NULLS LAST, ...`` (see
    ``argos.web.services.feed.fetch_feed``). The original plain ascending btree
    ``ix_tech_items_feed_score`` cannot serve that ordering via an index scan —
    a DESC-NULLS-LAST scan needs an index whose own order matches — so the
    planner falls back to a full sort. Recreate the index with matching
    ``DESC NULLS LAST`` order so it can drive the leading-column ordering
    (with an incremental sort for the recency/id tiebreak).
    """
    op.drop_index(op.f('ix_tech_items_feed_score'), table_name='tech_items')
    op.create_index(
        'ix_tech_items_feed_score',
        'tech_items',
        [sa.text('feed_score DESC NULLS LAST')],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_tech_items_feed_score', table_name='tech_items')
    op.create_index(
        op.f('ix_tech_items_feed_score'), 'tech_items', ['feed_score'], unique=False
    )
