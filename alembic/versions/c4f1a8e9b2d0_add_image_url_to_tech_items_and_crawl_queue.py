"""add image_url to tech_items and crawl_queue

Revision ID: c4f1a8e9b2d0
Revises: 9be364737d2a
Create Date: 2026-06-13 03:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c4f1a8e9b2d0'
down_revision: Union[str, Sequence[str], None] = '9be364737d2a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'tech_items',
        sa.Column('image_url', sa.String(length=2048), nullable=True),
    )
    op.add_column(
        'crawl_queue',
        sa.Column('image_url', sa.String(length=2048), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('crawl_queue', 'image_url')
    op.drop_column('tech_items', 'image_url')
