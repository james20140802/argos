"""add tech_items.digest

Revision ID: 5777c234802b
Revises: c4f1a8e9b2d0
Create Date: 2026-07-01 22:33:16.155068

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5777c234802b'
down_revision: Union[str, Sequence[str], None] = 'c4f1a8e9b2d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('tech_items', sa.Column('digest', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('tech_items', 'digest')
