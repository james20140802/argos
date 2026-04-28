"""shrink tech_items.embedding to 768 dims for nomic-embed-text

Revision ID: 997853887d46
Revises: de47078f87d5
Create Date: 2026-04-28 19:17:03.531763

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = '997853887d46'
down_revision: Union[str, Sequence[str], None] = 'de47078f87d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # pgvector cannot coerce existing 1536-d values to 768-d, so clear them
    # before resizing. Embeddings must be regenerated with nomic-embed-text.
    op.execute("UPDATE tech_items SET embedding = NULL WHERE embedding IS NOT NULL")
    op.alter_column('tech_items', 'embedding',
               existing_type=pgvector.sqlalchemy.vector.VECTOR(dim=1536),
               type_=pgvector.sqlalchemy.vector.VECTOR(dim=768),
               existing_nullable=True,
               postgresql_using='NULL::vector(768)')


def downgrade() -> None:
    """Downgrade schema."""
    # Same story in reverse: 768-d values cannot be coerced back to 1536-d.
    op.execute("UPDATE tech_items SET embedding = NULL WHERE embedding IS NOT NULL")
    op.alter_column('tech_items', 'embedding',
               existing_type=pgvector.sqlalchemy.vector.VECTOR(dim=768),
               type_=pgvector.sqlalchemy.vector.VECTOR(dim=1536),
               existing_nullable=True,
               postgresql_using='NULL::vector(1536)')
