"""initial schema

Revision ID: de47078f87d5
Revises: 
Create Date: 2026-04-23 17:17:45.635645

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = 'de47078f87d5'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    op.create_table('tech_items',
    sa.Column('title', sa.String(length=500), nullable=False),
    sa.Column('source_url', sa.String(length=2048), nullable=False),
    sa.Column('raw_content', sa.Text(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1536), nullable=True),
    sa.Column('category', sa.Enum('Mainstream', 'Alpha', name='category_type'), nullable=True),
    sa.Column('trust_score', sa.Float(), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source_url')
    )
    op.create_table('tech_succession',
    sa.Column('predecessor_id', sa.UUID(), nullable=False),
    sa.Column('successor_id', sa.UUID(), nullable=False),
    sa.Column('relation_type', sa.Enum('Replace', 'Enhance', 'Fork', name='relation_type'), nullable=False),
    sa.Column('reasoning', sa.Text(), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['predecessor_id'], ['tech_items.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['successor_id'], ['tech_items.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user_assets',
    sa.Column('tech_id', sa.UUID(), nullable=False),
    sa.Column('status', sa.Enum('Keep', 'Tracking', 'Archived', name='asset_status'), server_default='Keep', nullable=False),
    sa.Column('last_monitored_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['tech_id'], ['tech_items.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('track_history',
    sa.Column('user_asset_id', sa.UUID(), nullable=False),
    sa.Column('changed_from', sa.String(length=50), nullable=False),
    sa.Column('changed_to', sa.String(length=50), nullable=False),
    sa.Column('changed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['user_asset_id'], ['user_assets.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('track_history')
    op.drop_table('user_assets')
    op.drop_table('tech_succession')
    op.drop_table('tech_items')

    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute(sa.text('DROP TYPE IF EXISTS asset_status'))
        op.execute(sa.text('DROP TYPE IF EXISTS relation_type'))
        op.execute(sa.text('DROP TYPE IF EXISTS category_type'))
