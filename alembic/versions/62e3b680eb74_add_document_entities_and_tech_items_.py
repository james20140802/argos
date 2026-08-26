"""add document_entities and tech_items.simhash

ARG-262 (Task 1 of the online-event-assignment plan): opens the two storage
slots later tasks in that plan need.

(a) ``document_entities`` — a document↔entity N:N link table, mirroring
``event_documents``/``event_entities`` but hung off ``tech_items`` directly
rather than off ``tech_events``. This lets a document's extracted entity
names feed edge-weight name matching even before any event exists for it —
see ``src/argos/models/document_entity.py`` for the full rationale.

(b) ``tech_items.simhash`` — a nullable ``BIGINT`` column. It is nullable so
this revision deploys with **no backfill**: existing rows simply get NULL,
and backfilling the historical corpus is a separate later task. Values are
unsigned 64-bit SimHash digests folded into signed BIGINT range by
``argos.brain.simhash_storage.to_storage``/``from_storage`` — see that
module's docstring for why the fold is safe (Hamming distance only looks at
the bit pattern, not the sign interpretation).

This revision does not touch any existing column on ``tech_items`` — the only
operation against that table is the single ``op.add_column`` below — so an
upgrade/downgrade round trip leaves every pre-existing row's id and
``source_url`` untouched.

Revision ID: 62e3b680eb74
Revises: 855cb67b5096
Create Date: 2026-08-26 13:56:07.667401

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '62e3b680eb74'
down_revision: Union[str, Sequence[str], None] = '855cb67b5096'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'document_entities',
        sa.Column('tech_item_id', sa.UUID(), nullable=False),
        sa.Column('entity_id', sa.UUID(), nullable=False),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['entity_id'], ['entities.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tech_item_id'], ['tech_items.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tech_item_id', 'entity_id', name='uq_document_entities_item_entity'),
    )
    op.create_index(op.f('ix_document_entities_entity_id'), 'document_entities', ['entity_id'], unique=False)
    op.add_column('tech_items', sa.Column('simhash', sa.BigInteger(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('tech_items', 'simhash')
    op.drop_index(op.f('ix_document_entities_entity_id'), table_name='document_entities')
    op.drop_table('document_entities')
