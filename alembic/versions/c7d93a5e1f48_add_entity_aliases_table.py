"""add entity_aliases table

ARG-312: "같은 기술의 다른 이름"을 하나로 잇는 별칭 층. ``entities``와
``document_entities``는 사건 묶기가 읽는 입력이라 건드리지 않고, 그 위에 이름 →
대표 이름 링크 테이블만 새로 만든다.

- ``alias_entity_id`` UNIQUE: 한 이름은 별칭 행을 최대 하나만 가진다.
- 두 FK 모두 ``ON DELETE CASCADE``이지만 지워지는 건 **별칭 행뿐**이다. 별칭
  테이블은 ``entities``를 참조할 뿐 그 반대가 아니라서, ``entities``나
  ``document_entities`` 행은 이 테이블 때문에 사라지지 않는다.
- CHECK: 자기 자신을 대표로 삼는 행을 막는다.

새 테이블과 새 enum(``alias_rule``)만 만든다 — 기존 테이블·열은 건드리지 않으므로
왕복(upgrade -> downgrade -> upgrade)에서 기존 행이 보존된다. downgrade는 테이블과
enum을 지운다.

주의: ARG-311(갈래 저장)도 같은 head(62e3b680eb74)에서 리비전을 낸다. 311이 먼저
머지되면 이 리비전을 main에 rebase하고 ``down_revision``을 311 리비전으로 고쳐
``alembic heads``가 한 줄이 되게 한다(merge 리비전은 만들지 않는다).

Revision ID: c7d93a5e1f48
Revises: 62e3b680eb74
Create Date: 2026-10-23 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c7d93a5e1f48'
down_revision: Union[str, Sequence[str], None] = '62e3b680eb74'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'entity_aliases',
        sa.Column('alias_entity_id', sa.UUID(), nullable=False),
        sa.Column('canonical_entity_id', sa.UUID(), nullable=False),
        sa.Column('rule', sa.Enum('Spacing', 'VendorPrefix', 'Manual', name='alias_rule'), nullable=False),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('alias_entity_id <> canonical_entity_id', name='ck_entity_aliases_not_self'),
        sa.ForeignKeyConstraint(['alias_entity_id'], ['entities.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['canonical_entity_id'], ['entities.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('alias_entity_id'),
    )
    op.create_index(op.f('ix_entity_aliases_canonical_entity_id'), 'entity_aliases', ['canonical_entity_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_entity_aliases_canonical_entity_id'), table_name='entity_aliases')
    op.drop_table('entity_aliases')
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute(sa.text('DROP TYPE IF EXISTS alias_rule'))
