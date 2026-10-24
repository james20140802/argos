"""add technology_profiles and role_assignments

ARG-313: 새 기술을 승인 없이 역할 아래 갈래에 자동 배정하는 배치의 저장 층.

- ``technology_profiles``: 대표 이름(entities 행)마다 기술 판정·설명 한 행
  (``entity_id`` UNIQUE). ``entities.kind``는 건드리지 않는다.
- ``role_assignments``: 대표 이름마다 배정 한 행(``entity_id`` UNIQUE).
  ``branch_id`` NULL = 미분류. 근거는 ``evidence`` JSONB.
- FK는 전부 ``ON DELETE CASCADE``. ``role_branches``는 참조만 한다 — 갈래가
  지워지면 배정 행만 사라진다.

새 테이블과 새 enum(``tech_verdict``, ``assignment_source``)만 만든다. 기존 테이블·열은
건드리지 않으므로 왕복에서 기존 행이 보존된다. downgrade는 테이블과 enum을 지운다.

Revision ID: e3a9c4d17b20
Revises: c7d93a5e1f48
Create Date: 2026-10-25 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e3a9c4d17b20'
down_revision: Union[str, Sequence[str], None] = 'c7d93a5e1f48'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'technology_profiles',
        sa.Column('entity_id', sa.UUID(), nullable=False),
        sa.Column(
            'verdict',
            sa.Enum('Technology', 'NotTechnology', 'Insufficient', name='tech_verdict'),
            nullable=False,
        ),
        sa.Column('not_tech_reason', sa.String(length=20), nullable=True),
        sa.Column('kind', sa.String(length=120), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('role_en', sa.Text(), nullable=True),
        sa.Column('described_doc_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('described_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('model', sa.String(length=100), nullable=True),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['entity_id'], ['entities.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('entity_id'),
    )
    op.create_table(
        'role_assignments',
        sa.Column('entity_id', sa.UUID(), nullable=False),
        sa.Column('branch_id', sa.UUID(), nullable=True),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('unclassified_reason', sa.String(length=40), nullable=True),
        sa.Column('source', sa.Enum('Auto', 'Manual', name='assignment_source'), nullable=False),
        sa.Column('assigned_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('run_id', sa.UUID(), nullable=True),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['branch_id'], ['role_branches.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['entity_id'], ['entities.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('entity_id'),
    )
    op.create_index(op.f('ix_role_assignments_branch_id'), 'role_assignments', ['branch_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_role_assignments_branch_id'), table_name='role_assignments')
    op.drop_table('role_assignments')
    op.drop_table('technology_profiles')
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        op.execute(sa.text('DROP TYPE IF EXISTS assignment_source'))
        op.execute(sa.text('DROP TYPE IF EXISTS tech_verdict'))
