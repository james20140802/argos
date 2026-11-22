"""add event_deep_analyses

ARG-333: 사건별 심층 분석의 저장 층.

- ``event_deep_analyses``: 사건 하나당 한 행(``event_id`` UNIQUE, 사건이 지워지면 CASCADE).
  한 행에 "최근 성공 결과"(sections·analyzed_at·doc_count·doc_ids·model)와
  "지금 진행 상태"(status·requested_at·started_at·failure_reason)를 따로 둔다.
  둘은 전부 NULL 허용이라 서로 독립으로 비울 수 있다.
- 새 enum ``deep_analysis_status``(Waiting, Running, Failed). ``status`` NULL = 없음.

새 테이블과 새 enum만 만들고 기존 테이블·열은 건드리지 않는다.
downgrade는 테이블과 enum을 지운다(저장된 분석 결과는 사라진다).

Revision ID: a7c41e9d2b63
Revises: f5b21c7e9a30
Create Date: 2026-11-22 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a7c41e9d2b63'
down_revision: Union[str, Sequence[str], None] = 'f5b21c7e9a30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'event_deep_analyses',
        sa.Column('event_id', sa.UUID(), nullable=False),
        sa.Column('sections', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('analyzed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('doc_count', sa.Integer(), nullable=True),
        sa.Column('doc_ids', postgresql.ARRAY(sa.UUID()), nullable=True),
        sa.Column('model', sa.String(length=100), nullable=True),
        sa.Column(
            'status',
            sa.Enum('Waiting', 'Running', 'Failed', name='deep_analysis_status'),
            nullable=True,
        ),
        sa.Column('requested_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failure_reason', sa.String(length=200), nullable=True),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column(
            'created_at', sa.DateTime(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.ForeignKeyConstraint(['event_id'], ['tech_events.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('event_id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('event_deep_analyses')
    op.execute(sa.text('DROP TYPE IF EXISTS deep_analysis_status'))
