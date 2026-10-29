"""add push_subscriptions and push_dispatches

ARG-321: 아침 브리핑 웹 푸시의 저장 층.

- ``push_subscriptions``: 알림을 켠 기기 하나당 한 행(``endpoint`` UNIQUE).
- ``push_dispatches``: 기기들에 보낸 브리핑 회차마다 한 행(전부 실패한 회차 포함).
  ``event_ids``에 그 회차가 담은 사건을 남겨, 한 기기라도 받은 사건은 다음 회차에서 뺀다.

새 테이블만 만들고 기존 테이블·열은 건드리지 않는다. FK 없음.
downgrade는 두 테이블을 지운다.

Revision ID: f5b21c7e9a30
Revises: e3a9c4d17b20
Create Date: 2026-10-29 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'f5b21c7e9a30'
down_revision: Union[str, Sequence[str], None] = 'e3a9c4d17b20'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'push_subscriptions',
        sa.Column('endpoint', sa.Text(), nullable=False),
        sa.Column('p256dh', sa.String(length=255), nullable=False),
        sa.Column('auth', sa.String(length=255), nullable=False),
        sa.Column('label', sa.String(length=100), nullable=False),
        sa.Column('origin', sa.String(length=255), nullable=False),
        sa.Column('last_success_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column(
            'created_at', sa.DateTime(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('endpoint'),
    )
    op.create_table(
        'push_dispatches',
        sa.Column(
            'sent_at', sa.DateTime(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.Column('event_count', sa.Integer(), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('devices_ok', sa.Integer(), nullable=False),
        sa.Column('devices_failed', sa.Integer(), nullable=False),
        sa.Column('devices_expired', sa.Integer(), nullable=False),
        sa.Column(
            'event_ids', postgresql.ARRAY(sa.UUID()),
            server_default=sa.text("'{}'"), nullable=False,
        ),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_push_dispatches_sent_at'), 'push_dispatches', ['sent_at'], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_push_dispatches_sent_at'), table_name='push_dispatches')
    op.drop_table('push_dispatches')
    op.drop_table('push_subscriptions')
