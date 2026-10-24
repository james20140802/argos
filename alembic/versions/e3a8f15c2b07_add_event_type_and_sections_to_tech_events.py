"""add event_type and sections to tech_events

ARG-308: 사건 요약이 "제목 + 한 단락"에서 "짧은 요약 + 사건 유형별 소제목
상세"로 바뀐다.

- ``event_type`` String(16) NULL + CHECK(Launch/Research/Incident/Update/Adoption,
  PascalCase). 소제목 목록을 고르는 사건 성격.
- ``sections`` JSONB NULL ``[{heading, body}]``. NULL은 "새 형식으로 아직 요약되지
  않음"이라 ``argos resummarize-events``가 줍는다.

새 열 두 개만 더한다 — 기존 열·행은 건드리지 않으므로 왕복(upgrade ->
downgrade -> upgrade)에서 기존 행이 보존된다. downgrade는 CHECK와 두 열을 지운다.

Revision ID: e3a8f15c2b07
Revises: c7d93a5e1f48
Create Date: 2026-10-25 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'e3a8f15c2b07'
down_revision: Union[str, Sequence[str], None] = 'c7d93a5e1f48'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('tech_events', sa.Column('event_type', sa.String(length=16), nullable=True))
    op.add_column(
        'tech_events',
        sa.Column('sections', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_check_constraint(
        'ck_tech_events_event_type',
        'tech_events',
        "event_type IN ('Launch', 'Research', 'Incident', 'Update', 'Adoption')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('ck_tech_events_event_type', 'tech_events', type_='check')
    op.drop_column('tech_events', 'sections')
    op.drop_column('tech_events', 'event_type')
