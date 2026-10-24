"""add role_branches table

ARG-311: 확정된 역할 갈래(역할 축 위쪽 분류 틀)를 담는 테이블.

행이 존재하면 사람이 확정한 갈래다(초안은 JSON 파일이라 DB에 없다). 위 갈래는
``parent_id IS NULL``. 이름은 전역 유일 — 명령에서 이름으로 가리킨다. 부모 삭제는
CASCADE지만, 자동 처리는 이 테이블을 쓰지 않고 ``argos taxonomy apply``만 쓴다.

기존 테이블은 건드리지 않는다.

Revision ID: 7c3d9a1e5b42
Revises: 62e3b680eb74
Create Date: 2026-10-23 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7c3d9a1e5b42'
down_revision: Union[str, Sequence[str], None] = '62e3b680eb74'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'role_branches',
        sa.Column('parent_id', sa.UUID(), nullable=True),
        sa.Column('name', sa.String(length=80), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('position', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('confirmed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['parent_id'], ['role_branches.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )
    op.create_index(op.f('ix_role_branches_parent_id'), 'role_branches', ['parent_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_role_branches_parent_id'), table_name='role_branches')
    op.drop_table('role_branches')
