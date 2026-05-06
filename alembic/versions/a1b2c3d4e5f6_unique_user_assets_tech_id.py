"""unique constraint on user_assets.tech_id

Revision ID: a1b2c3d4e5f6
Revises: 66d2a7141756
Create Date: 2026-05-06 03:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '66d2a7141756'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Drops duplicate user_assets rows (keeping the most recently updated row
    per tech_id) before installing the unique constraint, so the migration
    succeeds on databases that already raced before the fix landed.
    """
    op.execute(
        """
        DELETE FROM user_assets a
        USING user_assets b
        WHERE a.tech_id = b.tech_id
          AND (a.updated_at, a.id) < (b.updated_at, b.id);
        """
    )
    op.create_unique_constraint(
        "uq_user_assets_tech_id", "user_assets", ["tech_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "uq_user_assets_tech_id", "user_assets", type_="unique"
    )
