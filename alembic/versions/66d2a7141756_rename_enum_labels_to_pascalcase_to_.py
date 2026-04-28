"""rename enum labels to PascalCase to match ORM values

Revision ID: 66d2a7141756
Revises: 997853887d46
Create Date: 2026-04-28 22:01:24.292410

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '66d2a7141756'
down_revision: Union[str, Sequence[str], None] = '997853887d46'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (enum_type, uppercase_label, pascalcase_label)
_RENAMES: tuple[tuple[str, str, str], ...] = (
    ("category_type", "MAINSTREAM", "Mainstream"),
    ("category_type", "ALPHA", "Alpha"),
    ("relation_type", "REPLACE", "Replace"),
    ("relation_type", "ENHANCE", "Enhance"),
    ("relation_type", "FORK", "Fork"),
    ("asset_status", "KEEP", "Keep"),
    ("asset_status", "TRACKING", "Tracking"),
    ("asset_status", "ARCHIVED", "Archived"),
)


def upgrade() -> None:
    """Upgrade schema."""
    for enum_name, old, new in _RENAMES:
        op.execute(f"ALTER TYPE {enum_name} RENAME VALUE '{old}' TO '{new}'")


def downgrade() -> None:
    """Downgrade schema."""
    for enum_name, old, new in _RENAMES:
        op.execute(f"ALTER TYPE {enum_name} RENAME VALUE '{new}' TO '{old}'")
