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


def _rename_if_exists(enum_name: str, src: str, dst: str) -> None:
    # Fresh DBs created from the initial schema already have PascalCase
    # labels, so a blind RENAME VALUE would fail at the first statement.
    # Gate on the source label's existence in pg_enum so the migration is
    # a no-op on fresh installs and only repairs DBs created with the
    # uppercase labels.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_enum e
                JOIN pg_type t ON t.oid = e.enumtypid
                WHERE t.typname = '{enum_name}' AND e.enumlabel = '{src}'
            ) THEN
                EXECUTE 'ALTER TYPE {enum_name} RENAME VALUE ''{src}'' TO ''{dst}''';
            END IF;
        END$$;
        """
    )


def upgrade() -> None:
    """Upgrade schema."""
    for enum_name, old, new in _RENAMES:
        _rename_if_exists(enum_name, old, new)


def downgrade() -> None:
    """Downgrade schema."""
    for enum_name, old, new in _RENAMES:
        _rename_if_exists(enum_name, new, old)
