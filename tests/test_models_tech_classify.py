"""ARG-313 모델 단위 테스트 — technology_profiles / role_assignments (DB 불필요)."""
from __future__ import annotations

from argos.models import RoleAssignment, TechnologyProfile
from argos.models.role_assignment import AssignmentSource
from argos.models.technology_profile import TechVerdict


def _fk(table, column):
    (fk,) = table.c[column].foreign_keys
    return fk


def test_enum_values_are_pascal_case():
    assert [v.value for v in TechVerdict] == ["Technology", "NotTechnology", "Insufficient"]
    assert [v.value for v in AssignmentSource] == ["Auto", "Manual"]


def test_profile_one_row_per_entity_with_cascade():
    t = TechnologyProfile.__table__
    assert t.name == "technology_profiles"
    assert t.c.entity_id.unique
    fk = _fk(t, "entity_id")
    assert fk.column.table.name == "entities"
    assert fk.ondelete == "CASCADE"
    for col in ("verdict", "not_tech_reason", "kind", "description", "role_en",
                "described_doc_count", "described_at", "model"):
        assert col in t.c


def test_assignment_branch_is_nullable_and_all_fks_cascade():
    t = RoleAssignment.__table__
    assert t.name == "role_assignments"
    assert t.c.entity_id.unique
    assert t.c.branch_id.nullable  # NULL = 미분류
    assert _fk(t, "entity_id").ondelete == "CASCADE"
    branch_fk = _fk(t, "branch_id")
    assert branch_fk.column.table.name == "role_branches"
    assert branch_fk.ondelete == "CASCADE"
    for col in ("evidence", "unclassified_reason", "source", "assigned_at", "run_id"):
        assert col in t.c
