"""배정 대상 고르기 (ARG-313) — 순수 함수, DB 불필요."""
from __future__ import annotations

from argos.brain.tech_classify import ExistingProfile
from argos.brain.tech_classify_store import NameRow, needs_refine, select_targets
from argos.config import TaxonomyConfig
from argos.models.role_assignment import AssignmentSource
from argos.models.technology_profile import TechVerdict

CFG = TaxonomyConfig()


def prof(verdict=TechVerdict.TECHNOLOGY, described=4):
    return ExistingProfile(verdict, None, "k", "설명", "role", described)


def row(key, docs, profile=None, source=None):
    return NameRow(entity_id=key, key=key, name=key, doc_count=docs, profile=profile, assignment_source=source)


def test_daily_batch_picks_new_and_refine_and_orphans_only():
    rows = [
        row("new", 3),
        row("rare", 1),  # 문서 1개 — 대상 아님
        row("stable", 5, prof(described=4), AssignmentSource.AUTO),  # 그대로 둔다
        row("grown", 10, prof(described=4), AssignmentSource.AUTO),  # 2배 이상 -> 다시 설명
        row("orphan", 5, prof(described=4), None),  # 설명은 있는데 배정 행 없음 -> 배정만
        row("manual", 50, None, AssignmentSource.MANUAL),  # 사람이 고친 배정은 건드리지 않음
        row("nottech", 5, prof(TechVerdict.NOT_TECHNOLOGY, 4), None),
    ]
    got = [(r.key, d) for r, d in select_targets(rows, CFG)]
    assert got == [("grown", True), ("orphan", False), ("new", True)]


def test_all_reassigns_everything_except_manual_and_rare_names():
    rows = [
        row("stable", 5, prof(described=4), AssignmentSource.AUTO),
        row("manual", 50, None, AssignmentSource.MANUAL),
        row("rare", 1),
        row("new", 3),
    ]
    got = [(r.key, d) for r, d in select_targets(rows, CFG, all_names=True)]
    assert got == [("stable", False), ("new", True)]


def test_limit_keeps_the_most_documented():
    rows = [row(f"n{i}", i + 2) for i in range(10)]
    got = [r.key for r, _d in select_targets(rows, CFG, limit=3)]
    assert got == ["n9", "n8", "n7"]


def test_refine_rule_ratio_or_absolute_growth():
    assert needs_refine(row("a", 3), CFG)  # 설명 없음
    assert not needs_refine(row("a", 5, prof(described=4)), CFG)
    assert needs_refine(row("a", 8, prof(described=4)), CFG)  # 2배
    assert needs_refine(row("a", 25, prof(described=20)), CFG)  # +5
