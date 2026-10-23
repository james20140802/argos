"""확정본과 새 트리의 차이 계산 (ARG-311 T4) — 순수 함수, DB 불필요."""
from __future__ import annotations

import pytest

from argos.brain import taxonomy as tx
from argos.brain.taxonomy import Branch, TaxonomyError
from argos.brain.taxonomy_store import plan_apply

DESC = "설명은 열 글자를 넘는 한 문장."


def B(name, children=(), id=None, desc=DESC):
    return Branch(name=name, description=desc, children=tuple(children), id=id)


def confirmed():
    return (
        B("모델", [B("기초 모델", id="c1"), B("특화 모델", id="c2")], id="t1"),
        B("에이전트", [B("코딩 에이전트", id="c3")], id="t2"),
        B("인프라", [B("저장소", id="c4"), B("런타임", id="c5")], id="t3"),
        B("평가", [B("벤치마크", id="c6")], id="t4"),
    )


def test_first_confirmation_adds_everything():
    plan = plan_apply(None, tx.SPIKE_SEED)
    assert not plan.is_empty
    assert len(plan.added) == 5 + 11
    assert not plan.removed and not plan.renamed
    assert ("AI 모델 및 엔진", None) in plan.added


def test_unchanged_tree_is_empty_plan():
    assert plan_apply(confirmed(), confirmed()).is_empty


def test_rename_is_rename_not_delete_plus_add():
    new = tx.rename(confirmed(), "기초 모델", "범용 모델")
    plan = plan_apply(confirmed(), new)
    assert plan.renamed == (("기초 모델", "범용 모델"),)
    assert not plan.added and not plan.removed


def test_merge_removes_absorbed_and_moves_children():
    new = tx.merge(confirmed(), "에이전트", "모델")
    plan = plan_apply(confirmed(), new)
    assert plan.removed == ("에이전트",)
    assert plan.moved == (("코딩 에이전트", "에이전트", "모델"),)
    assert plan.removed_ids == frozenset({"t2"})
    assert not plan.added


def test_description_change_is_detected():
    new = tx.redescribe(confirmed(), "저장소", "완전히 새로 쓴 설명 문장입니다.")
    assert plan_apply(confirmed(), new).redescribed == ("저장소",)


def test_split_adds_new_and_removes_original():
    new = tx.split(
        confirmed(),
        "저장소",
        [("데이터베이스", DESC), ("버전 관리", DESC)],
    )
    plan = plan_apply(confirmed(), new)
    assert plan.removed == ("저장소",)
    assert {a[0] for a in plan.added} == {"데이터베이스", "버전 관리"}


def test_reorder_only_is_reported_once():
    cur = confirmed()
    new = (cur[1], cur[0], cur[2], cur[3])
    plan = plan_apply(cur, new)
    assert not plan.is_empty
    assert set(plan.reordered) == {"모델", "에이전트"}


def test_invalid_new_tree_is_rejected():
    with pytest.raises(TaxonomyError):
        plan_apply(confirmed(), confirmed()[:2])


def test_unknown_ids_from_stale_file_are_rejected():
    stale = list(confirmed())
    stale[0] = B("모델", [B("기초 모델", id="zzz")], id="t1")
    with pytest.raises(TaxonomyError) as ei:
        plan_apply(confirmed(), tuple(stale))
    assert "export" in str(ei.value)


def test_summary_lines_mention_each_change():
    new = tx.rename(confirmed(), "기초 모델", "범용 모델")
    text = "\n".join(plan_apply(confirmed(), new).summary_lines())
    assert "기초 모델" in text and "범용 모델" in text
