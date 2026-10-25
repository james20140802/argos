"""역할 갈래 배정 평가 지표 (ARG-313) — DB/LLM 불필요.

시험판(2026-10-22)의 A·D 방식 배정 결과를 픽스처로 두고, 같은 정답 묶음으로
쌍 일치도를 다시 재서 시험판 숫자(D 0.81 / A 0.59)가 그대로 나오는지 본다.
지표가 시험판과 같다는 것을 보장해야 운영 실측(0.81 이상)이 같은 잣대가 된다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from argos.brain import taxonomy as tx
from argos.brain.taxonomy_eval import load_truth, pair_f1

ROOT = Path(__file__).resolve().parents[2]
FIX = Path(__file__).parent / "fixtures" / "taxonomy_spike"
TRUTH = ROOT / "evals" / "taxonomy_truth_arg313.json"


def _fixture(name: str) -> dict[str, str]:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_truth_file_has_the_33_spike_technologies():
    gold = load_truth(TRUTH)
    assert len(gold) == 33
    assert gold["Codex"] == gold["Claude Code"]
    assert gold["OpenRouter"] != gold["MCP"]


def test_spike_method_d_scores_081():
    score = pair_f1(_fixture("assignD.json"), load_truth(TRUTH))
    assert (round(score.precision, 2), round(score.recall, 2), round(score.f1, 2)) == (
        0.82,
        0.80,
        0.81,
    )
    assert score.missing == ()


def test_spike_method_a_scores_059():
    score = pair_f1(_fixture("assignA.json"), load_truth(TRUTH))
    assert (round(score.precision, 2), round(score.recall, 2), round(score.f1, 2)) == (
        0.70,
        0.51,
        0.59,
    )


def test_unclassified_and_missing_names_are_singletons():
    gold = {"a": "g1", "b": "g1", "c": "g2"}
    # a·b 둘 다 미분류(None)여도 같은 묶음으로 치지 않는다.
    score = pair_f1({"a": None, "b": None, "c": "x"}, gold)
    assert score.tp == 0 and score.fp == 0 and score.fn == 1
    assert score.f1 == 0.0
    # 빠진 이름은 단독 + 보고.
    score = pair_f1({"a": "x", "c": "x"}, gold)
    assert score.missing == ("b",)
    assert (score.tp, score.fp, score.fn) == (0, 1, 1)


def test_perfect_assignment_scores_one():
    gold = {"a": "g1", "b": "g1", "c": "g2", "d": "g2"}
    score = pair_f1({"a": "L1", "b": "L1", "c": "L2", "d": "L2"}, gold)
    assert score.f1 == pytest.approx(1.0)


def test_spike_tree_file_is_a_valid_tree_with_ten_leaves():
    # 평가 전용 고정 트리다. 시험판 이름을 그대로 둬서(40자 넘는 영문 병기) 확정용
    # validate 규칙은 일부러 적용하지 않는다 — DB에 들어가지 않는다.
    tree = tx.load_file(ROOT / "evals" / "taxonomy_spike_treeA.json")
    leaves = [b for d, b, _p in tx.iter_branches(tree) if d == 2]
    assert len(leaves) == 10
    # 픽스처의 갈래 이름이 모두 이 트리의 "위 > 아래" 경로다.
    paths = {f"{p.name} > {b.name}" for d, b, p in tx.iter_branches(tree) if d == 2}
    assert set(_fixture("assignD.json").values()) <= paths
