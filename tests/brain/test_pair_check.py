"""pair_check 순수 함수·오케스트레이션 — ARG-307. LLM은 전부 가짜, DB 없음."""
from __future__ import annotations

import uuid

import pytest

from argos.brain.pair_check import (
    build_prompt,
    input_sha,
    parse_verdict,
    select_ambiguous_pairs,
)
from argos.brain.recluster_core import WeightedEdge


def _id(n: int) -> uuid.UUID:
    return uuid.UUID(int=n)


def _edge(a: int, b: int, weight: float) -> WeightedEdge:
    low, high = sorted([_id(a), _id(b)])
    return WeightedEdge(left_id=low, right_id=high, weight=weight)


SITE = {_id(1): "openai.com", _id(2): "openai.com", _id(3): "openai.com",
        _id(4): "kimi.com", _id(5): None, _id(6): None}


def _select(edges, **kw):
    kw.setdefault("lower", 0.55)
    kw.setdefault("upper", 0.65)
    kw.setdefault("same_source_only", True)
    return select_ambiguous_pairs(edges, SITE, **kw)


def test_band_includes_lower_and_excludes_upper():
    edges = [_edge(1, 2, 0.55), _edge(1, 3, 0.65), _edge(2, 3, 0.6499), _edge(1, 2, 0.5499)]
    picked = {(e.left_id, e.right_id, e.weight) for e in _select(edges)}
    assert (_id(1), _id(2), 0.55) in picked
    assert (_id(2), _id(3), 0.6499) in picked
    assert all(w != 0.65 and w != 0.5499 for _, _, w in picked)


def test_unknown_source_is_never_the_same_source():
    assert _select([_edge(5, 6, 0.6), _edge(1, 5, 0.6)]) == ()


def test_cross_source_pairs_toggle():
    edges = [_edge(1, 4, 0.6)]
    assert _select(edges) == ()
    assert len(_select(edges, same_source_only=False)) == 1


def test_selection_is_order_independent():
    edges = [_edge(2, 3, 0.6), _edge(1, 2, 0.58), _edge(1, 3, 0.62)]
    assert _select(edges) == _select(list(reversed(edges)))
    assert [(e.left_id, e.right_id) for e in _select(edges)] == sorted(
        (e.left_id, e.right_id) for e in edges
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("SAME - 같은 출시 발표", ("SAME", "같은 출시 발표")),
        ("different: 다른 고객", ("DIFFERENT", "다른 고객")),
        ("<think>SAME? no</think>\nDIFFERENT\n이유 없음", ("DIFFERENT", "이유 없음")),
        ("DIFFERENT", ("DIFFERENT", "")),
    ],
)
def test_parse_verdict_reads_one_word(text, expected):
    assert parse_verdict(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "maybe", "SAME or DIFFERENT?", "Both same and different"])
def test_parse_verdict_unparseable_is_none(text):
    assert parse_verdict(text) is None


def test_prompt_contains_both_texts_and_clips_summaries():
    prompt = build_prompt(("제목A", "가" * 1000), ("제목B", "요약B"))
    assert "제목A" in prompt and "제목B" in prompt and "요약B" in prompt
    assert "가" * 401 not in prompt
    assert "SAME" in prompt and "DIFFERENT" in prompt


def test_input_sha_tracks_text_and_is_order_insensitive():
    a, b = ("t1", "s1"), ("t2", "s2")
    assert input_sha(a, b) == input_sha(b, a)
    assert input_sha(a, b) != input_sha(a, ("t2", "changed"))
