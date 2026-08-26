"""간선 가중치와 사건 선택 — ARG-264. DB를 쓰지 않는다."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from argos.brain.event_scoring import (
    DocumentFeatures,
    EdgeWeights,
    NeighborEdge,
    choose_event,
    edge_weight,
)

_NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)
_WEIGHTS = EdgeWeights(cosine=0.55, entity=0.25, time=0.15, keyword=0.05)


def _doc(*, embedding, names=(), at=_NOW, keywords=()):
    return DocumentFeatures(
        embedding=tuple(embedding) if embedding is not None else None,
        names=frozenset(names),
        at=at,
        keywords=frozenset(keywords),
    )


def test_identical_documents_score_one():
    doc = _doc(embedding=[1.0, 0.0], names=["anthropic"], keywords=["claude"])
    assert edge_weight(doc, doc, weights=_WEIGHTS, window_days=14) == pytest.approx(1.0)


def test_orthogonal_documents_with_nothing_in_common_score_low():
    left = _doc(embedding=[1.0, 0.0], names=["anthropic"])
    right = _doc(embedding=[0.0, 1.0], names=["mistral"])
    # 코사인 0, 이름 0, 키워드 0, 시간만 1.0 → 0.15
    assert edge_weight(left, right, weights=_WEIGHTS, window_days=14) == pytest.approx(0.15)


def test_names_move_the_score_even_when_embeddings_match():
    """이름 항이 판정을 실제로 뒤집는가 — 빈 DB 첫 기사부터 걸리는 안전장치."""
    shared = _doc(embedding=[1.0, 0.0], names=["anthropic"])
    disjoint = _doc(embedding=[1.0, 0.0], names=["mistral"])
    same_names = _doc(embedding=[1.0, 0.0], names=["anthropic"])
    assert edge_weight(shared, same_names, weights=_WEIGHTS, window_days=14) > edge_weight(
        shared, disjoint, weights=_WEIGHTS, window_days=14
    )


def test_time_decay_reaches_zero_at_the_window_edge():
    left = _doc(embedding=[1.0, 0.0], at=_NOW)
    right = _doc(embedding=[1.0, 0.0], at=_NOW - timedelta(days=14))
    # 코사인 1.0 × 0.55만 남는다
    assert edge_weight(left, right, weights=_WEIGHTS, window_days=14) == pytest.approx(0.55)


def test_a_missing_embedding_does_not_inflate_the_other_terms():
    """코사인 항 가중치를 분모에서 빼면 임베딩 없는 문서가 더 잘 묶인다."""
    without = _doc(embedding=None, names=["anthropic"], keywords=["claude"])
    with_names = _doc(embedding=[1.0, 0.0], names=["anthropic"], keywords=["claude"])
    assert edge_weight(without, with_names, weights=_WEIGHTS, window_days=14) == pytest.approx(0.45)


def test_weights_that_do_not_sum_to_one_are_normalised():
    doubled = EdgeWeights(cosine=1.1, entity=0.5, time=0.3, keyword=0.1)
    doc = _doc(embedding=[1.0, 0.0], names=["anthropic"], keywords=["claude"])
    assert edge_weight(doc, doc, weights=doubled, window_days=14) == pytest.approx(1.0)


def test_choose_event_sums_weight_per_event():
    a, b = uuid.UUID(int=1), uuid.UUID(int=2)
    edges = [
        NeighborEdge(event_ids=(a,), weight=0.3),
        NeighborEdge(event_ids=(a,), weight=0.3),
        NeighborEdge(event_ids=(b,), weight=0.5),
    ]
    assert choose_event(edges, join_threshold=0.55) == a


def test_choose_event_returns_none_below_the_threshold():
    a = uuid.UUID(int=1)
    edges = [NeighborEdge(event_ids=(a,), weight=0.3)]
    assert choose_event(edges, join_threshold=0.55) is None


def test_a_zero_threshold_joins_even_a_zero_score_event():
    """join_threshold=0.0(config가 허용하는 최솟값)이면 0점 사건도 붙는다.

    동점 0점이 여럿이면 사건 id 오름차순으로 결정적으로 고른다 (PR #122 리뷰).
    """
    a, b = uuid.UUID(int=1), uuid.UUID(int=2)
    edges = [
        NeighborEdge(event_ids=(b,), weight=0.0),
        NeighborEdge(event_ids=(a,), weight=0.0),
    ]
    assert choose_event(edges, join_threshold=0.0) == a
    assert choose_event(edges, join_threshold=0.1) is None


def test_the_threshold_actually_changes_the_outcome():
    """임계값을 설정으로 바꾸면 묶이는 정도가 달라진다 (부모 AC)."""
    a = uuid.UUID(int=1)
    edges = [NeighborEdge(event_ids=(a,), weight=0.4)]
    assert choose_event(edges, join_threshold=0.3) == a
    assert choose_event(edges, join_threshold=0.5) is None


def test_ties_break_deterministically_by_event_id():
    a, b = uuid.UUID(int=1), uuid.UUID(int=2)
    edges = [NeighborEdge(event_ids=(b,), weight=0.6), NeighborEdge(event_ids=(a,), weight=0.6)]
    assert choose_event(edges, join_threshold=0.5) == a
    assert choose_event(list(reversed(edges)), join_threshold=0.5) == a


def test_a_neighbour_in_two_events_contributes_to_both():
    a, b = uuid.UUID(int=1), uuid.UUID(int=2)
    edges = [NeighborEdge(event_ids=(a, b), weight=0.6)]
    assert choose_event(edges, join_threshold=0.5) == a


def test_no_neighbours_means_a_new_event():
    assert choose_event([], join_threshold=0.5) is None
