"""간선 가중치와 사건 선택 — ARG-264. DB를 쓰지 않는다."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from argos.brain.event_scoring import (
    DocumentFeatures,
    NEAR_DUPLICATE_COSINE,
    EdgeWeights,
    NeighborEdge,
    choose_event,
    edge_weight,
)

_NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)
_WEIGHTS = EdgeWeights(cosine=0.55, entity=0.25, time=0.15, keyword=0.05)


def _doc(*, embedding, names=(), at=_NOW, keywords=(), source=None):
    return DocumentFeatures(
        embedding=tuple(embedding) if embedding is not None else None,
        names=frozenset(names),
        at=at,
        keywords=frozenset(keywords),
        source=source,
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


# ── choose_event: 밤(CPM)이 받아들일 배정만 고른다 (ARG-282) ──────────────
#
# 기본값 기준: join_threshold = γ = 0.55. 사건 E에 붙이는 이득은
# "τ 이상인 이웃 점수의 합 − γ × E의 크기"이고, 이득이 0보다 커야 붙는다.

_TAU = 0.55
_GAMMA = 0.55


def _choose(edges, sizes, *, tau=_TAU, gamma=_GAMMA):
    return choose_event(edges, event_sizes=sizes, join_threshold=tau, resolution=gamma)


def test_weak_votes_no_longer_add_up_to_a_join():
    """이슈 실패 예시: 0.32짜리 이웃 셋(합 0.96)은 밤에 간선이 하나도 없다.

    예전 낮 규칙은 합이 0.55를 넘어 붙였다 — 실제 코퍼스 1,669건 중
    1,603건을 사건 하나로 뭉친 눈덩이가 바로 이 모양이었다.
    """
    a = uuid.UUID(int=1)
    edges = [NeighborEdge(event_ids=(a,), weight=0.32)] * 3
    assert _choose(edges, {a: 3}) is None


def test_time_only_neighbours_do_not_form_an_event():
    """이슈 실패 예시: 임베딩 없이 시각만 같은 이웃(각 0.15) 넷."""
    a = uuid.UUID(int=1)
    edges = [NeighborEdge(event_ids=(a,), weight=0.15)] * 4
    assert _choose(edges, {a: 4}) is None


def test_one_strong_neighbour_joins_a_single_document_event():
    a = uuid.UUID(int=1)
    assert _choose([NeighborEdge(event_ids=(a,), weight=0.6)], {a: 1}) == a


def test_one_strong_link_is_not_enough_for_a_large_event():
    """사건이 크면 그 사건 전체와 평균적으로 가까워야 붙는다 — 눈덩이 차단."""
    a = uuid.UUID(int=1)
    assert _choose([NeighborEdge(event_ids=(a,), weight=0.9)], {a: 10}) is None


def test_strong_links_to_most_of_an_event_join_it():
    a = uuid.UUID(int=1)
    edges = [NeighborEdge(event_ids=(a,), weight=0.6)] * 2
    # 1.2 - 0.55*2 = 0.1 > 0
    assert _choose(edges, {a: 2}) == a


def test_a_score_exactly_at_the_threshold_does_not_join():
    """CPM에서 이득 0은 붙일 이유가 없다 — 밤도 이 쌍을 하나로 두지 않는다."""
    a = uuid.UUID(int=1)
    assert _choose([NeighborEdge(event_ids=(a,), weight=0.55)], {a: 1}) is None


def test_the_best_gain_wins_not_the_biggest_sum():
    big, small = uuid.UUID(int=1), uuid.UUID(int=2)
    edges = [NeighborEdge(event_ids=(big,), weight=0.7)] * 3 + [
        NeighborEdge(event_ids=(small,), weight=0.6)
    ]
    # big: 2.1 - 5.5 < 0 / small: 0.6 - 0.55 = 0.05
    assert _choose(edges, {big: 10, small: 1}) == small


def test_an_unknown_size_counts_at_least_the_voting_neighbours():
    """크기를 모르는 사건을 0으로 치면 페널티가 사라져 무조건 붙는다."""
    a = uuid.UUID(int=1)
    edges = [NeighborEdge(event_ids=(a,), weight=0.6)] * 2 + [
        NeighborEdge(event_ids=(a,), weight=0.1)
    ]
    # 크기 3으로 본다: 1.2 - 1.65 < 0
    assert _choose(edges, {}) is None


def test_the_threshold_actually_changes_the_outcome():
    """임계값을 설정으로 바꾸면 묶이는 정도가 같은 방향으로 달라진다 (부모 AC)."""
    a = uuid.UUID(int=1)
    edges = [NeighborEdge(event_ids=(a,), weight=0.5)]
    assert _choose(edges, {a: 1}, tau=0.45, gamma=0.45) == a
    assert _choose(edges, {a: 1}, tau=0.55, gamma=0.55) is None


def test_ties_break_deterministically_by_event_id():
    a, b = uuid.UUID(int=1), uuid.UUID(int=2)
    edges = [NeighborEdge(event_ids=(b,), weight=0.6), NeighborEdge(event_ids=(a,), weight=0.6)]
    assert _choose(edges, {a: 1, b: 1}) == a
    assert _choose(list(reversed(edges)), {a: 1, b: 1}) == a


def test_a_neighbour_in_two_events_contributes_to_both():
    a, b = uuid.UUID(int=1), uuid.UUID(int=2)
    edges = [NeighborEdge(event_ids=(a, b), weight=0.6)]
    assert _choose(edges, {a: 1, b: 1}) == a


def test_zero_scores_never_join_even_at_a_zero_threshold():
    """τ=γ=0이어도 이득이 0이면 붙지 않는다 — 밤의 CPM도 0점 쌍을 묶지 않는다."""
    a = uuid.UUID(int=1)
    assert _choose([NeighborEdge(event_ids=(a,), weight=0.0)], {a: 1}, tau=0.0, gamma=0.0) is None


def test_no_neighbours_means_a_new_event():
    assert _choose([], {}) is None


# --- ARG-295: 같은 출처 보정 ---------------------------------------------------


def _penalty_weights(penalty):
    return EdgeWeights(
        cosine=0.55, entity=0.25, time=0.15, keyword=0.05, same_source_penalty=penalty
    )


def _pair(*, left_source, right_source, right_embedding=(0.8, 0.6)):
    left = _doc(embedding=[1.0, 0.0], names=["openai"], source=left_source)
    right = _doc(embedding=right_embedding, names=["openai"], source=right_source)
    return left, right


def _score(left, right, penalty):
    return edge_weight(left, right, weights=_penalty_weights(penalty), window_days=14)


def test_same_source_pair_loses_exactly_the_penalty():
    left, right = _pair(left_source="openai.com", right_source="openai.com")
    base = _score(left, right, 0.0)
    assert _score(left, right, 0.2) == pytest.approx(base - 0.2)


def test_zero_penalty_returns_the_uncorrected_score_exactly():
    left, right = _pair(left_source="openai.com", right_source="openai.com")
    plain = edge_weight(left, right, weights=_WEIGHTS, window_days=14)
    assert _score(left, right, 0.0) == plain


def test_different_or_unknown_source_is_not_corrected():
    base = _score(*_pair(left_source="a.com", right_source="a.com"), 0.0)
    for left_source, right_source in [("a.com", "b.com"), ("a.com", None), (None, None)]:
        pair = _pair(left_source=left_source, right_source=right_source)
        assert _score(*pair, 0.3) == base


def test_near_duplicate_same_source_pair_is_exempt():
    left, right = _pair(
        left_source="openai.com", right_source="openai.com", right_embedding=(1.0, 0.0)
    )
    base = _score(left, right, 0.0)
    assert _score(left, right, 0.3) == base
    assert NEAR_DUPLICATE_COSINE == 0.95


def test_corrected_score_never_goes_negative():
    weak = edge_weight(
        _doc(embedding=[1.0, 0.0], source="x.com", at=_NOW - timedelta(days=30)),
        _doc(embedding=[0.0, 1.0], source="x.com"),
        weights=_penalty_weights(1.0),
        window_days=14,
    )
    assert weak == 0.0


def test_from_config_carries_penalty():
    from argos.config import EventDetectionConfig

    config = EventDetectionConfig(same_source_penalty=0.07)
    assert EdgeWeights.from_config(config).same_source_penalty == 0.07
    assert EventDetectionConfig().same_source_penalty == 0.0
