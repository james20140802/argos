"""recluster_core — 간선 채택 + Leiden 커뮤니티 판정 (ARG-279).

DB를 전혀 쓰지 않는다. 문서 피처를 손으로 만들어 넣고 커뮤니티 구성만 본다.
"""
from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timezone

import pytest

from argos.brain.event_scoring import DocumentFeatures, cosine_similarity
from argos.brain.recluster_core import (
    Community,
    WeightedEdge,
    build_edges,
    detect_communities,
    merge_cross_source,
)
from argos.brain.recluster_input import NeighborPair, ReclusterDocument
from argos.config import EventDetectionConfig
from tests.conftest import requires_graph_libs

_AT = datetime(2026, 8, 10, tzinfo=timezone.utc)

# 기본 가중치는 cosine .55 / entity .25 / time .15 / keyword .05, 합 1.0이고
# join_threshold는 0.55다. 모든 문서의 `at`을 같게 두면 time 항이 항상 1.0(=0.15)
# 이므로, 임베딩을 빼면 나머지를 다 채워도 0.45라 **임계값을 넘을 수가 없다.**
# 그래서 여기서는 각도로 코사인을 직접 통제한다: 단위벡터 (cos θ, sin θ) 둘의
# 코사인 유사도는 cos(θ₁-θ₂)다.
_THETA_APART = 0.6
"""이만큼 벌리면 코사인 ≈ 0.825 → 이름·키워드가 하나도 안 겹쳐도
0.55×0.825 + 0.15 ≈ 0.60으로 간선은 되지만, 두 칸 벌어진 짝(cos 1.2 ≈ 0.36 →
≈ 0.35)은 간선이 못 된다. 사슬 저항을 볼 때 쓰는 배치다."""


def _doc(
    index: int,
    *,
    theta: float = 0.0,
    names: set[str] | None = None,
    keywords: set[str] | None = None,
) -> ReclusterDocument:
    """id가 index 순서대로 커지는 문서. `theta`가 임베딩 방향을 정한다."""
    return ReclusterDocument(
        tech_item_id=uuid.UUID(int=index),
        features=DocumentFeatures(
            embedding=(math.cos(theta), math.sin(theta)),
            names=frozenset(names or ()),
            at=_AT,
            keywords=frozenset(keywords or ()),
        ),
        event_ids=(),
    )


def _pair(left: ReclusterDocument, right: ReclusterDocument) -> NeighborPair:
    low, high = sorted([left.tech_item_id, right.tech_item_id])
    return NeighborPair(left_id=low, right_id=high)


def _members(communities) -> set[frozenset[uuid.UUID]]:
    return {frozenset(community.members) for community in communities}


def _community_of(communities, doc: ReclusterDocument) -> frozenset[uuid.UUID]:
    for community in communities:
        if doc.tech_item_id in community.members:
            return frozenset(community.members)
    raise AssertionError("문서가 어떤 커뮤니티에도 없다")


def test_pairs_below_the_join_threshold_do_not_become_edges():
    # AC: 낮 배정 기준 미만이면 간선이 아니다 — 밤 전용 기준이 없다.
    # 직교 임베딩(90°) + 겹치는 이름·키워드 없음 → time 항 0.15만 남는다.
    config = EventDetectionConfig()
    left = _doc(1, theta=0.0, names={"alpha"}, keywords={"alpha"})
    right = _doc(2, theta=math.pi / 2, names={"omega"}, keywords={"omega"})
    edges = build_edges([left, right], [_pair(left, right)], config=config)
    assert edges == ()


def test_pairs_at_or_above_the_join_threshold_become_edges():
    config = EventDetectionConfig()
    left = _doc(1, theta=0.0, names={"a", "b"}, keywords={"a", "b"})
    right = _doc(2, theta=0.0, names={"a", "b"}, keywords={"a", "b"})  # 완전 일치
    edges = build_edges([left, right], [_pair(left, right)], config=config)
    assert len(edges) == 1
    assert edges[0].weight >= config.join_threshold


def test_lowering_the_threshold_in_config_admits_more_edges():
    # AC: config를 바꾸면 낮과 밤이 같이 바뀐다 — 코어가 config를 실제로 읽는다.
    left = _doc(1, theta=0.0, names={"a", "b"}, keywords={"a"})
    right = _doc(2, theta=math.pi / 2, names={"b", "c"}, keywords={"c"})
    pairs = [_pair(left, right)]
    strict = build_edges([left, right], pairs, config=EventDetectionConfig())
    loose = build_edges(
        [left, right], pairs, config=EventDetectionConfig(join_threshold=0.0)
    )
    assert strict == ()
    assert len(loose) == 1


@requires_graph_libs
def test_lowering_only_the_join_threshold_actually_changes_the_clustering():
    # AC: config에서 임계값을 바꾸면 낮 배정과 밤 재군집이 **같이** 바뀐다.
    # 예전에는 leiden_resolution이 0.55를 따로 복사해 갖고 있어서, "더 잘
    # 묶이라고" join_threshold만 내리면 채택된 간선(≈0.447)이 전부 γ=0.55보다
    # 낮아 묶이는 이득이 없었다 — 간선은 늘었는데 커뮤니티는 그대로 흩어진 채.
    a = _doc(1, theta=0.0)
    b = _doc(2, theta=1.0)  # 코사인 ≈0.540 → 가중치 ≈0.447
    pairs = [_pair(a, b)]
    apart = {frozenset({a.tech_item_id}), frozenset({b.tech_item_id})}

    strict = EventDetectionConfig()
    assert build_edges([a, b], pairs, config=strict) == ()
    assert _members(detect_communities([a, b], pairs, config=strict)) == apart

    loose = EventDetectionConfig(join_threshold=0.35)
    assert loose.effective_leiden_resolution == pytest.approx(0.35)  # 따라온다
    assert len(build_edges([a, b], pairs, config=loose)) == 1
    assert _members(detect_communities([a, b], pairs, config=loose)) == {
        frozenset({a.tech_item_id, b.tech_item_id})
    }

    # 명시 오버라이드로 끊으면 옛 동작이 그대로 재현된다 — "따라간다"가 우연이
    # 아니라 이 기본값의 효과임을 반대쪽에서 확인한다.
    pinned = EventDetectionConfig(join_threshold=0.35, leiden_resolution=0.55)
    assert len(build_edges([a, b], pairs, config=pinned)) == 1
    assert _members(detect_communities([a, b], pairs, config=pinned)) == apart


@requires_graph_libs
def test_a_weak_chain_does_not_collapse_into_one_blob():
    # AC: A–B–C처럼 약한 연결로만 이어진 사슬은 한 덩어리가 되지 않는다.
    # A–B와 B–C는 겨우 간선이 되는 세기(≈0.60)이고, A–C는 아예 간선이 못 된다.
    config = EventDetectionConfig()
    a = _doc(1, theta=0.0)
    b = _doc(2, theta=_THETA_APART)
    c = _doc(3, theta=2 * _THETA_APART)

    # 전제부터 확인한다 — 이 배치가 실제로 "약한 사슬"인지.
    chain_edges = build_edges([a, b, c], [_pair(a, b), _pair(b, c)], config=config)
    assert len(chain_edges) == 2
    assert build_edges([a, c], [_pair(a, c)], config=config) == ()

    communities = detect_communities(
        [a, b, c], [_pair(a, b), _pair(b, c), _pair(a, c)], config=config
    )
    assert _community_of(communities, a) != _community_of(communities, c)


@requires_graph_libs
def test_a_tightly_connected_group_stays_together():
    config = EventDetectionConfig()
    docs = [
        _doc(i, theta=0.0, names={"x", "y", "z"}, keywords={"x", "y"})
        for i in (1, 2, 3)
    ]
    pairs = [_pair(docs[0], docs[1]), _pair(docs[1], docs[2]), _pair(docs[0], docs[2])]
    communities = detect_communities(docs, pairs, config=config)
    assert _members(communities) == {frozenset(d.tech_item_id for d in docs)}


@requires_graph_libs
def test_every_community_is_internally_connected():
    # AC: 커뮤니티 안의 문서만 남기고 간선을 봐도 끊긴 조각으로 갈라지지 않는다.
    config = EventDetectionConfig()
    left = [
        _doc(i, theta=0.0, names={"L", "M", "N"}, keywords={"L"}) for i in (1, 2, 3)
    ]
    right = [
        _doc(i, theta=0.0, names={"P", "Q", "R"}, keywords={"P"}) for i in (4, 5, 6)
    ]
    docs = left + right
    pairs = [
        _pair(left[0], left[1]),
        _pair(left[1], left[2]),
        _pair(left[0], left[2]),
        _pair(right[0], right[1]),
        _pair(right[1], right[2]),
        _pair(right[0], right[2]),
    ]
    communities = detect_communities(docs, pairs, config=config)
    edges = build_edges(docs, pairs, config=config)
    adjacency: dict[uuid.UUID, set[uuid.UUID]] = {d.tech_item_id: set() for d in docs}
    for edge in edges:
        adjacency[edge.left_id].add(edge.right_id)
        adjacency[edge.right_id].add(edge.left_id)

    for community in communities:
        members = set(community.members)
        # 커뮤니티 안에서만 BFS — 한 번에 전부 닿아야 한다.
        start = next(iter(members))
        seen = {start}
        queue = [start]
        while queue:
            node = queue.pop()
            for neighbor in adjacency[node] & members:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        assert seen == members


@requires_graph_libs
def test_shuffling_the_input_order_does_not_change_the_partition():
    # AC: 입력 순서만 뒤섞어도 커뮤니티 구성이 완전히 같다.
    config = EventDetectionConfig()
    left = [
        _doc(i, theta=0.0, names={"L", "M", "N"}, keywords={"L"}) for i in (1, 2, 3)
    ]
    right = [
        _doc(i, theta=0.0, names={"P", "Q", "R"}, keywords={"P"}) for i in (4, 5, 6)
    ]
    docs = left + right
    pairs = [
        _pair(left[0], left[1]),
        _pair(left[1], left[2]),
        _pair(right[0], right[1]),
        _pair(right[1], right[2]),
    ]
    baseline = _members(detect_communities(docs, pairs, config=config))

    rng = random.Random(1234)
    for _ in range(5):
        shuffled_docs = docs[:]
        shuffled_pairs = pairs[:]
        rng.shuffle(shuffled_docs)
        rng.shuffle(shuffled_pairs)
        assert (
            _members(detect_communities(shuffled_docs, shuffled_pairs, config=config))
            == baseline
        )


@requires_graph_libs
def test_documents_with_no_edges_come_back_as_singletons():
    config = EventDetectionConfig()
    lonely = _doc(1, theta=0.0, names={"only"}, keywords={"only"})
    communities = detect_communities([lonely], [], config=config)
    assert _members(communities) == {frozenset({lonely.tech_item_id})}


@requires_graph_libs
def test_communities_and_members_are_sorted_for_stable_output():
    config = EventDetectionConfig()
    docs = [_doc(i, theta=0.0, names={"x", "y", "z"}, keywords={"x"}) for i in (3, 1, 2)]
    pairs = [_pair(docs[0], docs[1]), _pair(docs[1], docs[2])]
    communities = detect_communities(docs, pairs, config=config)
    for community in communities:
        assert list(community.members) == sorted(community.members)
    assert [c.members[0] for c in communities] == sorted(c.members[0] for c in communities)


def test_detect_raises_a_readable_error_without_the_libraries(monkeypatch):
    # AC(T1과 짝): 미설치면 스택트레이스가 아니라 설치 방법을 담은 메시지.
    from argos.brain import graph_backend

    monkeypatch.setattr(graph_backend, "load_graph_libs", lambda: None)
    with pytest.raises(graph_backend.GraphLibsUnavailable) as excinfo:
        detect_communities(
            [_doc(1, theta=0.0, names={"a"}, keywords={"a"})],
            [],
            config=EventDetectionConfig(),
        )
    assert "uv sync --all-extras" in str(excinfo.value)


@requires_graph_libs
def test_identical_documents_stay_together_at_the_strictest_threshold():
    # config가 허용하는 가장 엄격한 설정(join_threshold=1.0)에서는 가중치가
    # 정확히 1.0인 간선만 살아남는다. 그런데 γ까지 1.0이면 그 간선의 CPM
    # 이득이 0이라 Leiden이 붙이지 않는다 — **완전히 동일한 문서**가 싱글턴으로
    # 흩어져 거짓 "가를 후보"가 된다. 가중치 1.0은 이론값이 아니다: 네 항이
    # 모두 만점이면 나오고, 같은 기사를 두 소스에서 받으면 실제로 그렇다.
    same = dict(names={"Sonnet 5"}, keywords={"release", "anthropic"})
    docs = [_doc(1, **same), _doc(2, **same), _doc(3, **same)]
    pairs = [
        _pair(left, right)
        for i, left in enumerate(docs)
        for right in docs[i + 1 :]
    ]

    config = EventDetectionConfig(join_threshold=1.0)
    edges = build_edges(docs, pairs, config=config)
    # 전제부터 고정한다: 세 쌍 다 간선이고 가중치는 정확히 1.0이다.
    assert len(edges) == 3
    assert all(edge.weight == pytest.approx(1.0) for edge in edges)

    assert _members(detect_communities(docs, pairs, config=config)) == {
        frozenset(doc.tech_item_id for doc in docs)
    }


# --- ARG-283: 큰 사건이 크기 때문에 갈라지지 않는다 -------------------------


def _top_k_pairs(docs: list[ReclusterDocument], k: int) -> list[NeighborPair]:
    """`recluster_input._NEIGHBOR_PAIRS_SQL`이 고르는 이웃을 그대로 흉내 낸다.

    문서마다 코사인 거리 오름차순, 동점이면 id 오름차순으로 상위 k를 뽑고
    방향을 접는다. 동일 문서는 거리가 전부 0으로 동점이라 **모두가 id 작은
    k건을 꼽는다** — 실측에서 이론 상한보다 일찍 갈라지던 허브형 그래프가
    이 타이브레이크에서 나온다. 완전 그래프를 직접 넘기면 그 모양을 놓친다.
    """
    pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for doc in docs:
        others = sorted(
            (other for other in docs if other is not doc),
            key=lambda other: (
                1.0 - cosine_similarity(doc.features.embedding, other.features.embedding),
                other.tech_item_id,
            ),
        )
        for other in others[:k]:
            low, high = sorted([doc.tech_item_id, other.tech_item_id])
            pairs.add((low, high))
    return [NeighborPair(left_id=low, right_id=high) for low, high in sorted(pairs)]


_SAME_STORY = dict(names={"Sonnet 5"}, keywords={"release", "anthropic"})


@requires_graph_libs
def test_fifty_identical_documents_stay_one_community_at_the_default_cap():
    docs = [_doc(index, **_SAME_STORY) for index in range(1, 51)]
    config = EventDetectionConfig()

    communities = detect_communities(
        docs, _top_k_pairs(docs, config.candidate_k), config=config
    )

    assert _members(communities) == {frozenset(doc.tech_item_id for doc in docs)}


@requires_graph_libs
@pytest.mark.parametrize("size", [50, 92, 150])
def test_piece_count_does_not_follow_the_document_count(size):
    # 고치기 전 기본값(K=25)에서는 50 → 5조각, 92 → 47조각, 150 → 105조각이었다.
    docs = [_doc(index, **_SAME_STORY) for index in range(1, size + 1)]
    config = EventDetectionConfig()

    communities = detect_communities(
        docs, _top_k_pairs(docs, config.candidate_k), config=config
    )

    assert len(communities) == 1


@requires_graph_libs
def test_a_cap_below_the_event_size_is_what_splits_it():
    # 위 두 테스트가 "아무것도 안 하는 테스트"가 아님을 못 박는다: 같은 입력에
    # 옛 기본값 25를 걸면 조각이 난다. 갈라짐의 원인이 내용이 아니라 상한이다.
    docs = [_doc(index, **_SAME_STORY) for index in range(1, 151)]
    config = EventDetectionConfig(candidate_k=25)

    communities = detect_communities(
        docs, _top_k_pairs(docs, config.candidate_k), config=config
    )

    assert len(communities) > 1


@requires_graph_libs
def test_a_weak_chain_still_splits_when_every_pair_is_visible():
    # 상한을 올려 모든 쌍이 보여도, 이웃끼리만 이어진 사슬은 한 덩어리가 되지
    # 않는다 — 안 이어진 쌍(코사인 ≈ 0.36 이하)은 여전히 간선이 아니고 대가만
    # 낸다. 상한을 푸는 일이 ARG-244가 막아 둔 뭉침을 되살리면 안 된다.
    docs = [_doc(index, theta=index * _THETA_APART) for index in range(1, 9)]
    config = EventDetectionConfig()

    communities = detect_communities(
        docs, _top_k_pairs(docs, config.candidate_k), config=config
    )

    assert len(communities) > 1
    assert max(len(community.members) for community in communities) <= 2


@requires_graph_libs
def test_large_identical_events_partition_the_same_way_every_time():
    docs = [_doc(index, **_SAME_STORY) for index in range(1, 151)]
    config = EventDetectionConfig()
    pairs = _top_k_pairs(docs, config.candidate_k)

    first = detect_communities(docs, pairs, config=config)
    second = detect_communities(list(reversed(docs)), list(reversed(pairs)), config=config)

    assert first == second


# --- ARG-306: CPM 조각을 출처가 다를 때만 평균 연결로 합치기 -----------------


def _id(index: int) -> uuid.UUID:
    return uuid.UUID(int=index)


def _community(*indexes: int) -> Community:
    return Community(members=tuple(sorted(_id(i) for i in indexes)))


def _scored(weights: dict[tuple[int, int], float]) -> list[WeightedEdge]:
    return sorted(
        (
            WeightedEdge(left_id=_id(min(a, b)), right_id=_id(max(a, b)), weight=w)
            for (a, b), w in weights.items()
        ),
        key=lambda edge: (edge.left_id, edge.right_id),
    )


# 조각 {1,2}와 {3,4,5}. 조각 사이 여섯 짝 중 넷은 τ를 겨우 넘고 둘은 조금 못 미친다
# — CPM(γ=τ)이 가르는 바로 그 모양이다. 평균 = (0.60×4 + 0.52×2) / 6 ≈ 0.573.
_KIMI_LIKE = {
    (1, 2): 0.68,
    (3, 4): 0.60,
    (3, 5): 0.60,
    (4, 5): 0.60,
    (1, 3): 0.60,
    (1, 4): 0.60,
    (2, 3): 0.60,
    (2, 5): 0.60,
    (1, 5): 0.52,
    (2, 4): 0.52,
}


def test_pieces_from_different_sites_merge_when_their_average_clears_the_threshold():
    sources = {_id(1): "a.com", _id(2): "a.com", _id(3): "b.com", _id(4): "c.com", _id(5): "d.com"}
    merged = merge_cross_source(
        [_community(1, 2), _community(3, 4, 5)],
        _scored(_KIMI_LIKE),
        sources=sources,
        threshold=0.55,
    )
    assert _members(merged) == {frozenset(_id(i) for i in (1, 2, 3, 4, 5))}


def test_pieces_from_a_single_site_are_not_merged():
    # 같은 사이트의 다른 발표가 한 카드로 뭉치는 게 가장 흔한 오병합이었다.
    sources = {_id(i): "openai.com" for i in range(1, 6)}
    merged = merge_cross_source(
        [_community(1, 2), _community(3, 4, 5)],
        _scored(_KIMI_LIKE),
        sources=sources,
        threshold=0.55,
    )
    assert _members(merged) == {
        frozenset({_id(1), _id(2)}),
        frozenset({_id(3), _id(4), _id(5)}),
    }


def test_unknown_sources_do_not_count_as_a_second_site():
    sources = {_id(1): "a.com", _id(2): "a.com", _id(3): None, _id(4): None, _id(5): None}
    merged = merge_cross_source(
        [_community(1, 2), _community(3, 4, 5)],
        _scored(_KIMI_LIKE),
        sources=sources,
        threshold=0.55,
    )
    assert len(merged) == 2


def test_pairs_that_are_not_neighbors_count_as_zero_in_the_average():
    # 이웃으로 보인 짝 둘이 아무리 세도, 안 보인 짝 넷이 0으로 평균을 끌어내린다:
    # (0.9 + 0.9) / 6 = 0.30 < 0.55 → 합치지 않는다. 약한 사슬이 다시 뭉치지 않는 이유.
    sources = {_id(1): "a.com", _id(2): "a.com", _id(3): "b.com", _id(4): "b.com", _id(5): "b.com"}
    weights = {(1, 2): 0.9, (3, 4): 0.9, (4, 5): 0.9, (3, 5): 0.9, (1, 3): 0.9, (2, 3): 0.9}
    merged = merge_cross_source(
        [_community(1, 2), _community(3, 4, 5)],
        _scored(weights),
        sources=sources,
        threshold=0.55,
    )
    assert len(merged) == 2


def test_merging_happens_strongest_first_and_rechecks_the_grown_piece():
    # {1}+{2}가 0.9로 먼저 합쳐진 뒤 {3}과의 평균은 (0.6 + 0.0)/2 = 0.30이라 붙지 않는다.
    # 순서를 무시하고 짝별로 따로 판정하면 {3}이 {1}과의 0.6 하나로 끌려 들어온다.
    sources = {_id(1): "a.com", _id(2): "b.com", _id(3): "c.com"}
    merged = merge_cross_source(
        [_community(1), _community(2), _community(3)],
        _scored({(1, 2): 0.9, (1, 3): 0.6}),
        sources=sources,
        threshold=0.55,
    )
    assert _members(merged) == {frozenset({_id(1), _id(2)}), frozenset({_id(3)})}


def test_the_merge_result_does_not_depend_on_input_order():
    sources = {_id(1): "a.com", _id(2): "a.com", _id(3): "b.com", _id(4): "c.com", _id(5): "d.com"}
    communities = [_community(1, 2), _community(3, 4, 5)]
    expected = merge_cross_source(
        communities, _scored(_KIMI_LIKE), sources=sources, threshold=0.55
    )
    for seed in range(5):
        shuffled = list(communities)
        random.Random(seed).shuffle(shuffled)
        got = merge_cross_source(
            sorted(shuffled, key=lambda c: c.members),
            _scored(_KIMI_LIKE),
            sources=sources,
            threshold=0.55,
        )
        assert _members(got) == _members(expected)


@requires_graph_libs
def test_turning_the_merge_off_restores_the_cpm_only_partition(monkeypatch):
    # 끄면 Leiden 결과 그대로 — 합치기 단계가 호출조차 되지 않는다.
    import argos.brain.recluster_core as core

    def boom(*args, **kwargs):  # pragma: no cover - 호출되면 실패
        raise AssertionError("cross_source_merge=False인데 합치기가 돌았다")

    monkeypatch.setattr(core, "merge_cross_source", boom)
    config = EventDetectionConfig(cross_source_merge=False)
    docs = [_doc(i, theta=0.0, names={"x"}, keywords={"x"}) for i in (1, 2, 3)]
    pairs = [_pair(docs[0], docs[1]), _pair(docs[1], docs[2]), _pair(docs[0], docs[2])]
    communities = detect_communities(docs, pairs, config=config)
    assert _members(communities) == {frozenset(d.tech_item_id for d in docs)}
