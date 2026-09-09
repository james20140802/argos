"""recluster_core — 간선 채택 + Leiden 커뮤니티 판정 (ARG-279).

DB를 전혀 쓰지 않는다. 문서 피처를 손으로 만들어 넣고 커뮤니티 구성만 본다.
"""
from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timezone

import pytest

from argos.brain.event_scoring import DocumentFeatures
from argos.brain.recluster_core import build_edges, detect_communities
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
