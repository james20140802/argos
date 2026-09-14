"""간선 구성 + Leiden 커뮤니티 판정 — ARG-279. DB도 LLM도 쓰지 않는다.

`event_assignment.decide_event`가 온라인 배정의 순수 판정 코어인 것처럼, 이
모듈은 야간 재군집의 순수 판정 코어다. 입력은 문서 피처와 이웃 쌍뿐이고,
출력은 커뮤니티 목록뿐이다. 그래서 Postgres 없이 "약한 사슬이 뭉치지
않는다", "커뮤니티가 내부적으로 이어져 있다", "같은 입력에 같은 결과"를 전부
테스트로 확인할 수 있다.

**간선 채택 컷은 `join_threshold`다** — 낮 배정이 쓰는 바로 그 값. 야간 전용
임계값을 만들면 낮이 붙인 것을 밤이 떼고 다음 낮이 다시 붙이는 진동이 생긴다.
가중치도 `event_scoring.edge_weight`를 그대로 부른다. 이 모듈에는 점수 산식이
한 줄도 없다.

**다만 재사용되는 것은 값이지 기준이 아니다.** 두 경로는 같은 숫자를 서로 다른
양과 견준다. 낮의 `event_scoring.choose_event`는 후보 이웃(최대 `candidate_k`개)
가중치를 사건별로 **합산**해 `best_total >= join_threshold`를 보고, 그 경로에는
간선 하나에 걸리는 최소 세기가 아예 없다. 밤의 `build_edges`는 **한 쌍의 가중치
하나**를 같은 값과 견준다. 그래서 밤 기준이 구조적으로 더 엄격하다 — 실측:
한 쌍당 0.3203짜리 이웃 셋(합 0.9609)은 낮에는 사건에 붙지만 밤에는 간선이 하나도
그려지지 않고, 임베딩이 없고 시각이 같은 문서 넷(쌍당 0.15, 합 0.60)도 마찬가지다.
임계값 재보정은 사용자 판단이 필요한 **열린 문제**라 이 브랜치에서 손대지 않았다.
후보를 실제로 반영하는 ARG-245가 그 차이를 가장 먼저 체감할 소비자다.

**CPM 해상도 γ는 `join_threshold`를 따라간다.** `leiden_resolution`을 비워
두면(기본) `EventDetectionConfig.effective_leiden_resolution`이 `join_threshold`를
돌려준다. 둘을 값만 같은 독립 필드로 두면 "임계값만 낮췄는데 오히려 더 잘게
쪼개진다"가 조용히 벌어진다 — γ가 채택된 간선의 세기보다 높으면 뭉칠 이득이
없기 때문이다(CPM 품질 = 내부 가중치합 - γ × 쌍의 수). 조율이 필요하면
`leiden_resolution`을 명시해 끊을 수 있고, 그때만 둘이 따로 논다.
`leiden_objective="modularity"`에서는 γ 자체가 쓰이지 않는다.

**왜 Leiden인가.** 연결 요소는 체이닝에 무너지고(A~B, B~C면 A·C가 무관해도 한
덩어리), Louvain은 내부적으로 연결되지도 않은 노드를 한 커뮤니티에 넣을 수
있다. Leiden은 Louvain 저자들이 바로 그 결함을 고쳐 만든 후속이라 커뮤니티
연결성을 보장한다 — 화면에서 "무관한 기사 둘이 같은 카드에"가 나오지 않는
근거가 이 보장이다.

**결정성은 두 겹이다.** (1) 노드를 id 오름차순으로 삽입해 그래프의 정점 번호를
입력 순서와 무관하게 만들고, (2) leidenalg에 시드를 고정해 준다. 둘 중 하나만
해서는 부족하다 — 시드만 고정하면 정점 번호가 입력 순서를 타고, 정렬만 하면
난수가 실행마다 다르다. 출력도 멤버와 커뮤니티를 정렬해 돌려준다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from argos.brain.event_scoring import EdgeWeights, edge_weight
from argos.brain.graph_backend import require_graph_libs

if TYPE_CHECKING:
    from argos.brain.recluster_input import NeighborPair, ReclusterDocument
    from argos.config import EventDetectionConfig


@dataclass(frozen=True)
class WeightedEdge:
    """채택된 간선. `left_id < right_id`로 정규화돼 있다."""

    left_id: uuid.UUID
    right_id: uuid.UUID
    weight: float


@dataclass(frozen=True)
class Community:
    """한 덩어리로 판정된 문서들. `members`는 id 오름차순."""

    members: tuple[uuid.UUID, ...]


def build_edges(
    documents: Sequence["ReclusterDocument"],
    pairs: Sequence["NeighborPair"],
    *,
    config: "EventDetectionConfig",
) -> tuple[WeightedEdge, ...]:
    """`join_threshold` 이상인 쌍만 간선으로 만든다.

    비교 대상은 **쌍 하나의 가중치**다 — 낮 배정이 같은 값을 이웃 표의 *합*과
    견주는 것과 다르다(모듈 docstring의 비대칭 항목).

    반환 순서는 `(left_id, right_id)` 오름차순 — 같은 입력이면 항상 같다.
    양끝 중 하나라도 `documents`에 없는 쌍은 버린다(그래프의 노드는 넘겨받은
    문서 집합이 전부다).
    """
    weights = EdgeWeights.from_config(config)
    features_by_id = {doc.tech_item_id: doc.features for doc in documents}

    edges: list[WeightedEdge] = []
    seen: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for pair in pairs:
        left_id, right_id = sorted([pair.left_id, pair.right_id])
        if left_id == right_id or (left_id, right_id) in seen:
            continue
        left = features_by_id.get(left_id)
        right = features_by_id.get(right_id)
        if left is None or right is None:
            continue
        seen.add((left_id, right_id))
        weight = edge_weight(
            left, right, weights=weights, window_days=config.window_days
        )
        if weight < config.join_threshold:
            continue
        edges.append(WeightedEdge(left_id=left_id, right_id=right_id, weight=weight))

    return tuple(sorted(edges, key=lambda edge: (edge.left_id, edge.right_id)))


def detect_communities(
    documents: Sequence["ReclusterDocument"],
    pairs: Sequence["NeighborPair"],
    *,
    config: "EventDetectionConfig",
) -> tuple[Community, ...]:
    """문서들을 Leiden으로 커뮤니티로 나눈다.

    간선이 하나도 없는 문서는 자기 혼자인 커뮤니티로 나온다 — 재군집 대상에서
    조용히 사라지면 "가를 후보" 판정에서 그 문서의 표가 통째로 빠진다.

    **알려진 한계 — 큰 사건은 K 상한 때문에 갈라진다 (미해결, 사람 판단 대기).**
    CPM은 간선이 없는 쌍을 가중치 0으로 보면서도 해상도 페널티는 그 쌍에도
    매긴다. 그런데 이 그래프는 문서당 이웃이 `candidate_k`개로 잘려 있다. 내부
    가중치 합의 상한은 `n·K`인데 페널티는 `γ·n(n-1)/2`로 자라므로, n이 커지면
    모든 간선이 만점(1.0)이어도 한 덩어리로 남을 수 없다. 기본값
    (`candidate_k=25`, γ=0.55)의 상한은 n ≈ 92지만, 실제 타이브레이크가 만드는
    허브형 KNN에서는 **n ≈ 50에서 이미 갈라진다**(실측 2026-09-11: 동일 문서
    50건 → 5조각, 150건 → 105조각). 즉 문서가 아주 많은 진짜 사건은 유사도가
    아무리 강해도 "가를 후보"로 올라온다.

    고치려면 완전 쌍 가중치(제곱 폭발)·다른 목적함수(modularity의 resolution
    limit)·희소화에 맞춘 γ 보정(ARG-279 AC가 깨진다) 중 하나를 골라야 하는데,
    셋 다 이 함수 위쪽에서 확정된 결정을 뒤집는다. 읽기 전용 조언 명령이라
    지금 당장 깨지는 건 없지만, ARG-245가 실제로 반영하기 전에 정해야 한다.

    Raises:
        GraphLibsUnavailable: python-igraph/leidenalg 미설치.
    """
    igraph, leidenalg = require_graph_libs()

    # (1) 노드 순서를 id로 고정한다. 입력이 어떤 순서로 들어와도 정점 번호가
    #     같아야 파티션이 같다.
    node_ids = sorted({doc.tech_item_id for doc in documents})
    if not node_ids:
        return ()
    index_of = {node_id: index for index, node_id in enumerate(node_ids)}

    edges = build_edges(documents, pairs, config=config)

    graph = igraph.Graph(n=len(node_ids), directed=False)
    if edges:
        graph.add_edges(
            [(index_of[edge.left_id], index_of[edge.right_id]) for edge in edges]
        )
        graph.es["weight"] = [edge.weight for edge in edges]

    partition_type = (
        leidenalg.CPMVertexPartition
        if config.leiden_objective == "cpm"
        else leidenalg.ModularityVertexPartition
    )
    kwargs = {
        "weights": "weight" if edges else None,
        "seed": config.leiden_seed,  # (2) 난수 고정
    }
    if config.leiden_objective == "cpm":
        # 실효 γ는 config 한 곳에서 해석한다 — 비워 두면 join_threshold를
        # 따라가고, CLI 리포트도 같은 프로퍼티를 찍는다.
        kwargs["resolution_parameter"] = config.effective_leiden_resolution

    partition = leidenalg.find_partition(graph, partition_type, **kwargs)

    communities = [
        Community(members=tuple(sorted(node_ids[index] for index in group)))
        for group in partition
        if group
    ]
    # 커뮤니티 자체도 정렬해 돌려준다 — 소비자(후보 도출·CLI 출력)가 순서에
    # 기대도 안전하게.
    return tuple(sorted(communities, key=lambda community: community.members))
