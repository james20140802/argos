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

**낮도 같은 목적함수를 쓴다 (ARG-282).** 낮의 `event_scoring.choose_event`는
문서 하나를 사건에 넣었을 때의 CPM 이득(τ 이상 간선의 합 − γ × 사건 크기)이
양수일 때만 붙인다. 예전에는 같은 τ를 낮은 이웃 점수의 *합*과, 밤은 쌍 *하나*와
견줘 밤이 훨씬 엄격했다(0.32짜리 이웃 셋이 낮에는 붙고 밤에는 간선이 없었다).
남는 차이는 낮이 문서를 도착 순서대로 하나씩 탐욕적으로 붙인다는 것뿐이고,
그 차이가 곧 밤이 교정할 몫이다.

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

    비교 대상은 **쌍 하나의 가중치**다. 낮 배정도 이 값 미만인 이웃은 세지
    않는다 — 두 경로의 그래프가 같아야 같은 품질을 잰다(ARG-282).

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

    **이웃 상한과 큰 사건 (ARG-283).** CPM은 간선이 없는 쌍을 가중치 0으로
    보면서도 해상도 페널티는 그 쌍에도 매긴다. 이 그래프는 문서당 이웃이
    `candidate_k`개로 잘려 있으므로, 내부 가중치 합의 상한은 `n·K`인데
    페널티는 `γ·n(n-1)/2`로 자란다 — K가 사건 크기보다 한참 작으면 모든 간선이
    만점이어도 한 덩어리로 남을 수 없다. 예전 기본값 25에서는 동일 문서 50건이
    5조각, 150건이 105조각이었다(동점을 id로 깨는 정렬이 모두가 같은 K건을
    꼽는 허브형 그래프를 만들어 이론 상한 n ≈ 92보다도 일찍 갈라졌다).

    그래서 K를 판정 기준이 아니라 **비용 상한**으로 두고 창 안 밀도보다 크게
    잡았다(기본 200, config 주석). 판정은 여전히 τ 이상 간선과 γ가 한다 — γ를
    크기에 맞춰 깎거나(ARG-279 AC가 깨진다) 안 본 쌍의 대가를 빼면(약한 사슬이
    다시 뭉친다) 안 되기 때문이다. 남는 한계: 같은 허브형 그래프에서 K=200은
    동일 문서 350건까지 한 덩어리로 두고 400건부터 갈라진다(실측 2026-10-05:
    400 → 37조각, 500 → 137조각). 그런 사건은 이웃이 꽉 찬 문서를 품으므로
    `ReclusterInput.capped_document_ids` → `SplitCandidate.capped`로 표시된다.

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
