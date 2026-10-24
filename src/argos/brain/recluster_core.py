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

**CPM 뒤에 출처가 다른 조각끼리 평균 연결로 한 번 더 합친다 (ARG-306).**
γ = τ인 CPM은 짝 하나하나가 τ를 넘는, 거의 완전 그래프인 묶음만 남긴다.
τ를 겨우 넘는 간선(+0.01~0.05)은 이득이 거의 없는데, 간선이 없는 짝은 −γ를
통째로 물기 때문이다. 같은 출시를 여러 사이트가 각자 다른 결로 쓴 사건(공식
발표·벤치마크 분석·개인 감상)은 짝 점수가 고르게 높지 않아 바로 그 자리에서
갈라졌다 — Kimi K3 출시 주간 다섯 글이 2+3으로 쪼개졌다. 그래서 Leiden이 낸
조각 둘의 **모든 짝 평균**(이웃이 아닌 짝은 0)이 τ 이상이면 합친다. 평균이 τ
이상이면 그중 가장 센 짝은 반드시 τ 이상이므로, 합친 커뮤니티도 τ 이상 간선으로
이어져 있다는 보장은 그대로다.

합치기는 **출처(등록 도메인)가 둘 이상인 경우에만** 한다. 실측(2026-10-22,
1,902건)에서 출처 조건 없이 평균 연결로 합치면 판정상 틀린 묶음이 20 → 21로
늘고 판정 밖에서도 새 짝이 130개 생겼는데, 그 대부분이 같은 사이트의 서로
다른 발표(OpenAI 고객 사례·제휴 발표)였다. 출처 조건을 걸면 틀린 묶음은 20
그대로, 새 짝은 13개였다. 출처를 모르는 문서는 "다른 출처"의 근거로 세지
않는다 — 모르는 것을 다르다고 치면 근거 없이 합치는 셈이다. 간선 채택 컷이
τ인 것처럼 합치기 기준도 τ다. 밤 전용 기준을 두지 않는 이유는 위와 같다.

**결정성은 두 겹이다.** (1) 노드를 id 오름차순으로 삽입해 그래프의 정점 번호를
입력 순서와 무관하게 만들고, (2) leidenalg에 시드를 고정해 준다. 둘 중 하나만
해서는 부족하다 — 시드만 고정하면 정점 번호가 입력 순서를 타고, 정렬만 하면
난수가 실행마다 다르다. 출력도 멤버와 커뮤니티를 정렬해 돌려준다.
"""

from __future__ import annotations

import heapq
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, AbstractSet, Sequence

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
    return tuple(
        edge
        for edge in _scored_pairs(documents, pairs, config=config)
        if edge.weight >= config.join_threshold
    )


def _scored_pairs(
    documents: Sequence["ReclusterDocument"],
    pairs: Sequence["NeighborPair"],
    *,
    config: "EventDetectionConfig",
) -> tuple[WeightedEdge, ...]:
    """이웃 쌍 전부의 점수 — τ 미만도 버리지 않는다. 정렬·중복 제거는 `build_edges`와 같다."""
    weights = EdgeWeights.from_config(config)
    features_by_id = {doc.tech_item_id: doc.features for doc in documents}

    scored: list[WeightedEdge] = []
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
        scored.append(WeightedEdge(left_id=left_id, right_id=right_id, weight=weight))

    return tuple(sorted(scored, key=lambda edge: (edge.left_id, edge.right_id)))


def score_pairs(
    documents: Sequence["ReclusterDocument"],
    pairs: Sequence["NeighborPair"],
    *,
    config: "EventDetectionConfig",
) -> tuple[WeightedEdge, ...]:
    """이웃 쌍 전부의 점수(τ 미만 포함). 애매한 짝을 고르는 쪽(`pair_check`)이 쓴다."""
    return _scored_pairs(documents, pairs, config=config)


def exclude_vetoed(
    scored: Sequence[WeightedEdge],
    vetoed_pairs: AbstractSet[tuple[uuid.UUID, uuid.UUID]],
) -> tuple[WeightedEdge, ...]:
    """거부된 짝을 점수 목록에서 뺀다. 짝의 id 순서는 상관없다(ARG-307).

    빠진 짝은 Leiden 간선도, 출처 간 합치기 평균의 기여도 아니다(평균에서는 이웃이
    아닌 짝처럼 0으로 친다).
    """
    if not vetoed_pairs:
        return tuple(scored)
    keys = {tuple(sorted(pair)) for pair in vetoed_pairs}
    return tuple(
        edge for edge in scored if (edge.left_id, edge.right_id) not in keys
    )


def detect_communities(
    documents: Sequence["ReclusterDocument"],
    pairs: Sequence["NeighborPair"],
    *,
    config: "EventDetectionConfig",
    vetoed_pairs: AbstractSet[tuple[uuid.UUID, uuid.UUID]] = frozenset(),
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

    **거부된 짝 (ARG-307).** `vetoed_pairs`는 로컬 LLM이 "다른 발표"라 확인한 짝이다.
    점수 함수는 그대로 두고 그 짝만 간선(과 합치기 평균)에서 뺀다 — 낮·밤이 같은
    점수를 쓴다는 원칙(ARG-282)을 건드리지 않는다. 빈 집합이면 예전과 같다.
    둘이 다른 문서를 거쳐 이어지면 그 경로로 여전히 같은 덩어리가 될 수 있다.

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

    # 점수는 한 번만 매긴다 — τ 이상은 Leiden 간선, 전부는 출처 간 합치기 평균에.
    scored = exclude_vetoed(
        _scored_pairs(documents, pairs, config=config), vetoed_pairs
    )
    edges = tuple(edge for edge in scored if edge.weight >= config.join_threshold)

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
    if config.cross_source_merge:
        communities = merge_cross_source(
            sorted(communities, key=lambda community: community.members),
            scored,
            sources={doc.tech_item_id: doc.features.source for doc in documents},
            threshold=config.join_threshold,
        )
    # 커뮤니티 자체도 정렬해 돌려준다 — 소비자(후보 도출·CLI 출력)가 순서에
    # 기대도 안전하게.
    return tuple(sorted(communities, key=lambda community: community.members))


def merge_cross_source(
    communities: Sequence[Community],
    scored_pairs: Sequence[WeightedEdge],
    *,
    sources: dict[uuid.UUID, str | None],
    threshold: float,
) -> list[Community]:
    """출처가 둘 이상 섞이는 조각끼리만, 짝 평균이 `threshold` 이상인 동안 합친다.

    평균 연결(average linkage) 응집 군집이다. 두 조각 A·B의 연결 세기는
    A×B 모든 짝 점수의 합 ÷ (|A|·|B|) — `scored_pairs`에 없는 짝(이웃이 아님)은
    0으로 친다. 가장 센 짝부터 합치고, 합친 뒤에는 그 조각과 나머지의 세기를 다시
    잰다. 한 출처뿐인 합치기는 건너뛴다(모듈 docstring, ARG-306).

    입력 커뮤니티 순서와 `scored_pairs` 순서가 정렬돼 있으면 결과도 결정적이다 —
    동점은 조각 번호(입력 순서)로 깨고, 점수 합도 정렬된 순서로 더한다.
    """
    cluster_of: dict[uuid.UUID, int] = {}
    members: dict[int, list[uuid.UUID]] = {}
    for index, community in enumerate(communities):
        members[index] = list(community.members)
        for member in community.members:
            cluster_of[member] = index

    link: dict[int, dict[int, float]] = defaultdict(dict)
    for edge in scored_pairs:
        left = cluster_of.get(edge.left_id)
        right = cluster_of.get(edge.right_id)
        if left is None or right is None or left == right or edge.weight <= 0.0:
            continue
        link[left][right] = link[left].get(right, 0.0) + edge.weight
        link[right][left] = link[right].get(left, 0.0) + edge.weight

    def known_sources(cluster: int) -> frozenset[str]:
        return frozenset(
            source for member in members[cluster] if (source := sources.get(member))
        )

    source_sets = {cluster: known_sources(cluster) for cluster in members}
    version = {cluster: 0 for cluster in members}
    heap: list[tuple[float, int, int, int, int]] = []

    def push(a: int, b: int) -> None:
        low, high = min(a, b), max(a, b)
        average = link[low][high] / (len(members[low]) * len(members[high]))
        if average >= threshold:
            heapq.heappush(heap, (-average, low, high, version[low], version[high]))

    for left in sorted(link):
        for right in sorted(link[left]):
            if left < right:
                push(left, right)

    next_id = len(members)
    while heap:
        _, left, right, left_version, right_version = heapq.heappop(heap)
        if version.get(left) != left_version or version.get(right) != right_version:
            continue  # 이미 다른 조각에 합쳐졌거나 세기가 바뀐 낡은 항목
        if len(source_sets[left] | source_sets[right]) < 2:
            continue
        merged = next_id
        next_id += 1
        members[merged] = members.pop(left) + members.pop(right)
        source_sets[merged] = source_sets.pop(left) | source_sets.pop(right)
        del version[left], version[right]
        version[merged] = 0
        neighbors: dict[int, float] = {}
        for old in (left, right):
            for other, total in link.pop(old).items():
                if other in (left, right):
                    continue
                neighbors[other] = neighbors.get(other, 0.0) + total
                del link[other][old]
        link[merged] = neighbors
        for other in sorted(neighbors):
            link[other][merged] = neighbors[other]
            push(other, merged)

    return [Community(members=tuple(sorted(group))) for group in members.values()]
