"""간선 가중치와 사건 선택 판정 — ARG-264 / ARG-282. DB도 LLM도 쓰지 않는다.

두 문서가 "같은 사건"에 속할 근거를 네 항으로 잰다:

- **cosine**: 임베딩 코사인 유사도 — 의미가 얼마나 겹치는가.
- **entity**: 추출된 고유명사 집합의 자카드 — 같은 회사/모델을 말하는가.
- **time**: 발행 시각 차이에 대한 선형 감쇠 — 같은 사건이면 보통 가깝게 터진다.
- **keyword**: 키워드 집합의 자카드 — 이름 추출이 놓친 주제 겹침을 보완.

**정규화:** `weights`의 네 값이 1로 합쳐진다는 보장이 없다(사용자가 config에서
하나만 올릴 수 있다). 그래서 네 항의 가중합을 **가중치 총합으로 나눈 뒤**
`join_threshold`와 비교한다. 나누지 않으면 가중치를 올릴 때마다 최댓값이
같이 올라가 버려서 `join_threshold`가 "네 항의 가중 평균 몇 이상이면 묶는다"는
뜻을 유지하지 못하고, 가중치 하나만 세게 키운 사용자에게 조용히 다른 임계값을
적용하는 꼴이 된다.

**시간감쇠가 선형인 이유:** `max(0.0, 1.0 - Δdays / window_days)`는 창 경계
(Δdays == window_days)에서 정확히 0이 되어 "window_days 밖은 더 이상 같은
사건 후보가 아니다"라는 설정값의 의미와 모순이 없다. 지수감쇠는 점근적이라
경계에서도 잔값이 남아 "밖"이라는 말이 근사적으로만 맞게 된다.

**야간 재군집(2단계)도 `edge_weight`를 그대로 부른다.** 그래서 이 모듈은 사건이라는
개념 자체를 모른다 — `NeighborEdge`가 실어 나르는 `event_ids` 튜플과 사건 크기
이상으로 사건 전용 자료구조(DB 모델, ORM row 등)에 결합하지 않는다.

**낮의 판정은 밤의 목적함수에서 나온다 (ARG-282).** 밤은 기간 전체를 CPM으로
묶는다 — 품질 = (`join_threshold` 이상 간선의 내부 가중치 합) − γ × (쌍의 수).
문서 d를 크기 n인 사건 E에 넣을 때 그 품질이 변하는 양은
``Σ_{e∈E, w≥τ} w(d,e) − γ·n``이다. `choose_event`는 정확히 이 값을 이득으로 보고,
이득이 0보다 큰 사건 중 최댓값에 붙인다. 그래서 같은 설정값이 낮과 밤에서 **같은
것을** 재고, 값을 바꾸면 둘이 같은 방향으로 움직인다.

예전 규칙("이웃 점수를 사건별로 합해 τ 이상이면 붙인다")은 간선 하나에 걸리는
하한도, 사건 크기에 대한 대가도 없었다. 이 코퍼스에서 상위 이웃의 쌍 점수는
중앙값이 0.43이라 무관한 이웃 둘이면 합이 τ를 넘는다. 사건이 커질수록 이웃
자리를 더 차지해 표가 더 모이고, 실측(2026-10-05, 1,669건)에서 1,603건이 사건
하나로 뭉쳤다. τ를 0.65로 올려도 결과가 같았다 — 값이 아니라 규칙의 모양 문제다.

**같은 출처 보정 (ARG-284).** 같은 등록 도메인 글 쌍은 주제가 달라도 엔티티·시간·
문체가 겹쳐 원래 점수가 높다. 그래서 두 문서의 출처가 같으면 가중합 점수에서
`same_source_penalty`를 뺀다("같은 출처 쌍은 τ + penalty를 넘어야 같은 사건의
근거"로 읽혀 `join_threshold`의 뜻이 유지된다). 한쪽이라도 출처를 모르면
보정하지 않는다. 코사인이 `NEAR_DUPLICATE_COSINE` 이상이면 면제한다 — 같은 글이
두 URL로 수집된 경우는 같은 출처라서 비슷한 게 아니라 같은 글이라서 비슷한
것이고, 이 면제가 있어야 한 도메인의 내용 같은 큰 사건이 크기 때문에 갈라지지
않는다(ARG-283). 면제 기준은 노브가 아니라 상수다 — 설정은 하나다(ARG-282).
낮(`decide_event`)과 밤(`build_edges`)이 모두 `EdgeWeights.from_config`로 같은
값을 읽으므로 이 보정도 둘이 함께 움직인다.

**보정은 발행 간격에 비례한다 (ARG-299).** 같은 출처라는 사실만으로 깎으면 맞는
묶음과 틀린 묶음이 같은 비율로 깨졌다(ARG-284 스윕, 무릎 없음). 같은 회사가 같은
소식을 여러 번 쓴 글(발표 글 + 시스템 카드, 한꺼번에 낸 사례 글 묶음)은 거의 같은
시각에 나오고, 같은 블로그의 서로 다른 발표는 하루 이상 떨어져 나온다. 그래서
같은 출처 쌍의 감점을 ``penalty × min(1, 간격 / same_source_gap_days)``로 둔다 —
동시 발행이면 0, 간격이 벌어질수록 전액까지. 계단 대신 경사인 이유는 판정 128개
스윕에서 계단(간격 ≥ g면 전액)이 같은 유지 수준에서 틀린 비율을 덜 줄였고, 경계
바로 앞뒤 쌍이 정반대 대접을 받는 절벽이 없어서다.

**이름이 확실히 다르면 감점한다 (ARG-303).** 같은 회사가 같은 날 낸 다른 제품
발표는 문체·시각·출처가 같아 점수가 높지만 기사에 오른 이름은 서로 다르다.
두 글 모두 이름이 `entity_mismatch_min_names`개 이상인데 하나도 겹치지 않고
코사인이 `NEAR_DUPLICATE_COSINE` 미만이면 `entity_mismatch_penalty`를 뺀다. 이름이
빈약한 글(0개·1~2개)은 증거가 없으므로 면제한다 — 이름이 적다는 이유만으로
떨어지면 "정보가 적을수록 쉽게 갈라지는" 역전이 생긴다. 같은 출처 감점과는 독립
이라 둘 다 빼기만 하고 0에서 자른다(적용 순서와 무관). 이름 추출·정리는 이 모듈의
몫이 아니라(Task 1~2) 여기서는 이미 정리된 집합만 비교한다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from argos.config import EventDetectionConfig


@dataclass(frozen=True)
class DocumentFeatures:
    """`edge_weight`가 비교하는 문서 한 건의 피처. DB row가 아니라 순수 값이다."""

    embedding: tuple[float, ...] | None
    names: frozenset[str]
    at: datetime | None
    keywords: frozenset[str]
    # 원문 URL의 등록 도메인(`source_domain.registered_domain`). 모르면 None.
    # 기본값이 있는 마지막 필드라 기존 생성자 호출은 그대로 유효하다.
    source: str | None = None


NEAR_DUPLICATE_COSINE = 0.95
"""이 코사인 이상이면 같은 글의 재수집으로 보고 같은 출처 보정을 면제한다."""


@dataclass(frozen=True)
class EdgeWeights:
    """네 항의 가중치. 합이 1일 필요는 없다 — `edge_weight`가 정규화한다."""

    cosine: float
    entity: float
    time: float
    keyword: float
    # 같은 출처 쌍의 점수에서 빼는 값. 0.0이면 보정 없음(기존 동작).
    same_source_penalty: float = 0.0
    # 같은 출처 보정이 전액이 되는 발행 간격(일). 간격 0이면 보정 0, 이 값까지
    # 선형으로 커진다. 0.0이면 간격과 무관하게 전액(ARG-284의 일괄 감점).
    same_source_gap_days: float = 0.0
    # 두 글의 이름이 모두 `entity_mismatch_min_names`개 이상이고 서로소일 때 빼는
    # 값(ARG-303). 0.0이면 감점 없음. 같은 출처 감점과 독립이다.
    entity_mismatch_penalty: float = 0.0
    entity_mismatch_min_names: int = 3

    @classmethod
    def from_config(cls, config: "EventDetectionConfig") -> "EdgeWeights":
        return cls(
            cosine=config.weight_cosine,
            entity=config.weight_entity,
            time=config.weight_time,
            keyword=config.weight_keyword,
            same_source_penalty=config.same_source_penalty,
            same_source_gap_days=config.same_source_gap_days,
            entity_mismatch_penalty=config.entity_mismatch_penalty,
            entity_mismatch_min_names=config.entity_mismatch_min_names,
        )


@dataclass(frozen=True)
class NeighborEdge:
    """이웃 문서 하나가 표를 던지는 사건(들)과 그 표의 무게.

    `event_ids`가 튜플인 건 한 이웃이 이미 여러 사건에 걸쳐 있을 수 있어서다
    (예: 병합 전 상태, 혹은 야간 재군집 중간 산출물). 그 경우 이 이웃의
    weight는 각 사건에 그대로 더해진다 — 쪼개지 않는다.
    """

    event_ids: tuple[uuid.UUID, ...]
    weight: float


def cosine_similarity(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    # 임베딩에 음수 성분이 있으면 코사인이 음수가 될 수 있다. 0으로 자른다 —
    # 음수를 그대로 두면 다른 항이 벌어 놓은 점수를 깎아 "무관함"이 "반대"로
    # 잘못 취급된다.
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        # 한쪽에 집합이 하나도 없으면 겹침의 증거도 반증도 없다 → 0.
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _time_decay(left: datetime | None, right: datetime | None, window_days: float) -> float:
    if left is None or right is None or window_days <= 0:
        return 0.0
    delta_days = abs((left - right).total_seconds()) / 86400.0
    return max(0.0, 1.0 - delta_days / window_days)


def _same_source_scale(
    left: datetime | None, right: datetime | None, full_at_days: float
) -> float:
    """같은 출처 보정을 몇 할 적용할지 — 발행 간격 0이면 0, `full_at_days` 이상이면 1.

    부르는 쪽은 발행 시각이 없으면 수집 시각(``COALESCE(published_at,
    created_at)``, 온라인은 지금 시각)을 넘긴다. 일부러 그대로 둔다: 한꺼번에
    긁힌 글은 간격이 0에 가까워 "모르면 보정 안 함"과 결과가 같고, 매일 새로
    긁힌 글은 수집 시각이 발행 시각에 가까워 오히려 다른 발표를 떼어 준다.
    모르면 전부 감점하는 쪽은 ARG-284의 일괄 감점으로 되돌아간다.
    """
    if full_at_days <= 0:
        return 1.0
    if left is None or right is None:
        # 간격을 모르면 "다른 발표"의 근거도 없다 — 출처를 모를 때처럼 보정하지 않는다.
        return 0.0
    gap_days = abs((left - right).total_seconds()) / 86400.0
    return min(1.0, gap_days / full_at_days)


def edge_weight(
    left: DocumentFeatures,
    right: DocumentFeatures,
    *,
    weights: EdgeWeights,
    window_days: float,
) -> float:
    """두 문서가 같은 사건일 근거를 0~1로 반환한다.

    분모는 항상 네 가중치의 총합이다 — 임베딩이 없어 코사인 항이 0으로
    깔려도 그 항의 가중치를 분모에서 빼지 않는다. 빼면 임베딩 없는 문서가
    나머지 세 항만으로 쉽게 임계값을 넘어 "정보가 적을수록 더 잘 묶인다"는
    역전이 생긴다.
    """
    total_weight = weights.cosine + weights.entity + weights.time + weights.keyword
    if total_weight <= 0:
        return 0.0

    cosine = cosine_similarity(left.embedding, right.embedding)
    score = (
        weights.cosine * cosine
        + weights.entity * _jaccard(left.names, right.names)
        + weights.time * _time_decay(left.at, right.at, window_days)
        + weights.keyword * _jaccard(left.keywords, right.keywords)
    )
    score /= total_weight
    if (
        weights.same_source_penalty > 0.0
        and left.source
        and right.source
        and left.source == right.source
        and cosine < NEAR_DUPLICATE_COSINE
    ):
        scale = _same_source_scale(left.at, right.at, weights.same_source_gap_days)
        score -= weights.same_source_penalty * scale
    if (
        weights.entity_mismatch_penalty > 0.0
        and len(left.names) >= weights.entity_mismatch_min_names
        and len(right.names) >= weights.entity_mismatch_min_names
        and not (left.names & right.names)
        and cosine < NEAR_DUPLICATE_COSINE
    ):
        score -= weights.entity_mismatch_penalty
    return max(0.0, score)


def choose_event(
    edges: Sequence[NeighborEdge],
    *,
    event_sizes: Mapping[uuid.UUID, int],
    join_threshold: float,
    resolution: float,
) -> uuid.UUID | None:
    """밤의 CPM이 받아들일 사건을 고른다. 없으면 ``None``(= 새 사건).

    사건 E의 이득 = (``join_threshold`` 이상인 이웃 점수의 합) − ``resolution`` ×
    E의 크기. 이득이 **0보다 큰** 사건 중 최댓값을 고른다.

    - **τ 미만 간선을 빼는 이유:** 밤의 그래프에는 그 간선이 없다. 낮만 세면
      약한 표가 쌓여 밤이 곧바로 되돌릴 배정을 만든다.
    - **크기에 대가를 매기는 이유:** CPM은 사건 안의 *모든* 쌍에 γ를 물린다.
      큰 사건에 붙으려면 그 사건 전체와 평균적으로 가까워야 하고, 이게 눈덩이를
      막는다.
    - **이득 0은 붙이지 않는다:** CPM에서 이득 0은 묶을 이유가 없다는 뜻이고,
      밤의 Leiden도 그 쌍을 하나로 두지 않는다.

    ``event_sizes``는 사건마다 부르는 쪽이 아는 크기(시간 창 안 문서 수)다. 표를
    던진 이웃 수보다 작을 수는 없으므로, 모르거나 작게 적힌 사건은 그 이웃
    수로 올려 잡는다 — 0으로 치면 대가가 사라져 무엇이든 붙는다.

    동점이면 사건 id의 문자열 오름차순으로 고른다 — 같은 입력이 항상 같은
    사건에 배정되게 하는, 함수 수준의 결정성 보장이다. 집합 순회 순서에
    기대면 파이썬 버전/실행마다 달라질 수 있어 명시적으로 정렬한다.
    """
    strong: dict[uuid.UUID, float] = {}
    voters: dict[uuid.UUID, int] = {}
    for edge in edges:
        for event_id in edge.event_ids:
            voters[event_id] = voters.get(event_id, 0) + 1
            if edge.weight >= join_threshold:
                strong[event_id] = strong.get(event_id, 0.0) + edge.weight

    best_id: uuid.UUID | None = None
    best_gain = 0.0
    for event_id in sorted(strong, key=str):
        size = max(event_sizes.get(event_id, 0), voters[event_id])
        gain = strong[event_id] - resolution * size
        if gain > best_gain:
            best_id, best_gain = event_id, gain
    return best_id
