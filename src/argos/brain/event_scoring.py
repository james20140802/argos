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


@dataclass(frozen=True)
class EdgeWeights:
    """네 항의 가중치. 합이 1일 필요는 없다 — `edge_weight`가 정규화한다."""

    cosine: float
    entity: float
    time: float
    keyword: float

    @classmethod
    def from_config(cls, config: "EventDetectionConfig") -> "EdgeWeights":
        return cls(
            cosine=config.weight_cosine,
            entity=config.weight_entity,
            time=config.weight_time,
            keyword=config.weight_keyword,
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

    score = (
        weights.cosine * cosine_similarity(left.embedding, right.embedding)
        + weights.entity * _jaccard(left.names, right.names)
        + weights.time * _time_decay(left.at, right.at, window_days)
        + weights.keyword * _jaccard(left.keywords, right.keywords)
    )
    return score / total_weight


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
