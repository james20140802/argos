"""간선 가중치와 사건별 합산·임계값 판정 — ARG-264. DB도 LLM도 쓰지 않는다.

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

**야간 재군집(2단계)도 이 함수를 그대로 부른다.** 그래서 이 모듈은 사건이라는
개념 자체를 모른다 — `NeighborEdge`가 실어 나르는 `event_ids` 튜플 이상으로
사건 전용 자료구조(DB 모델, ORM row 등)에 결합하지 않는다. 온라인 배정이든
야간 전체 재계산이든 "두 문서가 얼마나 가까운가"와 "이웃들의 표를 사건별로
합산해 임계값과 비교한다"는 동일한 산식이어야 결과가 일관된다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

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
    weight는 각 사건에 그대로 더해진다 — 쪼개지 않는다. LP(Label Propagation)
    1스텝의 표준 형태다.
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
    join_threshold: float,
) -> uuid.UUID | None:
    """이웃들의 표를 사건별로 합산해 임계값을 넘는 최댓값 사건을 고른다.

    동점이면 사건 id의 문자열 오름차순으로 고른다 — 같은 입력이 항상 같은
    사건에 배정되게 하는, 함수 수준의 결정성 보장이다. 집합 순회 순서에
    기대면 파이썬 버전/실행마다 달라질 수 있어 명시적으로 정렬한다.
    """
    totals: dict[uuid.UUID, float] = {}
    for edge in edges:
        for event_id in edge.event_ids:
            totals[event_id] = totals.get(event_id, 0.0) + edge.weight

    best_id: uuid.UUID | None = None
    # -inf로 시작해야 합계 0.0인 후보도 선택될 수 있다 — join_threshold는
    # 0.0을 허용하므로(config ge=0.0) 0점 사건도 임계값 판정까지는 가야 한다.
    best_total = float("-inf")
    for event_id in sorted(totals, key=str):
        total = totals[event_id]
        if total > best_total:
            best_id, best_total = event_id, total

    return best_id if best_id is not None and best_total >= join_threshold else None
