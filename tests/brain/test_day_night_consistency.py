"""낮의 배정과 밤의 재군집이 같은 것을 잰다 — ARG-282.

같은 문서들을 두 경로에 넣고 판정을 비교한다. 낮은 ``decide_event``(새 문서 하나를
기존 사건에 붙일까), 밤은 ``detect_communities``(기간 전체를 Leiden으로 묶기)다.
DB는 쓰지 않는다.

배치는 2차원 단위벡터 각도로 통제한다. 모든 문서의 시각이 같고 이름·키워드가
없으므로 쌍 점수는 ``0.55·cos(Δθ) + 0.15``이고, 기준값 0.55를 넘으려면
``cos(Δθ) ≥ 0.727``(Δθ ≲ 0.76)이어야 한다.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

import pytest

from argos.brain.event_assignment import decide_event
from argos.brain.event_candidates import CandidateNeighbor
from argos.brain.event_scoring import DocumentFeatures
from argos.brain.recluster_core import detect_communities
from argos.brain.recluster_input import NeighborPair, ReclusterDocument
from argos.config import EventDetectionConfig
from tests.conftest import requires_graph_libs

_AT = datetime(2026, 10, 1, tzinfo=timezone.utc)
_EVENT = uuid.UUID(int=999)
_SUBJECT_ID = uuid.UUID(int=100)

# 서로 아주 가까운 사건 구성원 넷 (쌍 점수 ≈ 0.70).
_MEMBERS = (0.0, 0.02, 0.04, 0.06)


def _features(theta: float) -> DocumentFeatures:
    return DocumentFeatures(
        embedding=(math.cos(theta), math.sin(theta)),
        names=frozenset(),
        at=_AT,
        keywords=frozenset(),
    )


def _day_joins(subject_theta: float, config: EventDetectionConfig) -> bool:
    candidates = [
        CandidateNeighbor(
            tech_item_id=uuid.UUID(int=index + 1),
            features=_features(theta),
            event_ids=(_EVENT,),
        )
        for index, theta in enumerate(_MEMBERS)
    ]
    verdict = decide_event(
        _features(subject_theta),
        candidates,
        event_sizes={_EVENT: len(_MEMBERS)},
        config=config,
    )
    return verdict == _EVENT


def _night_joins(subject_theta: float, config: EventDetectionConfig) -> bool:
    documents = [
        ReclusterDocument(
            tech_item_id=uuid.UUID(int=index + 1),
            features=_features(theta),
            event_ids=(),
        )
        for index, theta in enumerate(_MEMBERS)
    ] + [
        ReclusterDocument(
            tech_item_id=_SUBJECT_ID, features=_features(subject_theta), event_ids=()
        )
    ]
    ids = sorted(doc.tech_item_id for doc in documents)
    pairs = [
        NeighborPair(left_id=left, right_id=right)
        for i, left in enumerate(ids)
        for right in ids[i + 1 :]
    ]
    communities = detect_communities(documents, pairs, config=config)
    member_ids = {uuid.UUID(int=index + 1) for index in range(len(_MEMBERS))}
    (home,) = [c for c in communities if _SUBJECT_ID in c.members]
    # 전제: 구성원 넷은 밤에도 한 덩어리다. 아니면 비교 자체가 무의미하다.
    assert any(member_ids <= set(c.members) for c in communities)
    return member_ids <= set(home.members)


# (새 문서 각도, 기대 판정, 설명)
_CASES = [
    (0.03, True, "넷 모두와 가깝다"),
    (-0.6, True, "넷 모두와 기준값을 조금 넘는다 — 합이 대가를 넘는다"),
    (-0.7, False, "기준값을 겨우 넘는 간선 셋 — 4건짜리 사건 전체에는 못 미친다"),
    (1.2, False, "아무와도 기준값을 못 넘는다"),
]


@requires_graph_libs
@pytest.mark.parametrize(("theta", "expected", "why"), _CASES)
def test_day_and_night_agree_on_the_same_documents(theta, expected, why):
    config = EventDetectionConfig()
    day = _day_joins(theta, config)
    night = _night_joins(theta, config)
    assert day == night == expected, f"{why}: 낮={day} 밤={night}"


@requires_graph_libs
def test_one_setting_moves_day_and_night_in_the_same_direction():
    """기준값을 낮추면 낮과 밤이 같이 더 잘 묶인다 (ARG-282 AC 1).

    예전 규칙에서는 낮만 합산이라, 같은 값을 바꿔도 둘의 경계가 따로 놀았다.
    """
    theta = -0.7
    strict = EventDetectionConfig(join_threshold=0.55)
    loose = EventDetectionConfig(join_threshold=0.45)

    assert (_day_joins(theta, strict), _night_joins(theta, strict)) == (False, False)
    assert (_day_joins(theta, loose), _night_joins(theta, loose)) == (True, True)


def test_the_cases_cover_both_verdicts():
    """등가성이 '항상 붙는다'나 '항상 안 붙는다'로 우연히 맞은 게 아님을 보인다."""
    config = EventDetectionConfig()
    assert {_day_joins(theta, config) for theta, _, _ in _CASES} == {True, False}
