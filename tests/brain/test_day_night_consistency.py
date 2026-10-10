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


# --- ARG-283: 큰 사건에서도 낮과 밤이 같다 ---------------------------------

_BIG_EVENT_SIZE = 60


def _big_event_verdicts(config: EventDetectionConfig) -> tuple[bool, bool]:
    """구성원 60건과 새 문서가 전부 같은 내용일 때 (낮, 밤)이 붙이는가.

    낮도 밤도 이웃을 SQL과 같은 규칙으로 고른다: 거리 동점이라 id 오름차순
    상위 K. 낮은 새 문서가 꼽은 K건만 보고, 사건 크기 대가는 60건 전부에
    매긴다 — PR #125 Codex 리뷰가 짚은 비대칭이 바로 이 모양이다.
    """
    member_ids = [uuid.UUID(int=index + 1) for index in range(_BIG_EVENT_SIZE)]
    k = config.candidate_k

    day_candidates = [
        CandidateNeighbor(
            tech_item_id=member_id, features=_features(0.0), event_ids=(_EVENT,)
        )
        for member_id in member_ids[:k]
    ]
    day = decide_event(
        _features(0.0),
        day_candidates,
        event_sizes={_EVENT: _BIG_EVENT_SIZE},
        config=config,
    ) == _EVENT

    all_ids = sorted([*member_ids, _SUBJECT_ID])
    documents = [
        ReclusterDocument(tech_item_id=doc_id, features=_features(0.0), event_ids=())
        for doc_id in all_ids
    ]
    pairs = {
        tuple(sorted((doc_id, other)))
        for doc_id in all_ids
        for other in [o for o in all_ids if o != doc_id][:k]
    }
    communities = detect_communities(
        documents,
        [NeighborPair(left_id=left, right_id=right) for left, right in sorted(pairs)],
        config=config,
    )
    (home,) = [c for c in communities if _SUBJECT_ID in c.members]
    night = set(member_ids) <= set(home.members)
    return day, night


@requires_graph_libs
def test_a_large_event_takes_a_matching_document_day_and_night():
    assert _big_event_verdicts(EventDetectionConfig()) == (True, True)


@requires_graph_libs
def test_a_cap_below_the_event_size_is_what_made_the_day_refuse():
    # 옛 기본값 25: 낮은 새 문서가 꼽은 25건만 보고 60건 전체의 대가를 치러
    # 새 사건을 만들고, 밤은 사건 자체를 쪼갠다. 상한을 올린 이유가 이것이다.
    day, night = _big_event_verdicts(EventDetectionConfig(candidate_k=25))
    assert day is False
    assert night is False


# --- ARG-295: 같은 출처 보정도 낮과 밤이 함께 움직인다 --------------------------

_SAME = "openai.com"


def _src_features(theta: float, source: str | None) -> DocumentFeatures:
    return DocumentFeatures(
        embedding=(math.cos(theta), math.sin(theta)),
        names=frozenset({"openai"}),
        at=_AT,
        keywords=frozenset(),
        source=source,
    )


def _pair_verdicts(
    existing_source: str, new_source: str, config: EventDetectionConfig
) -> tuple[bool, bool]:
    """기존 문서 하나(코사인 ≈0.8)에 새 문서가 (낮, 밤)에서 붙는가."""
    theta = math.acos(0.8)
    old = _src_features(0.0, existing_source)
    new = _src_features(theta, new_source)
    old_id, new_id = uuid.UUID(int=1), uuid.UUID(int=2)
    day = (
        decide_event(
            new,
            [CandidateNeighbor(tech_item_id=old_id, features=old, event_ids=(_EVENT,))],
            event_sizes={_EVENT: 1},
            config=config,
        )
        == _EVENT
    )
    documents = [
        ReclusterDocument(tech_item_id=old_id, features=old, event_ids=()),
        ReclusterDocument(tech_item_id=new_id, features=new, event_ids=()),
    ]
    communities = detect_communities(
        documents, [NeighborPair(left_id=old_id, right_id=new_id)], config=config
    )
    night = any({old_id, new_id} <= set(c.members) for c in communities)
    return day, night


@requires_graph_libs
def test_same_source_penalty_moves_day_and_night_together():
    assert _pair_verdicts(_SAME, _SAME, EventDetectionConfig()) == (True, True)
    corrected = EventDetectionConfig(same_source_penalty=0.3)
    assert _pair_verdicts(_SAME, _SAME, corrected) == (False, False)


@requires_graph_libs
def test_different_source_pair_unaffected_by_penalty():
    corrected = EventDetectionConfig(same_source_penalty=0.3)
    assert _pair_verdicts(_SAME, "techcrunch.com", corrected) == (True, True)


def _identical_same_source_documents(count: int) -> list[ReclusterDocument]:
    return [
        ReclusterDocument(
            tech_item_id=uuid.UUID(int=1000 + index),
            features=_src_features(0.3, _SAME),
            event_ids=(),
        )
        for index in range(count)
    ]


def _all_pairs(documents: list[ReclusterDocument]) -> list[NeighborPair]:
    ids = sorted(doc.tech_item_id for doc in documents)
    return [
        NeighborPair(left_id=left, right_id=right)
        for i, left in enumerate(ids)
        for right in ids[i + 1 :]
    ]


@requires_graph_libs
def test_large_identical_same_source_event_does_not_split():
    """근사 중복 면제 덕에 보정 세기와 무관하게 큰 사건이 갈라지지 않는다 (ARG-283)."""
    documents = _identical_same_source_documents(60)
    config = EventDetectionConfig(same_source_penalty=0.3)
    communities = detect_communities(documents, _all_pairs(documents), config=config)
    assert len(communities) == 1


@requires_graph_libs
def test_same_input_same_result_with_penalty():
    documents = _identical_same_source_documents(60)
    config = EventDetectionConfig(same_source_penalty=0.3)
    forward = detect_communities(documents, _all_pairs(documents), config=config)
    backward = detect_communities(
        list(reversed(documents)), list(reversed(_all_pairs(documents))), config=config
    )
    assert [sorted(c.members) for c in forward] == [sorted(c.members) for c in backward]

    candidates = [
        CandidateNeighbor(
            tech_item_id=doc.tech_item_id, features=doc.features, event_ids=(_EVENT,)
        )
        for doc in documents
    ]
    subject = _src_features(0.3, _SAME)
    verdicts = {
        decide_event(subject, candidates, event_sizes={_EVENT: 60}, config=config)
        for _ in range(2)
    }
    assert verdicts == {_EVENT}
