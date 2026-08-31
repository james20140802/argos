"""온라인 배정과 소급 배정이 같은 판정을 내는지 못박는다 — ARG-275.

'같은 판정'의 대상은 **두 문서가 같은 사건에 묶이는지 여부**다. 사건 id 값
자체의 일치는 요구하지 않는다 — 생성 순서가 다르면 id는 달라진다.

DB도 LLM도 목이다. 릴리스 CI에 Postgres가 없으므로 이 테스트는 DB 없이
통과해야 한다.

**파라미터의 점수 산정 (기본 config: weight_cosine=0.55, weight_entity=0.25,
weight_time=0.15, weight_keyword=0.05, join_threshold=0.55, window_days=14.0):**

``_pair``가 만드는 두 문서는 entity/keyword 자카드가 항상 1.0, 시간차가
항상 3시간(=0.125일)으로 고정이다. 그래서 time 항은 모든 케이스에서
``1 - 0.125/14 = 0.9910714286``으로 동일하고, 점수는 코사인 유사도만의
1차함수가 된다:

    score = 0.55*cos + 0.25*1.0 + 0.15*0.9910714286 + 0.05*1.0
          = 0.55*cos + 0.4486607143

즉 ``join_threshold``(0.55)를 넘는 경계는 ``cos ≈ 0.1842532468``이다. 아래
파라미터 각각의 실측 코사인·점수·판정(파이썬으로 직접 계산, 반올림 6자리):

| partner_embedding              | cos       | score     | 판정 (join_threshold=0.55) |
|---------------------------------|-----------|-----------|------------------------------|
| [1.0, 0.0]                      | 1.000000  | 0.998661  | 위 (+0.448661) — 완전 동일   |
| [0.98, 0.2]                     | 0.979804  | 0.987553  | 위 (+0.437553)               |
| [0.5, 0.87]                     | 0.498284  | 0.722717  | 위 (+0.172717) — 경계와는 거리 있음 |
| [0.2, 0.98]                     | 0.199960  | 0.558639  | 위 (+0.008639) — 경계 바로 위 |
| [0.18, 0.9836667...]            | 0.180000  | 0.547661  | 아래 (-0.002339) — 경계 바로 아래 |
| [0.0, 1.0]                      | 0.000000  | 0.448661  | 아래 (-0.101339) — 직교, 훨씬 아래 |

브리프 원문의 ``[0.2, 0.98]`` 주석("임계값 아래")은 계산 오류였다 — 실제로는
임계값보다 근소하게(+0.0086) 위다. 단언은 등가성(온라인==소급)만 확인하므로
그 자체는 여전히 유효하지만, 주석은 실측값으로 바로잡는다.
``[0.18, 0.9836667...]``는 이 브리프에 없던 케이스로, 경계 바로 아래(참여
케이스가 항상 join되는 쪽으로 쏠리지 않도록)를 추가한 것이다.
"""

import math
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain import event_backfill
from argos.brain.event_assignment import decide_event
from argos.brain.event_backfill import BackfillDoc, plan_backfill
from argos.brain.event_candidates import CandidateNeighbor
from argos.brain.event_scoring import DocumentFeatures
from argos.config import settings

AT = datetime(2026, 9, 1, tzinfo=timezone.utc)

# cos(subject, partner) == 0.18 exactly, via a unit vector [c, sqrt(1-c^2)] —
# since the subject embedding [1.0, 0.0] is already unit, cosine_similarity
# reduces to the partner's first component.
_JUST_BELOW_THRESHOLD = [0.18, math.sqrt(1 - 0.18**2)]


def _session() -> MagicMock:
    @asynccontextmanager
    async def _cm():
        yield None

    session = MagicMock()
    session.begin_nested = MagicMock(side_effect=lambda: _cm())
    return session


def _features(embedding, names, at, keywords) -> DocumentFeatures:
    return DocumentFeatures(
        embedding=tuple(embedding),
        names=frozenset(names),
        at=at,
        keywords=frozenset(keywords),
    )


def _pair(cosine_partner):
    """첫 문서와 (유사도를 조절한) 둘째 문서."""
    first = _features([1.0, 0.0], ["anthropic"], AT, ["claude", "release"])
    second = _features(cosine_partner, ["anthropic"], AT + timedelta(hours=3), ["claude", "release"])
    return first, second


def _online_verdict(first: DocumentFeatures, second: DocumentFeatures) -> bool:
    """온라인 경로: 첫 문서가 이미 사건 E에 있고, 둘째가 들어온다."""
    config = settings.user.event_detection
    event_id = uuid.uuid4()
    neighbour = CandidateNeighbor(
        tech_item_id=uuid.uuid4(), features=first, event_ids=(event_id,)
    )
    return decide_event(second, [neighbour], config=config) == event_id


async def _backfill_verdict(
    monkeypatch, first: DocumentFeatures, second: DocumentFeatures
) -> bool:
    """소급 경로: 두 문서를 순회 엔진에 통과시킨다."""
    monkeypatch.setattr(event_backfill, "db_candidate_source", AsyncMock(return_value=[]))
    docs = [
        BackfillDoc(tech_item_id=uuid.uuid4(), features=first, title="a", summary=None),
        BackfillDoc(tech_item_id=uuid.uuid4(), features=second, title="b", summary=None),
    ]
    plan = await plan_backfill(_session(), docs, config=settings.user.event_detection)
    return len({assignment.event_id for assignment in plan.assignments}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "partner_embedding",
    [
        [1.0, 0.0],                   # cos=1.000000, score=0.998661 — 완전 동일, 임계값 훨씬 위
        [0.98, 0.2],                  # cos=0.979804, score=0.987553 — 임계값 위
        [0.5, 0.87],                  # cos=0.498284, score=0.722717 — 임계값 위 (경계와는 거리 있음)
        [0.2, 0.98],                  # cos=0.199960, score=0.558639 — 경계 바로 위 (+0.008639)
        _JUST_BELOW_THRESHOLD,        # cos=0.180000, score=0.547661 — 경계 바로 아래 (-0.002339)
        [0.0, 1.0],                   # cos=0.000000, score=0.448661 — 직교, 임계값 훨씬 아래
    ],
)
async def test_online_and_backfill_agree(monkeypatch, partner_embedding):
    first, second = _pair(partner_embedding)
    online = _online_verdict(first, second)
    backfill = await _backfill_verdict(monkeypatch, first, second)
    assert online == backfill, (
        f"판정 불일치: 온라인={online} 소급={backfill} "
        f"(partner_embedding={partner_embedding})"
    )


@pytest.mark.asyncio
async def test_the_parametrization_actually_covers_both_verdicts(monkeypatch):
    """등가성이 '항상 True'나 '항상 False'로 우연히 맞은 게 아님을 보인다."""
    verdicts = set()
    for partner in ([1.0, 0.0], [0.0, 1.0]):
        first, second = _pair(partner)
        verdicts.add(_online_verdict(first, second))
    assert verdicts == {True, False}
