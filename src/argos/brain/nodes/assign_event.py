"""문서를 사건에 배정하는 단계 — ARG-266.

``save_node`` **앞**에 선다. 저장 시점에 이미 사건이 정해져 있어야 하기
때문이다. 후보 이웃(ARG-265)을 뽑아 간선 가중치를 사건별로 합산하고
(ARG-264) 임계값을 넘는 사건이 있으면 그 id를 state에 싣는다.

넘는 사건이 없으면 ``event_id``를 ``None``으로 둔다 — 여기서 새 사건을
만들지 않는다. 사건을 미리 만들어 두면 뒤이은 저장이 실패했을 때 문서
하나 없는 빈 사건이 남는다. 새 사건 생성은 문서 저장과 같은 자리에서
``save_node``가 한다.

**``event_id=None``은 두 가지 다른 뜻을 가릴 수 있다** — "판정을 끝냈지만
임계값을 넘는 사건이 없었다"와 "판정 자체가 실패했다"는 서로 다르다. 앞의
경우만 새 사건을 만들어도 된다; 뒤의 경우 새 사건을 만들면 실패가 영구적인
잘못된 사건으로 굳어버려 나중 백필이 되돌릴 수 없다(무소속 문서는 링크가
없다는 신호로 찾지만, 잘못 만들어진 사건에는 링크가 **있다**). 그래서 이
둘을 ``event_assigned``로 명시적으로 분리한다:

- 배정이 끝까지 돌았다 → ``event_assigned=True`` (``event_id``는 찾은 사건
  또는 ``None`` — 새 사건이 필요하다는 뜻으로만 쓰인다).
- 임베딩이 없어 시도조차 안 했다, 또는 도중에 실패했다 → ``event_assigned=False``.
  이때 ``save_node``는 사건도 링크도 만들지 않는다 — 문서는 그냥 무소속으로
  저장되고, 링크 부재 자체가 나중 백필의 대상 표시가 된다.

이 단계의 실패는 삼킨다. 배정은 품질 기능이지 필수 경로가 아니다 —
실패하면 문서는 사건 없이 저장되고 나중 백필이 줍는다 (부모 확정 결정).

``fetch_candidates``만 ``session.begin_nested()`` 세이브포인트 안에서 부른다
— DB 예외(예: 임베딩 차원 불일치로 인한 ``InFailedSQLTransactionError``)를
그냥 잡기만 하면 세션의 트랜잭션은 이미 중단된(aborted) 상태로 남아, 뒤이어
``save_node``가 던지는 첫 쿼리부터 다시 실패해 문서 저장 자체가 무산된다.
세이브포인트가 그 중단 상태를 롤백해 세션을 다시 쓸 수 있게 돌려놓는다.

ARG-270: 판정과 후보 조회는 ``brain/event_assignment``의 공용 코어로 옮겼다.
이 노드는 ``BrainState``에서 피처를 꺼내 코어에 넘기고 결과를 다시 state에
싣는 얇은 껍데기다 — 소급 배정이 같은 코어를 부른다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.event_assignment import db_candidate_source, decide_event
from argos.brain.event_candidates import keywords_of
from argos.brain.event_scoring import DocumentFeatures
from argos.brain.graph_state import BrainState
from argos.config import settings

logger = logging.getLogger(__name__)


async def assign_event_node(state: BrainState, session: AsyncSession) -> BrainState:
    if not state.get("is_valid"):
        return state

    extracted_info = state.get("extracted_info") or {}
    embedding = extracted_info.get("embedding")
    if not embedding:
        return {**state, "event_id": None, "event_assigned": False}

    try:
        # 전제부(발행 시각 계산, config 조회)도 try 안에 둔다 — 여기서 나는
        # 예외(예: 설정 파싱 오류)도 "배정 실패, 새 사건은 만들지 않는다"로
        # 처리돼야 하기 때문이다.
        at = state.get("published_at") or datetime.now(timezone.utc)
        config = settings.user.event_detection

        # DB를 건드리는 부분만 세이브포인트로 감싼다 — 실패해도 세션을
        # 계속 쓸 수 있어야 한다(모듈 docstring 참고). 세이브포인트는
        # db_candidate_source 안에 있다.
        candidates = await db_candidate_source(
            session, embedding=embedding, at=at, config=config
        )

        subject = DocumentFeatures(
            embedding=tuple(float(value) for value in embedding),
            names=frozenset(state.get("entity_names") or ()),
            at=at,
            keywords=keywords_of(state.get("summary") or state.get("digest")),
        )
        event_id = decide_event(subject, candidates, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "assign_event_node: assignment failed for %s: %r — saving without an event",
            state.get("source_url"),
            exc,
        )
        return {**state, "event_id": None, "event_assigned": False}

    return {**state, "event_id": event_id, "event_assigned": True}
