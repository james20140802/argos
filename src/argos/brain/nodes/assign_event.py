"""문서를 사건에 배정하는 단계 — ARG-266.

``save_node`` **앞**에 선다. 저장 시점에 이미 사건이 정해져 있어야 하기
때문이다. 후보 이웃(ARG-265)을 뽑아 간선 가중치를 사건별로 합산하고
(ARG-264) 임계값을 넘는 사건이 있으면 그 id를 state에 싣는다.

넘는 사건이 없으면 ``event_id``를 ``None``으로 둔다 — 여기서 새 사건을
만들지 않는다. 사건을 미리 만들어 두면 뒤이은 저장이 실패했을 때 문서
하나 없는 빈 사건이 남는다. 새 사건 생성은 문서 저장과 같은 자리에서
``save_node``가 한다.

이 단계의 실패는 삼킨다. 배정은 품질 기능이지 필수 경로가 아니다 —
실패하면 문서는 사건 없이 저장되고 나중 백필이 줍는다 (부모 확정 결정).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.event_candidates import fetch_candidates, keywords_of
from argos.brain.event_scoring import DocumentFeatures, EdgeWeights, NeighborEdge, choose_event, edge_weight
from argos.brain.graph_state import BrainState
from argos.config import settings

logger = logging.getLogger(__name__)


async def assign_event_node(state: BrainState, session: AsyncSession) -> BrainState:
    if not state.get("is_valid"):
        return state

    extracted_info = state.get("extracted_info") or {}
    embedding = extracted_info.get("embedding")
    if not embedding:
        return {**state, "event_id": None}

    at = state.get("published_at") or datetime.now(timezone.utc)
    config = settings.user.event_detection
    try:
        candidates = await fetch_candidates(session, embedding=embedding, at=at)
        subject = DocumentFeatures(
            embedding=tuple(float(value) for value in embedding),
            names=frozenset(state.get("entity_names") or ()),
            at=at,
            keywords=keywords_of(state.get("summary") or state.get("digest")),
        )
        weights = EdgeWeights.from_config(config)
        edges = [
            NeighborEdge(
                event_ids=candidate.event_ids,
                weight=edge_weight(
                    subject,
                    candidate.features,
                    weights=weights,
                    window_days=config.window_days,
                ),
            )
            for candidate in candidates
            if candidate.event_ids
        ]
        event_id = choose_event(edges, join_threshold=config.join_threshold)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "assign_event_node: assignment failed for %s: %r — saving without an event",
            state.get("source_url"),
            exc,
        )
        return {**state, "event_id": None}

    return {**state, "event_id": event_id}
