from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_store import attach_names
from argos.brain.event_assignment import link_document_to_event
from argos.brain.graph_state import BrainState
from argos.brain.simhash_storage import to_storage
from argos.brain.titles import derive_title
from argos.models.tech_item import CategoryType, TechItem
from argos.models.tech_succession import RelationType, TechSuccession

logger = logging.getLogger(__name__)

_RELATION_MAP: dict[str, RelationType] = {
    "Replace": RelationType.REPLACE,
    "Enhance": RelationType.ENHANCE,
    "Fork": RelationType.FORK,
}


async def save_node(
    state: BrainState, session: AsyncSession, *, flush: bool = True
) -> BrainState:
    """Persist a BrainState to the database.

    Parameters
    ----------
    flush:
        When ``True`` (default) an explicit ``await session.flush()`` is issued
        after adding the item, and ``state["saved"]`` is set to ``True`` only
        after that flush succeeds.  Pass ``flush=False`` in the batch pipeline
        so save_node does not flush and does not set ``saved=True``; the caller
        must flush inside a savepoint and set ``saved["saved"] = True`` only
        after the flush succeeds, ensuring a failed flush leaves the state with
        ``saved=False`` for correct retry handling.

        Note: TechItem.id is pre-assigned via ``uuid.uuid4()`` in the
        constructor, so ``saved_item_id`` and succession FKs are available
        regardless of whether flush was called.

    Autoflush caveat
    ----------------
    The session factory (``database.py``) leaves ``autoflush=True`` (SQLAlchemy
    default).  This means each ``session.execute(SELECT ...)`` call inside this
    function — e.g. the duplicate-URL check and the predecessor existence check
    — can still trigger an implicit flush for any pending items.  Passing
    ``flush=False`` eliminates the *explicit* per-item flush, reducing round-trips
    from N to 1 at the batch level, but does not suppress autoflush-triggered
    flushes during in-function SELECT queries.  This is an acceptable trade-off
    for the batch pipeline; callers that need strict flush control should wrap
    the session in a ``with session.no_autoflush:`` block.
    """
    if not state["is_valid"]:
        return state

    if not state["source_url"]:
        logger.warning("save_node: empty source_url, skipping")
        return state

    # ARG-269: 제목 파생 규칙은 사건 명명의 폴백과 공유한다 (brain/titles.py).
    title = derive_title(state["raw_text"])

    existing = await session.execute(
        select(TechItem.id).where(TechItem.source_url == state["source_url"])
    )
    if existing.scalar_one_or_none() is not None:
        return state

    # Pre-assign the PK so it is available for saved_item_id and succession FK
    # even when flush=False (SQLAlchemy populates callable defaults at flush
    # time, not at object construction, so we assign explicitly here).
    item = TechItem(
        id=uuid.uuid4(),
        title=title,
        source_url=state["source_url"],
        raw_content=state["raw_text"],
        summary=state.get("summary"),
        digest=state.get("digest"),
        # Use triage-decided category, falling back to ALPHA as a safe default
        # in case it was not set (e.g. state produced by an older code path).
        category=state.get("category") or CategoryType.ALPHA,
        trust_score=state.get("trust_score"),
        trust_rubric=state.get("trust_rubric"),
        published_at=state.get("published_at"),
        image_url=state.get("image_url"),
        # ARG-267: 근거 수 집계 때마다 본문을 다시 해싱하지 않으려고 저장한다.
        simhash=to_storage(state["simhash"]) if state.get("simhash") is not None else None,
    )

    extracted_info = state.get("extracted_info") or {}
    if "embedding" in extracted_info:
        item.embedding = extracted_info["embedding"]

    session.add(item)

    # ARG-266: 배정 노드(assign_event_node)가 고른 사건에 문서를 매단다.
    # event_assigned가 True일 때만 이 블록을 탄다 — False(또는 키 자체가
    # 없음)는 배정이 시도조차 안 됐거나 도중에 실패했다는 뜻이고, 이때는
    # 사건도 링크도 만들지 않는다. event_id=None을 "찾지 못함"과 "실패함"
    # 둘 다에 새 사건을 만드는 신호로 겹쳐 쓰면, 배정이 조직적으로 실패하는
    # 사고(설정 오류, DB 장애 등)가 문서마다 잘못된 사건을 하나씩 영구히
    # 남긴다 — 그 사건들은 링크가 "있어서" 나중 백필이 무소속으로 찾아내지도
    # 못한다(부모 AC: 배정에 **성공한** 문서만 무소속 없음을 보장한다).
    #
    # event_assigned=True이고 event_id가 None이면(임계값을 넘는 기존 사건이
    # 없었다는 뜻) 여기서 새 사건을 만든다 — 배정 노드가 미리 만들지 않는
    # 이유는 그 모듈 docstring 참고. occurred_at은 이 문서의 published_at
    # (없으면 지금) — 아직 flush 전이라 created_at은 쓸 수 없다. 링크 쓰기
    # 자체의 실패는 삼킨다: 사건 배정은 품질 기능이지 필수 경로가 아니다 —
    # 문서 저장 자체를 막으면 안 된다.
    #
    # 다만 **삼키려면 세이브포인트가 반드시 있어야 한다**(assign_event_node와
    # 같은 이유, 그 모듈 docstring 참고). 이 블록의 첫 execute가 autoflush로
    # 위의 item INSERT를 먼저 흘려보내기 때문에, 그 뒤 문장이 실패하면
    # (예: 동시에 지워진 사건을 가리켜 FK 위반) 트랜잭션은 중단됐는데 세션에
    # 남은 pending은 없는 상태가 된다. 그러면 아래 ``session.flush()``는
    # 아무것도 보내지 않아 예외도 안 나고 ``saved=True``가 그대로 찍힌다 —
    # 실제로는 한 줄도 안 들어갔는데. ``run_full_pipeline``은 그 saved를 믿고
    # crawl_queue에서 URL을 지우므로 문서가 영구히 사라진다. 배치 경로는 더
    # 나쁘다: 블록이 "정상" 종료돼 중단된 트랜잭션에 RELEASE SAVEPOINT를 쏘고,
    # 그 오염이 **다음** 문서 저장까지 끌고 내려간다.
    #
    # ARG-270: 실제 쓰기는 brain/event_assignment.link_document_to_event가
    # 한다 — 소급 배정이 같은 함수를 부른다. 세이브포인트는 여전히 여기서
    # 건다(삼키는 쪽이 거는 것이 규칙이다).
    if state.get("event_assigned"):
        try:
            async with session.begin_nested():
                link = await link_document_to_event(
                    session,
                    tech_item_id=item.id,
                    event_id=state.get("event_id"),
                    occurred_at=state.get("published_at") or datetime.now(timezone.utc),
                )
            # ARG-273: 새로 생긴 사건만 명명 훅이 주워 가도록 노출한다.
            # 기존 사건에 붙은 경우는 naming_stale만 서고 재명명은
            # ``backfill-events --rename-stale``이 배치로 처리한다.
            state["created_event_id"] = link.event_id if link.created else None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "save_node: linking %s to an event failed: %r", state["source_url"], exc
            )

    # ARG-263: 이 문서에서 뽑은 이름을 가제티어 + document_entities 링크로
    # 옮긴다. item.id는 생성자에서 미리 배정돼 flush 전에도 쓸 수 있다.
    # 실패해도 저장 자체는 막지 않는다 — 이름 매달기가 크롤을 멈추면 안 된다.
    # 위 사건 블록과 같은 이유로 세이브포인트가 필요하다: 삼킨 예외가 중단된
    # 트랜잭션을 남기면 뒤따르는 flush는 보낼 게 없어 조용히 통과하고
    # ``saved=True``가 거짓말이 된다.
    extracted = state.get("entity_names_extracted") or []
    if extracted:
        try:
            async with session.begin_nested():
                await attach_names(session, item.id, extracted)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "save_node: attaching names failed for %s: %r", state["source_url"], exc
            )

    if flush:
        await session.flush()
        state["saved"] = True
    # Surface the new item's PK so downstream stages (ARG-103: succession
    # alerts) can collect just the freshly-saved IDs without re-querying.
    # Pre-assigned via uuid.uuid4() so this is available regardless of flush.
    state["saved_item_id"] = item.id

    succession_result = state.get("succession_result")
    if succession_result is not None and succession_result.get("replace_target_id") is not None:
        relation_str = succession_result.get("relation_type")
        mapped_enum = _RELATION_MAP.get(relation_str) if relation_str else None
        if relation_str and mapped_enum is None:
            logger.warning("save_node: unrecognized relation_type %r, skipping succession", relation_str)
        if mapped_enum is not None:
            try:
                predecessor_uuid = uuid.UUID(succession_result["replace_target_id"])
            except (ValueError, AttributeError):
                logger.warning(
                    "save_node: invalid replace_target_id UUID %r, skipping succession",
                    succession_result["replace_target_id"],
                )
                predecessor_uuid = None
            if predecessor_uuid is not None:
                predecessor_exists = await session.execute(
                    select(TechItem.id).where(TechItem.id == predecessor_uuid)
                )
                if predecessor_exists.scalar_one_or_none() is None:
                    logger.warning(
                        "save_node: predecessor %s not found in DB, skipping succession",
                        predecessor_uuid,
                    )
                else:
                    succession = TechSuccession(
                        predecessor_id=predecessor_uuid,
                        successor_id=item.id,
                        relation_type=mapped_enum,
                        reasoning=succession_result.get("reason", ""),
                    )
                    session.add(succession)

    return state
