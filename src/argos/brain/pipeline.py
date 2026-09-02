from __future__ import annotations
import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession
from argos.brain.entity_extraction import extract_names
from argos.brain.entity_store import canonical_names
from argos.brain.event_naming import EvidenceDoc, apply_event_naming
from argos.brain.graph_state import BrainState
from argos.brain.nodes.triage import triage_node, batch_triage_states
from argos.brain.nodes.digest import digest_node, batch_digest_states
from argos.brain.nodes.embed import embed_and_search_node, batch_embed_and_search_node
from argos.brain.nodes.genealogist import genealogist_node
from argos.brain.near_duplicate import simhash
from argos.brain.nodes.assign_event import assign_event_node
from argos.brain.nodes.save import save_node
from argos.brain.llm_client import get_genealogist_llm_client
from argos.brain.titles import derive_title
from argos.models.tech_item import CategoryType

logger = logging.getLogger(__name__)


async def _assign_then_save(
    state: BrainState, session: AsyncSession, *, flush: bool = True
) -> BrainState:
    """``assign_event_node`` 뒤에 ``save_node``를 잇는다 — ARG-266.

    ``save_node``를 부르는 세 자리(genealogy-skip, low-trust, 정상 경로) 모두
    이 헬퍼를 거친다. ``flush``는 그대로 ``save_node``에 넘긴다 — 배치 경로가
    ``flush=False``로 불러 자체 savepoint 안에서 직접 flush하기 때문에, 여기서
    누락하면 배치가 매 문서마다 조용히 flush를 두 번 하게 된다.

    ``assign_event_node``는 계약상 이미 모든 예외를 삼키고
    ``event_assigned=False``로 돌아온다. 그래도 여기서 한 번 더 감싼다 —
    배정은 품질 기능이지 필수 경로가 아니므로(부모 AC), 그 계약이 나중에
    깨지더라도(회귀) 저장까지 막지 않기 위한 이중 방어다. ``event_assigned``도
    반드시 함께 ``False``로 둔다 — ``event_id=None``만으로는 "판정을 끝냈지만
    못 찾음"과 "판정 자체가 안 됨"을 구분할 수 없고, save_node는 후자일 때
    새 사건을 만들면 안 된다(assign_event.py 모듈 docstring 참고).

    ARG-273: 저장이 새 사건을 만들었으면(``created_event_id``) 그 자리에서
    이름·요약을 짓는다. 실패해도 저장을 되돌리지 않는다.
    """
    try:
        assigned = await assign_event_node(state, session=session)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_assign_then_save: assign_event_node raised for %s: %r — saving without an event",
            state.get("source_url"),
            exc,
        )
        assigned = {**state, "event_id": None, "event_assigned": False}
    saved = await save_node(assigned, session=session, flush=flush)

    # ARG-273: 방금 새로 생긴 사건에만 이름을 붙인다. 안 하면 백필 직후엔
    # 이름이 있어도 이후 유입되는 사건이 전부 무명이 된다. 기존 사건에 붙은
    # 경우는 link_document_to_event가 naming_stale만 세우고, 재명명은
    # ``backfill-events --rename-stale``이 배치로 처리한다 — 크롤 1건마다
    # 사건 전체를 다시 짓게 하면 파이프라인이 LLM 호출로 늘어진다.
    #
    # 명명 실패는 삼킨다. 사건은 무명 + naming_stale=True로 남아 재명명
    # 패스가 줍는다 — 배정과 같은 "품질 기능은 저장을 막지 않는다" 규약이다.
    # 삼키려면 세이브포인트가 필요하다(save_node 모듈 docstring 참고).
    created_event_id = saved.get("created_event_id")
    if created_event_id is not None:
        try:
            async with session.begin_nested():
                await apply_event_naming(
                    session,
                    created_event_id,
                    [
                        EvidenceDoc(
                            title=derive_title(saved.get("raw_text")),
                            summary=saved.get("summary") or saved.get("digest"),
                        )
                    ],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_assign_then_save: naming the new event for %s failed: %r",
                saved.get("source_url"),
                exc,
            )
    return saved


def _attach_extracted_names(states: list[BrainState]) -> list[BrainState]:
    """배치 단위로 이름을 뽑아 state에 싣는다.

    문서빈도를 배치 안에서 세는 추출기 계약 때문에 배치로 한 번에 부른다 —
    문서마다 따로 부르면 흔한 말을 걷어낼 근거가 사라진다. 유효하지 않은
    state는 애초에 저장되지 않으므로 건드리지 않는다. 추출이 실패해도 크롤을
    멈추면 안 되므로 예외를 삼키고 빈 목록으로 이어간다.

    ``extract_names``는 이 모듈에 이름으로 바인딩된 전역을 직접 부른다 —
    테스트는 ``monkeypatch.setattr(brain_pipeline, "extract_names", mock)``로
    갈아 끼운다 (이 파일의 다른 노드 함수들과 같은 관례). 기본 인자로
    def-time에 바인딩해 버리면 그 관용구가 조용히 안 먹는다.

    ARG-267: SimHash도 여기서 같이 계산해 싣는다 — raw_text를 이미 손에 든
    자리라 근거 수 집계를 위한 별도 패스를 만들지 않는다. 이름 추출과 달리
    실패할 일이 없는(예외를 던지지 않는 순수 함수) 계산이라 try/except 밖에
    둔다 — 이름 추출이 실패해 빈 목록으로 대체되는 경로에서도 SimHash는
    정상적으로 채워져야 한다.
    """
    valid_indices = [i for i, state in enumerate(states) if state.get("is_valid")]
    if not valid_indices:
        return states

    documents = [states[i]["raw_text"] for i in valid_indices]
    for i in valid_indices:
        states[i] = {**states[i], "simhash": simhash(states[i].get("raw_text") or "")}

    try:
        extracted_batch = extract_names(documents)
    except Exception as exc:  # noqa: BLE001
        logger.warning("_attach_extracted_names: extraction failed: %r", exc)
        for i in valid_indices:
            states[i] = {**states[i], "entity_names": [], "entity_names_extracted": []}
        return states

    for i, extracted in zip(valid_indices, extracted_batch):
        states[i] = {
            **states[i],
            "entity_names": canonical_names(extracted),
            "entity_names_extracted": list(extracted),
        }
    return states


async def run_brain_pipeline(
    raw_text: str,
    source_url: str,
    session: AsyncSession,
    *,
    source_category: CategoryType | None = None,
    published_at: datetime | None = None,
    image_url: str | None = None,
) -> BrainState:
    # source_category is an optional hint from the fetcher (e.g. RSS in ARG-52,
    # arXiv in ARG-53) indicating which category the source leans towards.
    # GitHub/HN fetchers do not pass it (defaults to None).
    # Callers in run_full_pipeline may forward item.get("source_category") here
    # once ARG-52/53 land; the field is ignored by current GitHub/HN paths.
    initial: BrainState = {
        "raw_text": raw_text,
        "source_url": source_url,
        "is_valid": False,
        "trust_score": None,
        "summary": None,
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": source_category,
        "category": None,
        "published_at": published_at,
        "image_url": image_url,
    }
    triaged = await triage_node(initial)
    if not triaged["is_valid"]:
        return triaged

    # ARG-173: longform digest from raw_content (14B), between triage and embed.
    # digest does not depend on embed/genealogy results, so running it here keeps
    # the 32B genealogist swap isolated to its own branch.
    digested = await digest_node(triaged)

    # Run embed_and_search first so we can decide whether to spend VRAM on the
    # 32B prewarm. On cold start the genealogist branch is skipped and we never
    # need to load the large model.
    embedded = await embed_and_search_node(digested, session=session)
    embedded = _attach_extracted_names([embedded])[0]
    if embedded.get("genealogy_skipped"):
        return await _assign_then_save(embedded, session=session)

    trust_score = digested.get("trust_score")
    from argos.config import settings as _settings
    threshold = _settings.user.genealogist.trust_skip_threshold
    if trust_score is not None and trust_score < threshold:
        skipped: BrainState = {
            **embedded,
            "genealogy_skipped": True,
            "genealogy_skip_reason": "low_trust",
        }
        return await _assign_then_save(skipped, session=session)

    prewarm_task = asyncio.create_task(get_genealogist_llm_client().prewarm("large"))
    try:
        genealogized = await genealogist_node(embedded, prewarm_task=prewarm_task)
    finally:
        if not prewarm_task.done():
            prewarm_task.cancel()
        with contextlib.suppress(BaseException):
            await prewarm_task

    # ARG-273: 32B를 내리고 나서 저장·명명으로 넘어간다. _assign_then_save는 새
    # 사건이 생기면 그 자리에서 8B로 이름을 짓는데(event_naming 모듈 docstring),
    # VRAM 예산이 한 번에 한 모델뿐이라 32B가 얹힌 채로 8B를 부르면 명명이 OOM으로
    # 죽고 사건이 무명으로 남는다. genealogist_node는 keep_alive="5m"으로 질의하니
    # 놔두면 실제로 얹혀 있다. 배치 경로가 계보 작업을 끝내고 unload하는 자리와
    # 같고, 단건 경로의 triage_node/digest_node가 각자 finally에서 내리는 관례와도
    # 같다 — 계보 분기만 예외였다.
    #
    # 내리기 실패는 삼킨다: 명명이 OOM으로 실패해도 저장은 진행돼야 하므로
    # (품질 기능은 저장을 막지 않는다) 여기서 예외를 올릴 이유가 없다.
    try:
        await get_genealogist_llm_client().unload("large")
    except Exception:  # noqa: BLE001
        pass

    return await _assign_then_save(genealogized, session=session)


def _make_initial_state(item: dict) -> BrainState:
    source_category = item.get("_source_category")
    return {
        "raw_text": item.get("raw_content") or "",
        "source_url": item.get("source_url", "").strip(),
        "is_valid": False,
        "trust_score": None,
        "summary": None,
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": source_category,
        "category": None,
        "published_at": item.get("_published_at"),
        "image_url": item.get("image_url"),
    }


async def run_batch_brain_pipeline(
    items: list[dict],
    session: AsyncSession,
    *,
    on_triage_item_done: Callable[[], None] | None = None,
    on_digest_item_done: Callable[[], None] | None = None,
    on_embed_item_done: Callable[[], None] | None = None,
    on_genealogy_item_done: Callable[[], None] | None = None,
    on_save_item_done: Callable[[], None] | None = None,
) -> list[BrainState]:
    """Process N items through the brain pipeline with 3 model swaps total.

    Stage order: batch triage (8B × 1 swap) → batch embed (/api/embed × 1 call)
    → trust-score gate → batch genealogy (32B × 1 swap) → per-item save.

    The single-URL run_brain_pipeline is preserved for backwards compatibility
    and single-URL callers (e.g. tests, future Slack Deep-Dive paths).

    The returned list is in **assignment order, not input order** — before Stage 4
    the states are sorted by ``(published_at, source_url)`` so event assignment is
    deterministic (ARG-266). Match results by ``source_url``, never by index.

    Parameters
    ----------
    items, session:
        See module-level usage.
    on_triage_item_done, on_digest_item_done, on_embed_item_done,
    on_genealogy_item_done, on_save_item_done:
        Optional zero-arg callbacks for per-item progress reporting in each
        stage. ``on_triage_item_done`` / ``on_digest_item_done`` /
        ``on_embed_item_done`` are forwarded to ``batch_triage_states`` /
        ``batch_digest_states`` / ``batch_embed_and_search_node``;
        ``on_genealogy_item_done`` fires inside the genealogy loop once per
        candidate; ``on_save_item_done`` fires once per state in the save
        loop (including invalid states that are skipped, so the bar reflects
        every queue slot). Default ``None`` preserves existing behavior —
        they are the UI-injection point used by the CLI to drive a Rich
        progress bar (ARG-92/ARG-101). Exceptions raised by callbacks are
        swallowed so a broken UI cannot abort the pipeline.
    """
    from argos.config import settings as _settings

    if not items:
        return []

    states = [_make_initial_state(item) for item in items]

    # ── Stage 1: batch triage (8B loaded once) ────────────────────────────
    triaged_states = await batch_triage_states(
        states, on_item_done=on_triage_item_done
    )

    # ── Stage 1.5: batch digest (14B loaded once) ─────────────────────────
    # ARG-173. Invalid states pass through untouched (no LLM call).
    digested_states = await batch_digest_states(
        triaged_states, on_item_done=on_digest_item_done
    )

    # ── Stage 2: batch embed + similarity search ──────────────────────────
    embedded_states = await batch_embed_and_search_node(
        digested_states, session, on_item_done=on_embed_item_done
    )

    # ── Stage 2.5: batch name extraction (ARG-263) ────────────────────────
    # 배정(ARG-266)이 새 문서 쪽 이름 항 입력으로 쓴다. raw_text가 이 시점
    # 모든 유효한 state에 다 있으므로 배치 전체를 한 번에 넘긴다.
    embedded_states = _attach_extracted_names(embedded_states)

    # ── Stage 3: trust-score gate + batch genealogy (32B loaded once) ─────
    threshold = _settings.user.genealogist.trust_skip_threshold
    genealogy_candidates: list[int] = []
    for i, s in enumerate(embedded_states):
        if not s.get("is_valid"):
            continue
        if s.get("genealogy_skipped"):
            continue
        trust = s.get("trust_score")
        if trust is not None and trust < threshold:
            embedded_states[i] = {
                **s,
                "genealogy_skipped": True,
                "genealogy_skip_reason": "low_trust",
            }
            continue
        genealogy_candidates.append(i)

    if genealogy_candidates:
        prewarm_task = asyncio.create_task(get_genealogist_llm_client().prewarm("large"))
        try:
            passed_prewarm = False
            for i in genealogy_candidates:
                try:
                    embedded_states[i] = await genealogist_node(
                        embedded_states[i],
                        prewarm_task=prewarm_task if not passed_prewarm else None,
                    )
                    passed_prewarm = True
                finally:
                    if on_genealogy_item_done is not None:
                        try:
                            on_genealogy_item_done()
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "run_batch_brain_pipeline on_genealogy_item_done raised: %r",
                                exc,
                            )
        finally:
            if not prewarm_task.done():
                prewarm_task.cancel()
            with contextlib.suppress(BaseException):
                await prewarm_task
        # Unload 32B after all genealogy work is done.
        try:
            await get_genealogist_llm_client().unload("large")
        except Exception:
            pass

    # ARG-266: 배정은 앞 문서가 만든 사건을 뒤 문서가 볼 수 있어야 하므로
    # 순서가 결과를 가른다. 크롤이 준 순서에 맡기면 같은 입력에 다른 배정이
    # 나온다 — (시각, URL) 오름차순으로 고정한다. id는 아직 없으므로 URL을
    # 두 번째 키로 쓴다(문서마다 유일하다).
    embedded_states.sort(
        key=lambda s: (
            s.get("published_at") or datetime.max.replace(tzinfo=timezone.utc),
            s.get("source_url") or "",
        )
    )

    # ── Stage 4: per-item save (savepoint + flush per item) ──────────────────
    #
    # Each item is saved inside a begin_nested() savepoint and flushed within
    # that savepoint.  Flushing inside the savepoint ensures that DB-deferred
    # constraint violations (unique, FK, vector) are caught by the surrounding
    # except block and mark only that item as failed, rather than aborting the
    # entire batch.  The flush=False flag on save_node means save_node itself
    # does not issue the flush; the savepoint block does it explicitly after
    # save_node returns, giving us the same pre-flush logic-error isolation
    # while also catching post-flush constraint errors per item.
    #
    # Assignment (ARG-266) runs one item at a time inside the same savepoint,
    # immediately before save — never in parallel — so an event created by
    # one document is visible to the next document's candidate query.
    results: list[BrainState] = []
    for s in embedded_states:
        try:
            if not s.get("source_url"):
                logger.warning("run_batch_brain_pipeline: state missing source_url, skipping")
                results.append(s)
                continue
            if s.get("triage_error"):
                # Ollama-down rows are is_valid=False and never persist
                # (save_node no-ops), so running the save path would only set
                # saved=True after an empty flush — inflating saved_new with
                # phantom inserts. Skip it; Stage 6 still retains the row for
                # retry via triage_error. (ARG-190)
                results.append(s)
                continue
            try:
                async with session.begin_nested():
                    saved = await _assign_then_save(s, session=session, flush=False)
                    await session.flush()
                    saved["saved"] = True
                results.append(saved)
            except Exception as exc:
                logger.warning(
                    "run_batch_brain_pipeline: save failed for %s: %r",
                    s.get("source_url"),
                    exc,
                )
                results.append(s)
        finally:
            if on_save_item_done is not None:
                try:
                    on_save_item_done()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "run_batch_brain_pipeline on_save_item_done raised: %r",
                        exc,
                    )
    return results
