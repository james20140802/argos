"""웹 프로세스 안에서 심층 분석을 하나씩 돌리는 실행기 (ARG-336).

요청은 즉시 접수(진행 칸 차지)하고, 실제 생성은 백그라운드 태스크가 모델 순번
(``model_turn``)을 기다렸다 한다. 서로 다른 사건 둘을 연달아 눌러도 둘 다 접수되고
순번대로 하나씩 돈다. 모델 호출 동안에는 DB 트랜잭션을 쥐지 않는다.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from argos.web.services.deep_analysis import (
    failure_code,
    mark_failed,
    mark_running,
    mark_waiting,
    save_result,
)

if TYPE_CHECKING:
    from argos.brain.event_deep_analysis import DeepAnalysis, DeepAnalysisDoc

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]
Generate = Callable[[Sequence["DeepAnalysisDoc"]], Awaitable["DeepAnalysis"]]

TURN_JOB = "web analyze-event"


# argos.brain 패키지는 import 시 argos.database를 끌고 온다. build_web_app()이 DB 모듈을
# 가져오지 않는다는 가드(tests/web/test_app.py)를 지키려고 아래 셋은 호출 시점에 가져온다.
@asynccontextmanager
async def model_turn(job: str, *, on_wait: Callable[[], None] | None = None):
    from argos.brain.model_turn import model_turn as _model_turn

    async with _model_turn(job, on_wait=on_wait) as info:
        yield info


def turn_is_held() -> bool:
    from argos.brain.model_turn import turn_is_held as _held

    return _held()


async def fetch_event_docs(session: AsyncSession, event_id: uuid.UUID):
    from argos.brain.event_deep_analysis import fetch_event_docs as _fetch

    return await _fetch(session, event_id)


async def generate_deep_analysis(docs, **kwargs):
    from argos.brain.event_deep_analysis import generate_deep_analysis as _generate

    return await _generate(docs, **kwargs)


def _default_session_factory() -> AbstractAsyncContextManager[AsyncSession]:
    # build_web_app()이 argos.database를 import하지 않도록 호출 시점에 가져온다.
    from argos.database import AsyncSessionLocal

    return AsyncSessionLocal()


async def _default_generate(docs: Sequence[DeepAnalysisDoc]) -> DeepAnalysis:
    return await generate_deep_analysis(list(docs), keep_alive=0, label="web analyze-event")


class DeepAnalysisRunner:
    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        generate: Generate | None = None,
    ) -> None:
        self._session_factory: SessionFactory = session_factory or _default_session_factory
        self._generate: Generate = generate or _default_generate
        self._tasks: set[asyncio.Task[Any]] = set()

    def start(self, event_id: uuid.UUID) -> asyncio.Task[None]:
        task = asyncio.create_task(self.run(event_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def run(self, event_id: uuid.UUID) -> None:
        """한 사건을 분석해 저장한다. 어떤 예외도 밖으로 던지지 않는다."""
        try:
            waiting: list[asyncio.Task[None]] = []

            def _on_wait() -> None:
                waiting.append(asyncio.get_running_loop().create_task(self._mark_waiting(event_id)))

            async with model_turn(TURN_JOB, on_wait=_on_wait):
                # 대기 표시가 늦게 커밋돼 Running을 덮지 않도록 먼저 끝낸다.
                for t in waiting:
                    await t
                async with self._session_factory() as session:
                    await mark_running(session, event_id)
                    docs = await fetch_event_docs(session, event_id)
                analysis = await self._generate(docs)
                async with self._session_factory() as session:
                    await save_result(session, event_id, analysis)
        except Exception as exc:
            code = failure_code(exc)
            if code == "Unknown":
                logger.exception("deep analysis failed for event %s", event_id)
            else:
                logger.warning("deep analysis failed for event %s: %s", event_id, exc)
            await self._fail(event_id, code)

    async def _mark_waiting(self, event_id: uuid.UUID) -> None:
        try:
            async with self._session_factory() as session:
                await mark_waiting(session, event_id)
        except Exception:
            logger.exception("could not mark deep analysis waiting for event %s", event_id)

    async def _fail(self, event_id: uuid.UUID, code: str) -> None:
        try:
            async with self._session_factory() as session:
                await mark_failed(session, event_id, code)
        except Exception:
            logger.exception("could not record deep analysis failure for event %s", event_id)
