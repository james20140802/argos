from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import select

from argos.brain.ollama_client import LARGE_MODEL, LARGE_MODEL_TIMEOUT, SMALL_MODEL, query_ollama, unload_model
from argos.database import AsyncSessionLocal
from argos.models.tech_item import TechItem

logger = logging.getLogger(__name__)

_DEEP_DIVE_PROMPT = """You are an expert technology analyst. Provide a thorough deep-dive analysis of the following technology.

Technology: {title}
URL: {source_url}
Content:
{raw_content}

Analyze:
1. Core innovation and technical approach
2. Maturity and production-readiness
3. Ecosystem fit and integration potential
4. Risks and limitations
5. Verdict: Should this be tracked or adopted?

Be concise but substantive."""


async def _run_and_reply(body: dict, respond, tech_id: uuid.UUID) -> None:
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(TechItem).where(TechItem.id == tech_id)
            )
            item = result.scalar_one_or_none()

        if item is None:
            await respond("해당 tech_id를 찾을 수 없습니다.")
            return

        prompt = _DEEP_DIVE_PROMPT.format(
            title=item.title,
            source_url=item.source_url,
            raw_content=item.raw_content[:3000],
        )

        await unload_model(SMALL_MODEL)
        analysis = await query_ollama(
            LARGE_MODEL,
            prompt,
            keep_alive="5m",
            timeout=LARGE_MODEL_TIMEOUT,
            think=False,
        )

        await respond(f"*Deep Dive: {item.title}*\n\n{analysis}")
    except Exception as exc:
        logger.exception("deep_dive background task failed: %r", exc)
        await respond("심층 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")


async def handle_deep_dive(ack, body, respond):
    await ack()
    await respond("🧠 70B 모델을 깨워 심층 분석 중입니다. 약 1분 정도 소요됩니다...")
    tech_id_str: str = body["actions"][0]["value"]
    try:
        tech_id = uuid.UUID(tech_id_str)
    except ValueError:
        await respond("잘못된 tech_id입니다.")
        return
    asyncio.create_task(_run_and_reply(body, respond, tech_id))
