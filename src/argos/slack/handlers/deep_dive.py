from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import select

from argos.brain.llm_client import get_llm_client
from argos.brain.ollama_client import LARGE_MODEL_TIMEOUT
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


async def _run_and_reply(
    client,
    channel_id: str | None,
    thread_ts: str | None,
    respond,
    tech_id: uuid.UUID,
) -> None:
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(TechItem).where(TechItem.id == tech_id)
            )
            item = result.scalar_one_or_none()

        if item is None:
            await respond(
                "해당 tech_id를 찾을 수 없습니다.",
                response_type="ephemeral",
                replace_original=False,
            )
            return

        prompt = _DEEP_DIVE_PROMPT.format(
            title=item.title,
            source_url=item.source_url,
            raw_content=item.raw_content[:3000],
        )

        llm = get_llm_client()
        analysis = await llm.unload_then_query(
            "small",
            "large",
            prompt,
            keep_alive="5m",
            timeout=LARGE_MODEL_TIMEOUT,
            think=False,
        )

        text = f"*Deep Dive: {item.title}*\n\n{analysis}"
        if client is not None and channel_id and thread_ts:
            await client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=text,
            )
        else:
            await respond(
                text,
                response_type="ephemeral",
                replace_original=False,
            )
    except Exception as exc:
        logger.exception("deep_dive background task failed: %r", exc)
        await respond(
            "심층 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            response_type="ephemeral",
            replace_original=False,
        )


async def handle_deep_dive(ack, body, respond, client=None):
    await ack()
    await respond(
        "🧠 70B 모델을 깨워 심층 분석 중입니다. 약 1분 정도 소요됩니다...",
        response_type="ephemeral",
        replace_original=False,
    )
    tech_id_str: str = body["actions"][0]["value"]
    try:
        tech_id = uuid.UUID(tech_id_str)
    except ValueError:
        await respond(
            "잘못된 tech_id입니다.",
            response_type="ephemeral",
            replace_original=False,
        )
        return
    channel_id = (body.get("channel") or {}).get("id")
    message_ts = (body.get("message") or {}).get("ts")
    asyncio.create_task(
        _run_and_reply(client, channel_id, message_ts, respond, tech_id)
    )
