import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain.event_naming import EvidenceDoc, EventNaming, generate_event_naming


def _client(reply: str) -> AsyncMock:
    client = AsyncMock()
    client.query = AsyncMock(return_value=reply)
    return client


@pytest.mark.asyncio
async def test_parses_title_and_summary():
    client = _client("TITLE: OpenAI가 o5를 공개했다\nSUMMARY: 추론 성능이 크게 올랐다고 발표했다.")
    result = await generate_event_naming(
        [EvidenceDoc(title="o5 announced", summary="a new model")], client=client
    )
    assert result == EventNaming(
        title="OpenAI가 o5를 공개했다", summary="추론 성능이 크게 올랐다고 발표했다."
    )


@pytest.mark.asyncio
async def test_uses_the_small_model_role():
    client = _client("TITLE: t\nSUMMARY: s")
    await generate_event_naming([EvidenceDoc(title="a", summary=None)], client=client)
    assert client.query.await_args.args[0] == "small"


@pytest.mark.asyncio
async def test_strips_think_block_and_markdown_fence():
    client = _client(
        "<think>고민 중</think>\n```\nTITLE: **사건 제목**\nSUMMARY: 요약이다\n```"
    )
    result = await generate_event_naming([EvidenceDoc(title="a", summary=None)], client=client)
    assert result is not None
    assert result.title == "사건 제목"
    assert "```" not in result.title
    assert result.summary == "요약이다"


@pytest.mark.asyncio
async def test_returns_none_when_title_is_blank():
    client = _client("TITLE:\nSUMMARY: 요약만 있다")
    assert await generate_event_naming([EvidenceDoc(title="a", summary=None)], client=client) is None


@pytest.mark.asyncio
async def test_returns_none_when_llm_raises():
    client = AsyncMock()
    client.query = AsyncMock(side_effect=RuntimeError("ollama down"))
    assert await generate_event_naming([EvidenceDoc(title="a", summary=None)], client=client) is None


@pytest.mark.asyncio
async def test_returns_none_without_evidence():
    client = _client("TITLE: t\nSUMMARY: s")
    assert await generate_event_naming([], client=client) is None
    client.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_title_is_truncated_to_column_width():
    client = _client("TITLE: " + "가" * 600 + "\nSUMMARY: s")
    result = await generate_event_naming([EvidenceDoc(title="a", summary=None)], client=client)
    assert result is not None
    assert len(result.title) == 500


@pytest.mark.asyncio
async def test_prompt_carries_evidence_and_language_directive():
    client = _client("TITLE: t\nSUMMARY: s")
    await generate_event_naming(
        [EvidenceDoc(title="Claude 5 released", summary="Anthropic shipped it")],
        client=client,
    )
    prompt = client.query.await_args.args[1]
    assert "Claude 5 released" in prompt
    assert "Anthropic shipped it" in prompt
    assert "IMPORTANT: Write every natural-language output field" in prompt


@pytest.mark.asyncio
async def test_apply_event_naming_updates_and_clears_the_flag():
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock()
    client = _client("TITLE: 사건 제목\nSUMMARY: 사건 요약")
    event_id = uuid.uuid4()

    assert await apply_event_naming(
        session, event_id, [EvidenceDoc(title="a", summary="b")], client=client
    ) is True

    statement = session.execute.await_args.args[0]
    compiled = str(statement)
    assert "UPDATE tech_events" in compiled
    values = statement.compile().params
    assert values["title"] == "사건 제목"
    assert values["summary"] == "사건 요약"
    assert values["naming_stale"] is False


@pytest.mark.asyncio
async def test_apply_event_naming_writes_nothing_when_generation_fails():
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock()
    client = AsyncMock()
    client.query = AsyncMock(side_effect=RuntimeError("down"))

    assert await apply_event_naming(
        session, uuid.uuid4(), [EvidenceDoc(title="a", summary="b")], client=client
    ) is False
    session.execute.assert_not_awaited()
