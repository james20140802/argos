import uuid
from datetime import datetime, timezone
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
async def test_uses_the_large_slot_exclusively_with_configured_budget():
    from argos.config import settings

    cfg = settings.user.event_summary
    client = _client("TITLE: t\nSUMMARY: s")
    await generate_event_naming([EvidenceDoc(title="a", summary=None)], client=client)
    assert client.query.await_args.args[0] == "large"
    kwargs = client.query.await_args.kwargs
    assert kwargs["exclusive"] is True
    assert kwargs["num_ctx"] == cfg.num_ctx
    assert kwargs["think"] is False
    assert kwargs["timeout"] == cfg.timeout
    assert kwargs["temperature"] == cfg.temperature


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


@pytest.mark.parametrize(
    "raw, expected",
    [
        # 기호를 문자 단위로 지우면 깨지던 것들 (ARG-274 리뷰).
        ("C# 14 released", "C# 14 released"),
        ("snake_case 도입", "snake_case 도입"),
        ("Python __init__ 메서드 변경", "Python __init__ 메서드 변경"),
        ("2 * 3 계산", "2 * 3 계산"),
        # 짝이 맞는 마크다운 껍데기는 그대로 벗긴다.
        ("**중요** 발표", "중요 발표"),
        ("*강조* 텍스트", "강조 텍스트"),
        ("`snake_case` 사용", "snake_case 사용"),
        ("# 헤딩 제목", "헤딩 제목"),
        ("A **B** and `c_d` and C#", "A B and c_d and C#"),
    ],
)
def test_scrub_strips_markdown_wrappers_without_eating_literal_punctuation(raw, expected):
    from argos.brain.event_naming import _scrub

    assert _scrub(raw) == expected


@pytest.mark.asyncio
async def test_title_keeps_a_sharp_from_the_model_output():
    """종단 확인 — 모델이 낸 ``C#``이 제목까지 살아 남는다."""
    client = _client("TITLE: **C# 14** 출시\nSUMMARY: `snake_case` 지원 추가")

    naming = await generate_event_naming(
        [EvidenceDoc(title="a", summary="b")], client=client
    )

    assert naming == EventNaming(title="C# 14 출시", summary="snake_case 지원 추가")


@pytest.mark.asyncio
async def test_apply_event_naming_guards_the_write_on_the_snapshot_version():
    """스냅샷 버전을 주면 UPDATE가 그 버전에 걸린다 (ARG-274)."""
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=1))
    client = _client("TITLE: 사건 제목\nSUMMARY: 사건 요약")
    snapshot = datetime(2026, 9, 2, 4, 0, tzinfo=timezone.utc)

    assert await apply_event_naming(
        session,
        uuid.uuid4(),
        [EvidenceDoc(title="a", summary="b")],
        client=client,
        expected_updated_at=snapshot,
    ) is True

    where = str(session.execute.await_args.args[0]).split("WHERE", 1)[1]
    assert "tech_events.updated_at" in where
    assert snapshot in session.execute.await_args.args[0].compile().params.values()


@pytest.mark.asyncio
async def test_apply_event_naming_leaves_the_event_stale_when_it_changed_meanwhile():
    """가드가 걸리면(행이 안 맞음) False를 돌려 사건을 stale로 남긴다.

    LLM이 도는 동안 온라인 파이프라인이 문서를 하나 더 매달면
    link_document_to_event가 naming_stale을 다시 세운다. 옛 근거로 지은 이름을
    쓰면서 그 플래그까지 내리면 새 문서가 이름에도 --rename-stale에도 안 걸린다.
    """
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=0))
    client = _client("TITLE: 사건 제목\nSUMMARY: 사건 요약")

    assert await apply_event_naming(
        session,
        uuid.uuid4(),
        [EvidenceDoc(title="a", summary="b")],
        client=client,
        expected_updated_at=datetime(2026, 9, 2, 4, 0, tzinfo=timezone.utc),
    ) is False


@pytest.mark.asyncio
async def test_apply_event_naming_without_a_snapshot_does_not_guard():
    """온라인 경로는 스냅샷을 넘기지 않는다 — WHERE에 updated_at이 붙으면 안 된다."""
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=0))
    client = _client("TITLE: 사건 제목\nSUMMARY: 사건 요약")

    # rowcount=0이어도 가드를 안 걸었으므로 True다.
    assert await apply_event_naming(
        session, uuid.uuid4(), [EvidenceDoc(title="a", summary="b")], client=client
    ) is True
    # SET 절의 updated_at=now()(TimestampMixin의 onupdate)와 헷갈리면 안 되므로
    # WHERE 절만 본다.
    where = str(session.execute.await_args.args[0]).split("WHERE", 1)[1]
    assert "tech_events.updated_at" not in where


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


@pytest.mark.parametrize(
    "reply",
    [
        "TITLE: 사건 제목",
        "TITLE: 사건 제목\nSUMMARY:",
        "TITLE: 사건 제목\nSUMMARY:    ",
    ],
    ids=["no-summary-line", "blank-summary", "whitespace-summary"],
)
@pytest.mark.asyncio
async def test_returns_none_when_the_summary_is_missing_or_blank(reply):
    client = _client(reply)
    assert (
        await generate_event_naming([EvidenceDoc(title="a", summary="b")], client=client)
        is None
    )


@pytest.mark.asyncio
async def test_apply_writes_nothing_when_the_model_omits_the_summary():
    """제목만 온 출력이 기존 요약을 NULL로 덮고 플래그까지 내리면 안 된다."""
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock()
    client = _client("TITLE: 제목만 있다")

    assert await apply_event_naming(
        session, uuid.uuid4(), [EvidenceDoc(title="a", summary="b")], client=client
    ) is False
    session.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# ARG-310: body in the input, budget, map-reduce
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_call_when_evidence_fits_and_body_is_included():
    client = _client("TITLE: t\nSUMMARY: s")
    await generate_event_naming(
        [EvidenceDoc(title="h", summary="sum", body="BODY-TEXT-MARKER")], client=client
    )
    assert client.query.await_count == 1
    assert "BODY-TEXT-MARKER" in client.query.await_args.args[1]


@pytest.mark.asyncio
async def test_body_only_doc_counts_as_evidence():
    client = _client("TITLE: t\nSUMMARY: s")
    assert await generate_event_naming(
        [EvidenceDoc(title=None, summary=None, body="only a body")], client=client
    ) is not None


@pytest.mark.asyncio
async def test_body_is_capped_per_document(monkeypatch):
    from argos.config import settings

    monkeypatch.setattr(settings.user.event_summary, "doc_max_chars", 300)
    client = _client("TITLE: t\nSUMMARY: s")
    await generate_event_naming(
        [EvidenceDoc(title="h", summary="s", body="x" * 300 + "TAILMARK" * 50)], client=client
    )
    prompt = client.query.await_args.args[1]
    assert "TAILMARK" not in prompt  # past the 300-char cap
    assert "x" * 300 in prompt


@pytest.mark.asyncio
async def test_max_docs_cap_keeps_the_first_documents(monkeypatch):
    from argos.config import settings

    monkeypatch.setattr(settings.user.event_summary, "max_docs", 2)
    client = _client("TITLE: t\nSUMMARY: s")
    await generate_event_naming(
        [EvidenceDoc(title=f"doc-{n}", summary="s") for n in range(5)], client=client
    )
    prompt = client.query.await_args.args[1]
    assert "doc-0" in prompt and "doc-1" in prompt
    assert "doc-2" not in prompt


def _big_docs(n: int = 6, size: int = 5000) -> list[EvidenceDoc]:
    return [EvidenceDoc(title=f"doc-{i}", summary="s", body=f"B{i} " * (size // 3)) for i in range(n)]


def _scripted_client(final: str = "TITLE: 최종\nSUMMARY: 요약") -> AsyncMock:
    """Memo prompts get a short note; the final naming prompt gets ``final``."""

    async def query(role, prompt, **kwargs):
        if prompt.lstrip().startswith("You are preparing fact notes"):
            return "MEMO-LINE fact"
        return final

    client = AsyncMock()
    client.query = AsyncMock(side_effect=query)
    return client


@pytest.fixture
def small_budget(monkeypatch):
    from argos.config import settings

    monkeypatch.setattr(settings.user.event_summary, "num_ctx", 4096)
    monkeypatch.setattr(settings.user.event_summary, "output_reserve_tokens", 512)


@pytest.mark.asyncio
async def test_over_budget_evidence_runs_map_then_final(small_budget):
    client = _scripted_client()
    result = await generate_event_naming(_big_docs(), client=client)
    assert result == EventNaming(title="최종", summary="요약")
    prompts = [c.args[1] for c in client.query.await_args_list]
    assert client.query.await_count >= 3  # >=2 memo calls + 1 final
    assert all(p.lstrip().startswith("You are preparing fact notes") for p in prompts[:-1])
    assert prompts[-1].lstrip().startswith("You are a news editor")
    assert "MEMO-LINE fact" in prompts[-1]
    assert "B0 " not in prompts[-1]  # raw bodies do not reach the final call


@pytest.mark.asyncio
async def test_every_mapreduce_call_uses_same_ctx_and_exclusive(small_budget):
    client = _scripted_client()
    await generate_event_naming(_big_docs(), client=client)
    for call in client.query.await_args_list:
        assert call.kwargs["num_ctx"] == 4096
        assert call.kwargs["exclusive"] is True
        assert call.args[0] == "large"


@pytest.mark.asyncio
async def test_memo_call_failure_returns_none(small_budget):
    client = AsyncMock()
    client.query = AsyncMock(side_effect=RuntimeError("boom"))
    assert await generate_event_naming(_big_docs(), client=client) is None


@pytest.mark.asyncio
async def test_empty_memo_returns_none(small_budget):
    client = AsyncMock()
    client.query = AsyncMock(return_value="   ")
    assert await generate_event_naming(_big_docs(), client=client) is None


@pytest.mark.asyncio
async def test_fold_depth_exceeded_returns_none(small_budget, monkeypatch):
    import argos.brain.event_naming as en

    monkeypatch.setattr(en, "MAX_FOLD_DEPTH", 0)
    client = _scripted_client()
    assert await generate_event_naming(_big_docs(), client=client) is None
    client.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_many_memos_fold_again(small_budget, monkeypatch):
    """Enough documents that the memos alone exceed the budget fold a second level."""
    from argos.config import settings

    monkeypatch.setattr(settings.user.event_summary, "max_docs", 200)
    client = _scripted_client()
    docs = [EvidenceDoc(title=f"d{i}", summary="s", body="y" * 5000) for i in range(120)]
    result = await generate_event_naming(docs, client=client)
    assert result is not None
    assert client.query.await_args_list[-1].args[1].lstrip().startswith("You are a news editor")
