import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.brain.event_detail_schema import Section
from argos.brain.event_naming import EvidenceDoc, EventNaming, generate_event_naming


def _reply(
    title: str = "사건 제목",
    summary: str = "사건 요약이다. 두 번째 문장이다.",
    event_type: str = "Launch",
    sections: list[dict] | None = None,
) -> str:
    if sections is None:
        sections = [{"heading": "무엇이 새로운가", "body": "새 기능이 나왔다."}]
    return json.dumps(
        {"title": title, "summary": summary, "event_type": event_type, "sections": sections},
        ensure_ascii=False,
    )


def _client(*replies: str) -> AsyncMock:
    client = AsyncMock()
    if len(replies) == 1:
        client.query = AsyncMock(return_value=replies[0])
    else:
        client.query = AsyncMock(side_effect=list(replies))
    return client


@pytest.mark.asyncio
async def test_parses_title_summary_type_and_sections():
    client = _client(_reply(title="OpenAI가 o5를 공개했다", summary="추론 성능이 올랐다."))
    result = await generate_event_naming([EvidenceDoc(title="o5 announced", body="b")], client=client)
    assert result == EventNaming(
        title="OpenAI가 o5를 공개했다",
        summary="추론 성능이 올랐다.",
        event_type="Launch",
        sections=(Section("무엇이 새로운가", "새 기능이 나왔다."),),
    )
    assert client.query.await_count == 1


@pytest.mark.asyncio
async def test_uses_the_large_slot_exclusively_with_configured_budget_and_schema():
    from argos.config import settings

    cfg = settings.user.event_summary
    client = _client(_reply())
    await generate_event_naming([EvidenceDoc(title="a")], client=client)
    assert client.query.await_args.args[0] == "large"
    kwargs = client.query.await_args.kwargs
    assert kwargs["exclusive"] is True
    assert kwargs["num_ctx"] == cfg.num_ctx
    assert kwargs["think"] is False
    assert kwargs["timeout"] == cfg.timeout
    assert kwargs["temperature"] == cfg.temperature
    assert kwargs["format"]["properties"]["event_type"]["enum"] == [
        "Launch", "Research", "Incident", "Update", "Adoption",
    ]


def test_default_output_reserve_leaves_room_for_sections():
    from argos.config import EventSummaryConfig

    assert EventSummaryConfig().output_reserve_tok == 2048


@pytest.mark.asyncio
async def test_strips_think_block_and_markdown_fence():
    client = _client("<think>고민 중</think>\n```json\n" + _reply(title="**사건 제목**") + "\n```")
    result = await generate_event_naming([EvidenceDoc(title="a")], client=client)
    assert result is not None
    assert result.title == "사건 제목"


@pytest.mark.asyncio
async def test_returns_none_when_llm_raises():
    client = AsyncMock()
    client.query = AsyncMock(side_effect=RuntimeError("ollama down"))
    assert await generate_event_naming([EvidenceDoc(title="a")], client=client) is None


@pytest.mark.asyncio
async def test_returns_none_without_evidence():
    client = _client(_reply())
    assert await generate_event_naming([], client=client) is None
    assert await generate_event_naming([EvidenceDoc(title=None, summary="only summary")], client=client) is None
    client.query.assert_not_awaited()


# ---------------------------------------------------------------------------
# ARG-308: 입력 — 본문 전부, 요약 제외, 출처, 관점 지시
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_uses_bodies_not_item_summaries():
    client = _client(_reply())
    await generate_event_naming(
        [
            EvidenceDoc(
                title="Claude 5 released",
                summary="ITEM-SUMMARY-MARKER",
                body="BODY-TEXT-MARKER",
                source="https://www.anthropic.com/news/claude-5",
            )
        ],
        client=client,
    )
    prompt = client.query.await_args.args[1]
    assert "Claude 5 released" in prompt
    assert "BODY-TEXT-MARKER" in prompt
    assert "ITEM-SUMMARY-MARKER" not in prompt
    assert "출처: anthropic.com" in prompt
    assert "IMPORTANT: Write every natural-language output field" in prompt


@pytest.mark.asyncio
async def test_missing_body_is_marked_so_the_model_does_not_invent():
    client = _client(_reply())
    await generate_event_naming([EvidenceDoc(title="headline only", body=None)], client=client)
    prompt = client.query.await_args.args[1]
    assert "본문: (본문 없음)" in prompt
    assert "When an article body is short or missing" in prompt


@pytest.mark.asyncio
async def test_prompt_asks_to_cover_every_perspective_and_lists_every_article():
    client = _client(_reply())
    await generate_event_naming(
        [
            EvidenceDoc(title="Vendor says it is safe", body="VENDOR-VIEW"),
            EvidenceDoc(title="Researchers disagree", body="CRITIC-VIEW"),
        ],
        client=client,
    )
    prompt = client.query.await_args.args[1]
    assert "cover EVERY perspective" in prompt
    assert "[1] Vendor says it is safe" in prompt and "VENDOR-VIEW" in prompt
    assert "[2] Researchers disagree" in prompt and "CRITIC-VIEW" in prompt


@pytest.mark.asyncio
async def test_prompt_lists_the_fixed_headings_per_type():
    client = _client(_reply())
    await generate_event_naming([EvidenceDoc(title="a")], client=client)
    prompt = client.query.await_args.args[1]
    assert "Research (연구): 방법 / 결과 / 한계" in prompt
    assert "Adoption (도입): 누가 도입했나 / 무엇이 좋아졌나" in prompt


@pytest.mark.asyncio
async def test_body_is_capped_per_document(monkeypatch):
    from argos.config import settings

    monkeypatch.setattr(settings.user.event_summary, "doc_max_chars", 300)
    client = _client(_reply())
    await generate_event_naming(
        [EvidenceDoc(title="h", body="x" * 300 + "TAILMARK" * 50)], client=client
    )
    prompt = client.query.await_args.args[1]
    assert "TAILMARK" not in prompt
    assert "x" * 300 in prompt


@pytest.mark.asyncio
async def test_many_documents_share_the_budget_in_one_call():
    """기사 20건 × 6000자는 예산을 넘지만, 기사별로 나눠 주면 한 번에 다 들어간다."""
    client = _client(_reply())
    docs = [EvidenceDoc(title=f"doc-{i}", body=f"HEAD{i} " + "z" * 6000) for i in range(20)]
    await generate_event_naming(docs, client=client)
    assert client.query.await_count == 1
    prompt = client.query.await_args.args[1]
    for i in range(20):
        assert f"HEAD{i} " in prompt


@pytest.mark.asyncio
async def test_max_docs_cap_keeps_the_first_documents(monkeypatch):
    from argos.config import settings

    monkeypatch.setattr(settings.user.event_summary, "max_docs", 2)
    client = _client(_reply())
    await generate_event_naming([EvidenceDoc(title=f"doc-{n}") for n in range(5)], client=client)
    prompt = client.query.await_args.args[1]
    assert "doc-0" in prompt and "doc-1" in prompt
    assert "doc-2" not in prompt


# ---------------------------------------------------------------------------
# ARG-308: 재시도 — 이름 누락 / JSON 깨짐
# ---------------------------------------------------------------------------

_DUET = EvidenceDoc(
    title="Anthropic launches Claude Duet",
    body="Anthropic today launched Claude Duet, a paired-model mode.",
    names=("Claude Duet", "Anthropic"),
)


@pytest.mark.asyncio
async def test_transliterated_name_triggers_one_retry_that_names_it():
    client = _client(
        _reply(title="앤트로픽, 클라우드 둈트 출시", summary="둈트가 나왔다."),
        _reply(title="Anthropic, Claude Duet 출시", summary="Claude Duet이 나왔다."),
    )
    result = await generate_event_naming([_DUET], client=client)
    assert result is not None and result.title == "Anthropic, Claude Duet 출시"
    assert client.query.await_count == 2
    first, second = (c.args[1] for c in client.query.await_args_list)
    assert "MUST appear verbatim" not in first
    assert "MUST appear verbatim" in second
    assert "Claude Duet" in second.split("Articles:")[0]
    assert all(c.kwargs["exclusive"] is True for c in client.query.await_args_list)


@pytest.mark.asyncio
async def test_name_missing_twice_returns_none():
    bad = _reply(title="앤트로픽, 클라우드 둈트 출시", summary="둈트가 나왔다.")
    client = _client(bad, bad)
    assert await generate_event_naming([_DUET], client=client) is None
    assert client.query.await_count == 2


@pytest.mark.asyncio
async def test_broken_json_is_retried():
    client = _client("{not json", _reply(title="Anthropic, Claude Duet 출시"))
    result = await generate_event_naming([_DUET], client=client)
    assert result is not None
    assert client.query.await_count == 2


@pytest.mark.asyncio
async def test_invalid_twice_returns_none():
    client = _client("{not json", _reply(sections=[{"heading": "방법", "body": "x"}]))
    assert await generate_event_naming([EvidenceDoc(title="a")], client=client) is None


@pytest.mark.asyncio
async def test_body_only_generic_entity_does_not_trigger_a_retry():
    """실측 bcac21ae: 본문에만 나온 'German-language'를 빠뜨려도 실패가 아니다."""
    doc = EvidenceDoc(
        title="OpenAI disrupts covert influence operations",
        body="A German-language network used ChatGPT. OpenAI banned it.",
        names=("OpenAI", "German-language"),
    )
    client = _client(_reply(title="OpenAI, 독일어 영향 공작 차단"))
    assert await generate_event_naming([doc], client=client) is not None
    assert client.query.await_count == 1


@pytest.mark.asyncio
async def test_names_not_in_the_evidence_are_not_required():
    doc = EvidenceDoc(title="a", body="nothing here", names=("Gemini",))
    client = _client(_reply(title="제목"))
    assert await generate_event_naming([doc], client=client) is not None
    assert client.query.await_count == 1


# ---------------------------------------------------------------------------
# apply_event_naming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_event_naming_writes_detail_and_clears_the_flag():
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock()
    client = _client(_reply(title="사건 제목", summary="사건 요약", event_type="Research",
                            sections=[{"heading": "결과", "body": "정확도가 올랐다."}]))
    assert await apply_event_naming(
        session, uuid.uuid4(), [EvidenceDoc(title="a", body="b")], client=client
    ) is True

    statement = session.execute.await_args.args[0]
    assert "UPDATE tech_events" in str(statement)
    values = statement.compile().params
    assert values["title"] == "사건 제목"
    assert values["summary"] == "사건 요약"
    assert values["event_type"] == "Research"
    assert values["sections"] == [{"heading": "결과", "body": "정확도가 올랐다."}]
    assert values["naming_stale"] is False


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("C# 14 released", "C# 14 released"),
        ("snake_case 도입", "snake_case 도입"),
        ("Python __init__ 메서드 변경", "Python __init__ 메서드 변경"),
        ("2 * 3 계산", "2 * 3 계산"),
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
async def test_apply_event_naming_guards_the_write_on_the_snapshot_version():
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=1))
    snapshot = datetime(2026, 9, 2, 4, 0, tzinfo=timezone.utc)
    assert await apply_event_naming(
        session, uuid.uuid4(), [EvidenceDoc(title="a")], client=_client(_reply()),
        expected_updated_at=snapshot,
    ) is True
    where = str(session.execute.await_args.args[0]).split("WHERE", 1)[1]
    assert "tech_events.updated_at" in where
    assert snapshot in session.execute.await_args.args[0].compile().params.values()


@pytest.mark.asyncio
async def test_apply_event_naming_leaves_the_event_stale_when_it_changed_meanwhile():
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=0))
    assert await apply_event_naming(
        session, uuid.uuid4(), [EvidenceDoc(title="a")], client=_client(_reply()),
        expected_updated_at=datetime(2026, 9, 2, 4, 0, tzinfo=timezone.utc),
    ) is False


@pytest.mark.asyncio
async def test_apply_event_naming_without_a_snapshot_does_not_guard():
    from argos.brain.event_naming import apply_event_naming

    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=0))
    assert await apply_event_naming(
        session, uuid.uuid4(), [EvidenceDoc(title="a")], client=_client(_reply())
    ) is True
    where = str(session.execute.await_args.args[0]).split("WHERE", 1)[1]
    assert "tech_events.updated_at" not in where


@pytest.mark.parametrize(
    "client",
    [
        pytest.param("raise", id="llm-down"),
        pytest.param("names", id="name-dropped-twice"),
        pytest.param("blank-summary", id="blank-summary-twice"),
    ],
)
@pytest.mark.asyncio
async def test_apply_writes_nothing_when_generation_fails(client):
    """실패하면 이전 이름·요약·상세를 그대로 둔다 — UPDATE 자체가 없다."""
    from argos.brain.event_naming import apply_event_naming

    if client == "raise":
        llm = AsyncMock()
        llm.query = AsyncMock(side_effect=RuntimeError("down"))
    elif client == "names":
        bad = _reply(title="클라우드 둈트", summary="둈트")
        llm = _client(bad, bad)
    else:
        llm = _client(_reply(summary=""), _reply(summary="   "))
    session = MagicMock()
    session.execute = AsyncMock()
    assert await apply_event_naming(session, uuid.uuid4(), [_DUET], client=llm) is False
    session.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# ARG-310: map-reduce (본문이 기사별 바닥 아래로 깎일 만큼 많을 때)
# ---------------------------------------------------------------------------


def _big_docs(n: int = 6, size: int = 5000) -> list[EvidenceDoc]:
    return [EvidenceDoc(title=f"doc-{i}", body=f"B{i} " * (size // 3)) for i in range(n)]


def _scripted_client(final: str | None = None) -> AsyncMock:
    final = final or _reply(title="최종", summary="요약")

    async def query(role, prompt, **kwargs):
        if prompt.lstrip().startswith("You are preparing fact notes"):
            return "[1] MEMO-LINE fact"
        return final

    client = AsyncMock()
    client.query = AsyncMock(side_effect=query)
    return client


@pytest.fixture
def small_budget(monkeypatch):
    from argos.config import settings

    monkeypatch.setattr(settings.user.event_summary, "num_ctx", 4096)
    monkeypatch.setattr(settings.user.event_summary, "output_reserve_tok", 512)


@pytest.mark.asyncio
async def test_over_budget_evidence_runs_map_then_final(small_budget):
    client = _scripted_client()
    result = await generate_event_naming(_big_docs(), client=client)
    assert result is not None and result.title == "최종"
    prompts = [c.args[1] for c in client.query.await_args_list]
    assert client.query.await_count >= 3
    assert all(p.lstrip().startswith("You are preparing fact notes") for p in prompts[:-1])
    assert "article number" in prompts[0] and "different positions" in prompts[0]
    assert prompts[-1].lstrip().startswith("You are a technology news editor")
    assert "MEMO-LINE fact" in prompts[-1]
    assert "B0 " not in prompts[-1]
    # 메모 호출은 자유 텍스트, 최종 호출만 JSON 스키마
    calls = client.query.await_args_list
    assert all(c.kwargs["format"] is None for c in calls[:-1])
    assert calls[-1].kwargs["format"] is not None


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
    from argos.config import settings

    monkeypatch.setattr(settings.user.event_summary, "max_docs", 200)
    client = _scripted_client()
    docs = [EvidenceDoc(title=f"d{i}", body="y" * 5000) for i in range(120)]
    result = await generate_event_naming(docs, client=client)
    assert result is not None
    assert client.query.await_args_list[-1].args[1].lstrip().startswith(
        "You are a technology news editor"
    )
