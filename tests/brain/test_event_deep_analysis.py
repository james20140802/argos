from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from argos.brain import event_deep_analysis as eda
from argos.brain.event_deep_analysis import (
    DEEP_ANALYSIS_HEADINGS,
    SECTION_KEYS,
    DeepAnalysis,
    DeepAnalysisDoc,
    DeepAnalysisError,
    DeepAnalysisFailure,
    analyze_event,
    fetch_event_docs,
    generate_deep_analysis,
)
from argos.brain.event_detail_schema import required_names
from argos.brain.ollama_client import OllamaInfraError
from argos.config import settings


def _ok(**overrides: str) -> str:
    values = {key: f"{key} 본문이다." for key in SECTION_KEYS}
    values.update(overrides)
    return json.dumps(values, ensure_ascii=False)


class FakeClient:
    """``query`` 호출을 기록하고 준비된 응답(또는 예외)을 순서대로 돌려준다."""

    def __init__(self, *replies, handler=None) -> None:
        self.replies = list(replies)
        self.handler = handler
        self.calls: list[tuple[str, str, dict]] = []

    async def query(self, role, prompt, **kwargs):
        self.calls.append((role, prompt, kwargs))
        if self.handler is not None:
            return self.handler(prompt, kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    @property
    def prompts(self) -> list[str]:
        return [prompt for _, prompt, _ in self.calls]


def _doc(i: int = 1, **kw) -> DeepAnalysisDoc:
    kw.setdefault("title", f"기사 제목 {i}")
    kw.setdefault("body", f"기사 {i} 본문 내용이다.")
    return DeepAnalysisDoc(id=uuid.UUID(int=i), **kw)


@pytest.mark.asyncio
async def test_two_doc_event_returns_five_sections_in_order():
    client = FakeClient(_ok())
    docs = [_doc(1), _doc(2)]
    result = await generate_deep_analysis(docs, client=client)
    assert [s.heading for s in result.sections] == list(DEEP_ANALYSIS_HEADINGS)
    assert [s.body for s in result.sections] == [f"{k} 본문이다." for k in SECTION_KEYS]
    assert result.doc_count == 2
    assert result.doc_ids == (docs[0].id, docs[1].id)
    assert result.model == settings.user.deep_analysis_resolved().model
    as_json = result.sections_json()
    assert len(as_json) == 5
    assert all(set(item) == {"heading", "body"} for item in as_json)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_prompt_carries_every_doc_title_source_and_date():
    client = FakeClient(_ok())
    docs = [
        _doc(1, title="첫 기사", source="https://www.example.com/a",
             published_at=datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)),
        _doc(2, title="둘째 기사", source="https://blog.example.com/b",
             published_at=datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)),
    ]
    await generate_deep_analysis(docs, client=client)
    prompt = client.prompts[0]
    assert "첫 기사" in prompt and "둘째 기사" in prompt
    assert "출처: example.com" in prompt
    assert "날짜: 2026-11-20" in prompt


@pytest.mark.asyncio
async def test_all_docs_reach_the_model_beyond_thirty():
    def handler(prompt, kwargs):
        return _ok() if kwargs["format"] else "[1] 메모"

    client = FakeClient(handler=handler)
    docs = [_doc(i, body=f"고유{i}문자열 " + "가" * 2000) for i in range(1, 36)]
    result = await generate_deep_analysis(docs, client=client)
    joined = "\n".join(client.prompts)
    for i in range(1, 36):
        # 머리글 전체 형태(번호+제목+줄바꿈)와 본문 앞머리 표지를 기사마다 따로 본다.
        # 줄바꿈·구분 공백까지 넣어 1이 10~19에, 표지 1이 표지 10에 걸리지 않게 한다.
        assert f"[{i}] 기사 제목 {i}\n" in joined
        assert f"본문: 고유{i}문자열 " in joined
    assert result.doc_count == 35


@pytest.mark.asyncio
async def test_analyze_event_only_selects_and_never_writes(monkeypatch):
    result = MagicMock()
    result.all.return_value = [
        SimpleNamespace(doc_id=uuid.UUID(int=1), doc_title="a", doc_source=None,
                        doc_published_at=None, doc_body="본문", doc_names=None),
    ]
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    generated = AsyncMock(
        return_value=DeepAnalysis(sections=(), doc_ids=(), doc_count=1, model="m")
    )
    monkeypatch.setattr(eda, "generate_deep_analysis", generated)

    await analyze_event(session, uuid.uuid4(), client=FakeClient(_ok()))

    assert session.execute.await_count == 1
    sql = str(session.execute.await_args.args[0]).strip().upper()
    assert sql.startswith("SELECT")
    for verb in ("INSERT", "UPDATE", "DELETE"):
        assert verb not in sql
    session.commit.assert_not_called()
    session.flush.assert_not_called()
    session.add.assert_not_called()
    generated.assert_awaited_once()


@pytest.fixture
def small_ctx(monkeypatch):
    monkeypatch.setattr(settings.user.event_summary, "num_ctx", 4096)
    monkeypatch.setattr(settings.user.event_summary, "output_reserve_tok", 512)


@pytest.mark.asyncio
async def test_huge_event_folds_into_memos_then_analyzes(small_ctx):
    def handler(prompt, kwargs):
        return _ok() if kwargs["format"] else "[1] 사실 메모"

    client = FakeClient(handler=handler)
    docs = [_doc(i, body="나" * 1500) for i in range(1, 81)]
    result = await generate_deep_analysis(docs, client=client)
    unstructured = [c for c in client.calls if not c[2]["format"]]
    assert len(unstructured) >= 2
    assert client.calls[-1][2]["format"] is not None
    assert all(not c[2]["format"] for c in client.calls[:-1])
    assert result.doc_count == 80


@pytest.mark.asyncio
async def test_fold_that_never_converges_is_too_large(small_ctx, monkeypatch):
    monkeypatch.setattr(eda, "MAX_FOLD_DEPTH", 1)
    client = FakeClient("가" * 100_000)
    docs = [_doc(i, body="나" * 1500) for i in range(1, 81)]
    with pytest.raises(DeepAnalysisError) as exc:
        await generate_deep_analysis(docs, client=client)
    assert exc.value.reason is DeepAnalysisFailure.TooLarge


@pytest.mark.asyncio
async def test_empty_memo_is_bad_format(small_ctx):
    client = FakeClient("   ")
    docs = [_doc(i, body="나" * 1500) for i in range(1, 81)]
    with pytest.raises(DeepAnalysisError) as exc:
        await generate_deep_analysis(docs, client=client)
    assert exc.value.reason is DeepAnalysisFailure.BadFormat


@pytest.mark.asyncio
async def test_empty_bodies_are_marked_and_counted():
    client = FakeClient(_ok())
    docs = [_doc(1, body=None), _doc(2, body="  ")]
    result = await generate_deep_analysis(docs, client=client)
    assert client.prompts[0].count("(본문 없음)") == 2
    assert result.doc_count == 2


@pytest.mark.asyncio
async def test_prompt_states_no_invention_and_english_name_rules():
    client = FakeClient(_ok())
    await generate_deep_analysis([_doc(1)], client=client)
    prompt = client.prompts[0]
    assert "근거 부족" in prompt
    assert "기사에 없는 내용은 지어내지 않는다" in prompt
    assert "회사·제품·모델 이름은 번역·음차하지 말고 영어 원문 그대로 쓴다" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "docs",
    [[], [DeepAnalysisDoc(id=uuid.uuid4(), title=None, body=None),
          DeepAnalysisDoc(id=uuid.uuid4(), title="  ", body="")]],
)
async def test_no_docs_is_no_evidence(docs):
    client = FakeClient(_ok())
    with pytest.raises(DeepAnalysisError) as exc:
        await generate_deep_analysis(docs, client=client)
    assert exc.value.reason is DeepAnalysisFailure.NoEvidence
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        json.dumps({k: "x" for k in SECTION_KEYS[:-1]}),
        _ok(risks="   "),
        "분석 결과입니다",
        json.dumps(["a", "b"]),
    ],
)
async def test_missing_or_blank_section_fails_bad_format(bad):
    client = FakeClient(bad)
    with pytest.raises(DeepAnalysisError) as exc:
        await generate_deep_analysis([_doc(1)], client=client)
    assert exc.value.reason is DeepAnalysisFailure.BadFormat
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_second_attempt_recovers_from_bad_first_output():
    client = FakeClient("깨짐", _ok())
    result = await generate_deep_analysis([_doc(1)], client=client)
    assert len(result.sections) == 5
    assert len(client.calls) == 2


def _duet_docs() -> list[DeepAnalysisDoc]:
    return [
        _doc(1, title="Claude Duet 공개", body="Anthropic shipped Claude Duet today.",
             names=("Claude Duet",)),
        _doc(2, title="Claude Duet 첫인상", body="Early look at Claude Duet.",
             names=("Claude Duet",)),
    ]


@pytest.mark.asyncio
async def test_dropped_required_name_retries_with_note_then_fails():
    docs = _duet_docs()
    evidence = "\n".join(f"[{i}] {d.title}\n본문: {d.body}" for i, d in enumerate(docs, 1))
    assert required_names(
        [d.names for d in docs], evidence, titles=[d.title for d in docs]
    ) == ["Claude Duet"]

    client = FakeClient(_ok(core="클로드 듀엣이 나왔다."))
    with pytest.raises(DeepAnalysisError) as exc:
        await generate_deep_analysis(docs, client=client)
    assert exc.value.reason is DeepAnalysisFailure.BadFormat
    assert len(client.calls) == 2
    assert "Claude Duet" not in client.prompts[0].split("기사들:")[0]
    assert "반드시" in client.prompts[1] and "Claude Duet" in client.prompts[1].split("기사들:")[0]

    client = FakeClient(_ok(core="클로드 듀엣이 나왔다."), _ok(core="Claude Duet이 나왔다."))
    result = await generate_deep_analysis(docs, client=client)
    assert "Claude Duet" in result.sections[0].body


@pytest.mark.asyncio
async def test_timeout_is_classified():
    err = OllamaInfraError("t")
    err.__cause__ = httpx.ReadTimeout("t")
    client = FakeClient(err)
    with pytest.raises(DeepAnalysisError) as exc:
        await generate_deep_analysis([_doc(1)], client=client)
    assert exc.value.reason is DeepAnalysisFailure.Timeout


@pytest.mark.asyncio
async def test_connection_failure_is_model_unavailable():
    try:
        raise httpx.ConnectError("refused")
    except httpx.ConnectError as cause:
        err = OllamaInfraError("down")
        err.__cause__ = cause
    client = FakeClient(err)
    with pytest.raises(DeepAnalysisError) as exc:
        await generate_deep_analysis([_doc(1)], client=client)
    assert exc.value.reason is DeepAnalysisFailure.ModelUnavailable


@pytest.mark.asyncio
async def test_unexpected_exception_is_not_swallowed():
    client = FakeClient(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await generate_deep_analysis([_doc(1)], client=client)


@pytest.mark.asyncio
async def test_query_uses_resolved_role_settings_and_exclusive():
    resolved = settings.user.deep_analysis_resolved()
    client = FakeClient(_ok())
    await generate_deep_analysis([_doc(1)], client=client)
    role, _, kwargs = client.calls[0]
    assert role == "large"
    assert kwargs["exclusive"] is True
    assert kwargs["think"] is False
    assert kwargs["num_ctx"] == resolved.num_ctx
    assert kwargs["timeout"] == resolved.timeout
    assert kwargs["temperature"] == settings.user.event_summary.temperature
    schema = kwargs["format"]
    assert schema["required"] == list(SECTION_KEYS)
    assert set(schema["properties"]) == set(SECTION_KEYS)


@pytest.mark.asyncio
async def test_markdown_in_sections_is_scrubbed():
    client = FakeClient(_ok(core="**굵게** 쓴 문장", risks="```\n코드\n```"))
    result = await generate_deep_analysis([_doc(1)], client=client)
    assert result.sections[0].body == "굵게 쓴 문장"
    assert "```" not in result.sections[3].body


@pytest.mark.asyncio
async def test_fetch_event_docs_binds_event_id_and_has_no_limit():
    event_id = uuid.uuid4()
    rows = [
        SimpleNamespace(doc_id=uuid.UUID(int=1), doc_title="a", doc_source="https://x.com/1",
                        doc_published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        doc_body="본문", doc_names=["Foo", "Bar"]),
        SimpleNamespace(doc_id=uuid.UUID(int=2), doc_title=None, doc_source=None,
                        doc_published_at=None, doc_body=None, doc_names=None),
    ]
    result = MagicMock()
    result.all.return_value = rows
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)

    docs = await fetch_event_docs(session, event_id)

    statement, params = session.execute.await_args.args
    assert "LIMIT" not in str(statement).upper()
    assert params["event_id"] == event_id
    assert [d.id for d in docs] == [uuid.UUID(int=1), uuid.UUID(int=2)]
    assert docs[0].names == ("Foo", "Bar")
    assert docs[1].names == ()
    assert docs[0].body == "본문"


@pytest.mark.asyncio
async def test_analyze_event_passes_all_fetched_docs(monkeypatch):
    docs = [_doc(1), _doc(2), _doc(3)]
    monkeypatch.setattr(eda, "fetch_event_docs", AsyncMock(return_value=docs))
    result = await analyze_event(MagicMock(), uuid.uuid4(), client=FakeClient(_ok()))
    assert result.doc_count == 3
