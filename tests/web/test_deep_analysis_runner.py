"""DeepAnalysisRunner(ARG-336) — DB·모델 없이."""
from __future__ import annotations

import asyncio
import contextlib
import uuid

import pytest

from argos.brain import event_deep_analysis as eda
from argos.brain import model_turn as mt
from argos.brain.event_deep_analysis import DeepAnalysis, DeepAnalysisDoc, DeepAnalysisSection
from argos.web import deep_analysis_runner as runner_mod
from argos.web.deep_analysis_runner import DeepAnalysisRunner

HEADINGS = ("핵심 기술·접근", "성숙도", "생태계와의 관계", "위험·한계", "추적할 만한가")


def _analysis() -> DeepAnalysis:
    return DeepAnalysis(
        sections=tuple(DeepAnalysisSection(h, "b") for h in HEADINGS),
        doc_ids=(uuid.uuid4(),),
        doc_count=1,
        model="m",
    )


class _Calls:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.open_sessions = 0


@pytest.fixture
def calls(monkeypatch) -> _Calls:
    c = _Calls()

    @contextlib.asynccontextmanager
    async def factory():
        c.open_sessions += 1
        try:
            yield object()
        finally:
            c.open_sessions -= 1

    async def fake_mark_running(session, event_id):
        c.log.append("mark_running")

    async def fake_fetch_docs(session, event_id):
        c.log.append("fetch_docs")
        return [DeepAnalysisDoc(id=uuid.uuid4(), title="t")]

    async def fake_save(session, event_id, analysis):
        c.log.append("save_result")

    async def fake_mark_failed(session, event_id, code):
        c.log.append(f"mark_failed:{code}")

    monkeypatch.setattr(runner_mod, "mark_running", fake_mark_running)
    monkeypatch.setattr(runner_mod, "fetch_event_docs", fake_fetch_docs)
    monkeypatch.setattr(runner_mod, "save_result", fake_save)
    monkeypatch.setattr(runner_mod, "mark_failed", fake_mark_failed)
    c.factory = factory  # type: ignore[attr-defined]
    return c


@pytest.mark.asyncio
async def test_run_saves_result_inside_model_turn(calls) -> None:
    held: list[bool] = []
    sessions_during: list[int] = []

    async def generate(docs):
        held.append(mt._HELD.get())
        sessions_during.append(calls.open_sessions)
        return _analysis()

    runner = DeepAnalysisRunner(session_factory=calls.factory, generate=generate)
    await runner.run(uuid.uuid4())

    assert held == [True]
    assert sessions_during == [0]  # 모델 호출 중엔 DB 세션을 쥐지 않는다
    assert calls.log.count("save_result") == 1


@pytest.mark.asyncio
async def test_run_unexpected_error_records_unknown(calls) -> None:
    async def generate(docs):
        raise RuntimeError("boom")

    runner = DeepAnalysisRunner(session_factory=calls.factory, generate=generate)
    await runner.run(uuid.uuid4())  # 던지지 않는다

    assert calls.log.count("mark_failed:Unknown") == 1
    assert calls.log.count("save_result") == 0


@pytest.mark.asyncio
async def test_run_failure_records_code(calls) -> None:
    async def generate(docs):
        raise eda.DeepAnalysisError(eda.DeepAnalysisFailure.Timeout, "slow")

    runner = DeepAnalysisRunner(session_factory=calls.factory, generate=generate)
    await runner.run(uuid.uuid4())

    assert calls.log.count("mark_failed:Timeout") == 1


@pytest.mark.asyncio
async def test_default_generate_unloads_model(monkeypatch) -> None:
    seen: dict = {}

    async def fake_generate(docs, **kwargs):
        seen.update(kwargs)
        return _analysis()

    monkeypatch.setattr(runner_mod, "generate_deep_analysis", fake_generate)
    runner = DeepAnalysisRunner()
    await runner._generate([])
    assert seen["keep_alive"] == 0


@pytest.mark.asyncio
async def test_start_keeps_task_reference_until_done(calls) -> None:
    gate = asyncio.Event()

    async def generate(docs):
        await gate.wait()
        return _analysis()

    runner = DeepAnalysisRunner(session_factory=calls.factory, generate=generate)
    task = runner.start(uuid.uuid4())
    assert len(runner._tasks) == 1
    gate.set()
    await task
    await asyncio.sleep(0)
    assert len(runner._tasks) == 0


def test_eda_module_exposes_generate() -> None:
    assert hasattr(eda, "generate_deep_analysis")


@pytest.mark.asyncio
async def test_run_marks_waiting_then_running_when_turn_held(calls, monkeypatch) -> None:
    monkeypatch.setattr(mt, "DEFAULT_POLL_SECONDS", 0.01)

    async def fake_mark_waiting(session, event_id):
        calls.log.append("mark_waiting")

    monkeypatch.setattr(runner_mod, "mark_waiting", fake_mark_waiting)

    async def generate(docs):
        return _analysis()

    other = mt._try_acquire()  # 다른 fd로 순번을 쥔 "다른 작업"
    assert other is not None
    runner = DeepAnalysisRunner(session_factory=calls.factory, generate=generate)
    task = asyncio.create_task(runner.run(uuid.uuid4()))
    try:
        for _ in range(200):
            if "mark_waiting" in calls.log:
                break
            await asyncio.sleep(0.01)
        assert calls.log == ["mark_waiting"]
        assert not task.done()
    finally:
        mt._release(other)
    await asyncio.wait_for(task, 5)
    assert calls.log == ["mark_waiting", "mark_running", "fetch_docs", "save_result"]
