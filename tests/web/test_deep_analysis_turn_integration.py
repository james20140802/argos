"""웹 실행기와 다른 OS 프로세스가 모델 순번을 실제 flock으로 주고받는다 (ARG-337)."""
from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from argos.brain import model_turn as mt
from argos.brain.event_deep_analysis import DeepAnalysis, DeepAnalysisDoc, DeepAnalysisSection
from argos.web import deep_analysis_runner as runner_mod
from argos.web.deep_analysis_runner import DeepAnalysisRunner

# 자식은 argos.database를 끌고 오는 argos.brain을 import하지 않고, 같은 프로토콜(flock)을 직접 쓴다.
_HOLDER = """
import fcntl, os, sys, time
ready, release = sys.argv[1], sys.argv[2]
f = open(os.environ["ARGOS_MODEL_TURN_LOCK"], "a")
fcntl.flock(f, fcntl.LOCK_EX)
open(ready, "w").close()
while not os.path.exists(release):
    time.sleep(0.02)
"""
_WAITER = """
import fcntl, os, sys
started, acquired = sys.argv[1], sys.argv[2]
f = open(os.environ["ARGOS_MODEL_TURN_LOCK"], "a")
open(started, "w").close()
fcntl.flock(f, fcntl.LOCK_EX)
open(acquired, "w").close()
"""


def _spawn(code: str, *args: Path) -> subprocess.Popen:
    env = {**os.environ, mt.LOCK_ENV: str(mt.lock_path())}
    return subprocess.Popen([sys.executable, "-c", code, *map(str, args)], env=env)


async def _until(pred, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return pred()


def _analysis() -> DeepAnalysis:
    return DeepAnalysis(
        sections=(DeepAnalysisSection("h", "b"),),
        doc_ids=(uuid.uuid4(),),
        doc_count=1,
        model="m",
    )


@pytest.fixture
def harness(monkeypatch):
    """(entries, session_factory) — 실행기가 부른 DB 단계 기록과 가짜 세션 팩토리."""
    entries: list[str] = []

    @contextlib.asynccontextmanager
    async def factory():
        yield object()

    async def rec(name, *a):
        entries.append(name)

    async def fetch(session, event_id):
        return [DeepAnalysisDoc(id=uuid.uuid4(), title="t")]

    monkeypatch.setattr(runner_mod, "mark_waiting", lambda s, e: rec("mark_waiting"))
    monkeypatch.setattr(runner_mod, "mark_running", lambda s, e: rec("mark_running"))
    monkeypatch.setattr(runner_mod, "save_result", lambda s, e, a: rec("save_result"))
    monkeypatch.setattr(runner_mod, "mark_failed", lambda s, e, c: rec("mark_failed"))
    monkeypatch.setattr(runner_mod, "fetch_event_docs", fetch)
    monkeypatch.setattr(mt, "DEFAULT_POLL_SECONDS", 0.02)
    return entries, factory


@pytest.mark.asyncio
async def test_web_run_waits_for_other_process_then_continues(harness, tmp_path) -> None:
    log, session_factory = harness
    ready, release = tmp_path / "ready", tmp_path / "release"
    generated: list[int] = []

    async def generate(docs):
        generated.append(1)
        return _analysis()

    proc = _spawn(_HOLDER, ready, release)
    try:
        assert await _until(ready.exists), "holder never acquired the turn"
        runner = DeepAnalysisRunner(session_factory=session_factory, generate=generate)
        task = asyncio.create_task(runner.run(uuid.uuid4()))
        assert await _until(lambda: "mark_waiting" in log)
        await asyncio.sleep(0.2)
        assert generated == []
        release.touch()
        await asyncio.wait_for(task, 10)
        assert generated == [1]
        assert log == ["mark_waiting", "mark_running", "save_result"]
    finally:
        release.touch()
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.asyncio
async def test_other_process_waits_while_web_run_holds_turn(harness, tmp_path) -> None:
    _, session_factory = harness
    started, acquired = tmp_path / "started", tmp_path / "acquired"
    in_generate = asyncio.Event()
    gate = asyncio.Event()

    async def generate(docs):
        in_generate.set()
        await gate.wait()
        return _analysis()

    runner = DeepAnalysisRunner(session_factory=session_factory, generate=generate)
    task = asyncio.create_task(runner.run(uuid.uuid4()))
    proc = None
    try:
        await asyncio.wait_for(in_generate.wait(), 10)
        proc = _spawn(_WAITER, started, acquired)
        assert await _until(started.exists), "waiter never reached the flock call"
        await asyncio.sleep(0.5)
        assert not acquired.exists()
        assert proc.poll() is None
        gate.set()
        await asyncio.wait_for(task, 10)
        assert await _until(acquired.exists), "other process never got the turn"
    finally:
        gate.set()
        if proc is not None:
            proc.kill()
            proc.wait(timeout=10)
        if not task.done():
            task.cancel()
