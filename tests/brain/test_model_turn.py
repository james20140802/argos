"""프로세스 간 모델 사용 순번 (ARG-334).

실제 flock을 실제 자식 프로세스와 겹쳐 쓴다 — flock을 목으로 바꾸지 않는다.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from argos.brain import model_turn as mt
from argos.brain.model_turn import (
    LOCK_ENV,
    TurnInfo,
    lock_path,
    model_turn,
    model_turn_sync,
    turn_is_held,
)

POLL = 0.01


@pytest.fixture
def lock_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "turn.lock"
    monkeypatch.setenv(LOCK_ENV, str(path))
    return path


_HOLDER = """
import fcntl, sys, time
f = open(sys.argv[1], "a")
fcntl.flock(f, fcntl.LOCK_EX)
print("ready", flush=True)
time.sleep(60)
"""


@pytest.fixture
def spawn_holder(lock_file):
    """다른 프로세스가 순번을 쥔 상태를 만든다. 정리 때 남은 자식을 죽인다."""
    procs: list[subprocess.Popen] = []

    def spawn() -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(lock_file)],
            stdout=subprocess.PIPE,
            text=True,
        )
        procs.append(proc)
        assert proc.stdout.readline().strip() == "ready"
        return proc

    yield spawn
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()


def test_lock_path_honors_env_then_xdg_state_home(monkeypatch, tmp_path):
    monkeypatch.setenv(LOCK_ENV, str(tmp_path / "x.lock"))
    assert lock_path() == tmp_path / "x.lock"

    monkeypatch.delenv(LOCK_ENV)
    monkeypatch.setenv("XDG_STATE_HOME", "/x")
    assert lock_path() == Path("/x/argos/model_turn.lock")

    monkeypatch.delenv("XDG_STATE_HOME")
    assert lock_path() == Path.home() / ".local" / "state" / "argos" / "model_turn.lock"


async def test_free_turn_is_taken_without_waiting(lock_file):
    waits: list[int] = []
    acquired: list[TurnInfo] = []
    async with model_turn(
        "t",
        on_wait=lambda: waits.append(1),
        on_acquired=acquired.append,
        poll_seconds=POLL,
    ) as info:
        assert info.waited is False
        assert info.nested is False
        assert info.waited_seconds == 0.0
    assert waits == []
    assert acquired == [info]


async def test_two_tasks_in_one_process_take_turns(lock_file):
    inside = 0
    overlap = 0
    a_in = asyncio.Event()
    a_release = asyncio.Event()
    waits: list[int] = []
    b_info: list[TurnInfo] = []

    async def hold_a():
        nonlocal inside, overlap
        async with model_turn("a", poll_seconds=POLL):
            inside += 1
            overlap = max(overlap, inside)
            a_in.set()
            await a_release.wait()
            inside -= 1

    async def take_b():
        nonlocal inside, overlap
        async with model_turn(
            "b", on_wait=lambda: waits.append(1), poll_seconds=POLL
        ) as info:
            inside += 1
            overlap = max(overlap, inside)
            b_info.append(info)
            inside -= 1

    task_a = asyncio.create_task(hold_a())
    await asyncio.wait_for(a_in.wait(), 10)
    task_b = asyncio.create_task(take_b())

    for _ in range(20):  # B가 시도하고 막힐 시간을 준다 (고정 시간 단언 없음)
        await asyncio.sleep(POLL)
    assert not b_info
    assert waits == [1]

    a_release.set()
    await asyncio.wait_for(asyncio.gather(task_a, task_b), 10)
    assert b_info[0].waited is True
    assert b_info[0].waited_seconds > 0
    assert waits == [1]
    assert overlap == 1


async def test_waiting_does_not_block_event_loop(lock_file):
    a_in = asyncio.Event()
    a_release = asyncio.Event()
    ticks = 0
    b_waiting = asyncio.Event()

    async def hold_a():
        async with model_turn("a", poll_seconds=POLL):
            a_in.set()
            await a_release.wait()

    async def take_b():
        async with model_turn("b", on_wait=b_waiting.set, poll_seconds=POLL):
            pass

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    task_a = asyncio.create_task(hold_a())
    await asyncio.wait_for(a_in.wait(), 10)
    task_b = asyncio.create_task(take_b())
    await asyncio.wait_for(b_waiting.wait(), 10)

    tick_task = asyncio.create_task(ticker())
    before = ticks
    await asyncio.sleep(0.1)
    assert ticks > before + 10
    tick_task.cancel()

    a_release.set()
    await asyncio.wait_for(asyncio.gather(task_a, task_b), 10)


async def test_nested_turn_in_same_context_does_not_deadlock(lock_file):
    async def body():
        async with model_turn("outer", poll_seconds=POLL):
            async with model_turn("inner", poll_seconds=POLL) as inner:
                assert inner.nested is True
            # 안쪽이 끝나도 바깥이 여전히 쥐고 있다 — 순번 밖 컨텍스트에서 본다.
            held = await asyncio.get_running_loop().run_in_executor(None, turn_is_held)
            assert held is True

    await asyncio.wait_for(body(), 1)
    assert turn_is_held() is False


async def test_turn_released_on_exception_and_cancel(lock_file):
    with pytest.raises(RuntimeError):
        async with model_turn("boom", poll_seconds=POLL):
            raise RuntimeError("x")
    assert turn_is_held() is False

    entered = asyncio.Event()

    async def hold():
        async with model_turn("cancelled", poll_seconds=POLL):
            entered.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(hold())
    await asyncio.wait_for(entered.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with model_turn("next", poll_seconds=POLL) as info:
        assert info.waited is False


async def test_cancel_while_waiting_leaves_nothing_held(lock_file, spawn_holder):
    holder = spawn_holder()
    waiting = asyncio.Event()

    async def wait_turn():
        async with model_turn("w", on_wait=waiting.set, poll_seconds=POLL):
            pass

    task = asyncio.create_task(wait_turn())
    await asyncio.wait_for(waiting.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    holder.kill()
    holder.wait()
    assert turn_is_held() is False


async def test_turn_released_when_holder_process_dies(lock_file, spawn_holder):
    holder = spawn_holder()
    assert turn_is_held() is True

    waiting = asyncio.Event()

    async def take():
        async with model_turn("p", on_wait=waiting.set, poll_seconds=POLL) as info:
            return info

    task = asyncio.create_task(take())
    await asyncio.wait_for(waiting.wait(), 10)
    assert not task.done()

    holder.kill()
    info = await asyncio.wait_for(task, 10)
    assert info.waited is True


def test_sync_variant_waits_for_other_process_then_runs(lock_file, spawn_holder):
    holder = spawn_holder()
    killer = threading.Timer(0.2, holder.kill)
    killer.start()
    waits: list[int] = []
    try:
        with model_turn_sync(
            "cli", on_wait=lambda: waits.append(1), poll_seconds=POLL
        ) as info:
            assert info.waited is True
            assert waits == [1]
    finally:
        killer.cancel()
        killer.join()


def test_asyncio_run_inside_sync_turn_is_nested(lock_file):
    async def inner():
        async with model_turn("x", poll_seconds=POLL) as i:
            return i.nested

    with model_turn_sync("cli", poll_seconds=POLL):
        assert asyncio.run(asyncio.wait_for(inner(), 1)) is True
        assert turn_is_held() is True
    assert turn_is_held() is False


async def test_wait_and_acquire_are_logged_once_each(lock_file, spawn_holder, caplog):
    holder = spawn_holder()
    waiting = asyncio.Event()

    async def take():
        async with model_turn("logged", on_wait=waiting.set, poll_seconds=POLL):
            pass

    with caplog.at_level(logging.INFO, logger=mt.logger.name):
        task = asyncio.create_task(take())
        await asyncio.wait_for(waiting.wait(), 10)
        await asyncio.sleep(0.05)
        holder.kill()
        await asyncio.wait_for(task, 10)

    messages = [r.getMessage() for r in caplog.records]
    assert sum("logged waiting" in m for m in messages) == 1
    assert sum("logged acquired after" in m for m in messages) == 1


async def test_free_turn_logs_nothing(lock_file, caplog):
    with caplog.at_level(logging.INFO, logger=mt.logger.name):
        async with model_turn("quiet", poll_seconds=POLL):
            pass
    assert caplog.records == []


def test_tests_never_use_the_real_turn_lock(tmp_path):
    real = Path.home() / ".local" / "state" / "argos" / "model_turn.lock"
    assert lock_path() != real
    assert tmp_path in lock_path().parents


def test_real_turn_lock_path_is_refused_under_pytest(monkeypatch, tmp_path):
    monkeypatch.delenv(LOCK_ENV)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    real = Path.home() / ".local" / "state" / "argos" / "model_turn.lock"
    existed = real.exists()

    with pytest.raises(pytest.fail.Exception):
        with model_turn_sync("x", poll_seconds=POLL):
            pass

    assert real.exists() == existed


def test_flock_probe_does_not_leave_a_lock(lock_file):
    assert turn_is_held() is False
    other = open(lock_file, "a")
    try:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)  # 탐침이 남긴 잠금이 없어야 잡힌다
    finally:
        other.close()
