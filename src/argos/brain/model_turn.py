"""프로세스 간 모델 사용 순번 — ARG-334.

여러 OS 프로세스(예약된 ``argos run``, ``correct-events``, ``taxonomy assign``,
수동 backfill, 웹 서버)가 각자 큰 글 생성 모델을 올린다. 32GB 맥에서 둘이 겹치면
메모리가 터지므로, "모델 순번"이라는 **작업 단위** 잠금 하나를 파일 잠금(flock)으로
프로세스 간에 공유한다. 명령 시작부터 끝까지(확인 프롬프트 대기 포함) 쥔다.

``ollama_client._MODEL_LOCK``(프로세스 안, 질의 단위)과는 별개다 — 그쪽은 그대로 둔다.

규약:
- ``LOCK_EX | LOCK_NB``를 ``poll_seconds`` 간격으로 재시도한다. 블로킹 flock도 스레드도
  쓰지 않아 이벤트 루프를 막지 않는다. 대기에는 기한이 없다.
- 잠금 파일은 지우지 않는다(지우면 서로 다른 inode를 잠그는 경쟁이 생긴다).
  프로세스가 죽으면 OS가 잠금을 풀어 준다.
- 재진입: 이 흐름(ContextVar)이 이미 쥐고 있으면 flock을 다시 시도하지 않고 통과한다.
  ``asyncio.run``/``create_task``가 컨텍스트를 복사하므로 순번을 쥔 CLI 명령 안의
  코루틴은 자동으로 통과하고, 순번 밖에서 만든 다른 태스크는 새 fd로 기다린다.
"""

from __future__ import annotations

import asyncio
import contextvars
import fcntl
import logging
import os
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

LOCK_ENV = "ARGOS_MODEL_TURN_LOCK"
"""잠금 파일 경로 override (테스트·운영 공용)."""
WAIT_MESSAGE = "다른 작업이 모델 사용 중 — 대기"
DEFAULT_POLL_SECONDS = 1.0

_HELD: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "argos_model_turn_held", default=False
)


def lock_path() -> Path:
    """잠금 파일 경로. ``pair_check.default_cache_path``와 같은 XDG 상태 경로 규칙."""
    override = os.environ.get(LOCK_ENV)
    if override:
        return Path(override)
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "argos" / "model_turn.lock"


@dataclass
class TurnInfo:
    job: str
    waited: bool = False
    """한 번이라도 기다렸는가."""
    waited_seconds: float = 0.0
    """잡기까지 걸린 시간."""
    nested: bool = False
    """같은 흐름에서 이미 쥐고 있어 그냥 통과했는가."""


def _open_lock(path: Path) -> IO[str]:
    """잠금 파일을 연다(없으면 부모 디렉터리와 함께 만든다). 테스트 가드의 훅 자리."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a")  # noqa: SIM115 — 호출자가 닫는다


def _try_acquire() -> IO[str] | None:
    """한 번 시도한다. 잡으면 열린 파일 핸들, 막혔으면 None."""
    handle = _open_lock(lock_path())
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except BaseException:
        handle.close()
        raise
    return handle


def _release(handle: IO[str]) -> None:
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def turn_is_held() -> bool:
    """지금 다른 누군가(다른 프로세스 또는 이 프로세스의 다른 흐름)가 쥐고 있는가.

    non-blocking 탐침이라 잠금을 남기지 않는다. 탐침이 잡는 찰나에 들어온 요청은
    한 번 더 기다릴 뿐 틀리게 통과하지는 않는다.
    """
    handle = _try_acquire()
    if handle is None:
        return True
    _release(handle)
    return False


def _log_acquired(job: str, info: TurnInfo) -> None:
    if info.waited:
        logger.info("model turn: %s acquired after %.0fs wait", job, info.waited_seconds)


def _reset_held(token: contextvars.Token[bool]) -> None:
    try:
        _HELD.reset(token)
    except ValueError:  # 다른 컨텍스트에서 닫힌 경우 — 표시만 내린다
        _HELD.set(False)


@asynccontextmanager
async def model_turn(
    job: str,
    *,
    on_wait: Callable[[], None] | None = None,
    on_acquired: Callable[[TurnInfo], None] | None = None,
    poll_seconds: float | None = None,
) -> AsyncIterator[TurnInfo]:
    """순번을 잡고 본문을 실행한다. 이미 이 흐름이 쥐고 있으면 그냥 통과한다.

    ``poll_seconds=None``은 호출 시점의 ``DEFAULT_POLL_SECONDS``(테스트가 바꿀 수 있다).
    """
    if _HELD.get():
        info = TurnInfo(job=job, nested=True)
        if on_acquired is not None:
            on_acquired(info)
        yield info
        return

    poll = DEFAULT_POLL_SECONDS if poll_seconds is None else poll_seconds
    started = time.monotonic()
    handle = _try_acquire()
    waited = handle is None
    if waited:
        logger.info("model turn: %s waiting — another job holds it", job)
        if on_wait is not None:
            on_wait()
        while handle is None:
            await asyncio.sleep(poll)
            handle = _try_acquire()

    info = TurnInfo(
        job=job,
        waited=waited,
        waited_seconds=time.monotonic() - started if waited else 0.0,
    )
    token = _HELD.set(True)
    try:
        _log_acquired(job, info)
        if on_acquired is not None:
            on_acquired(info)
        yield info
    finally:
        _reset_held(token)
        _release(handle)


@contextmanager
def model_turn_sync(
    job: str,
    *,
    on_wait: Callable[[], None] | None = None,
    on_acquired: Callable[[TurnInfo], None] | None = None,
    poll_seconds: float | None = None,
) -> Iterator[TurnInfo]:
    """``model_turn``의 동기판 — CLI ``main()``이 명령 전체를 감쌀 때 쓴다."""
    if _HELD.get():
        info = TurnInfo(job=job, nested=True)
        if on_acquired is not None:
            on_acquired(info)
        yield info
        return

    poll = DEFAULT_POLL_SECONDS if poll_seconds is None else poll_seconds
    started = time.monotonic()
    handle = _try_acquire()
    waited = handle is None
    if waited:
        logger.info("model turn: %s waiting — another job holds it", job)
        if on_wait is not None:
            on_wait()
        while handle is None:
            time.sleep(poll)
            handle = _try_acquire()

    info = TurnInfo(
        job=job,
        waited=waited,
        waited_seconds=time.monotonic() - started if waited else 0.0,
    )
    token = _HELD.set(True)
    try:
        _log_acquired(job, info)
        if on_acquired is not None:
            on_acquired(info)
        yield info
    finally:
        _reset_held(token)
        _release(handle)
