"""툼스톤 체인 해석 테스트 — Postgres 없이 dict 기반 가짜 조회로 돈다."""
from __future__ import annotations

import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from argos.services.event_resolution import (
    MAX_MERGE_HOPS,
    resolve_event,
    resolve_event_chain,
)


def _ids(count: int) -> list[uuid.UUID]:
    return [uuid.UUID(int=n) for n in range(1, count + 1)]


def _fetcher(chain: dict[uuid.UUID, uuid.UUID | None]):
    """'id → merged_into_id' 조회를 흉내내는 주입용 async 함수."""

    async def _fetch(event_id: uuid.UUID) -> uuid.UUID | None:
        return chain.get(event_id)

    return _fetch


@pytest.mark.asyncio
class TestResolveEventChain:
    async def test_live_event_resolves_to_itself(self):
        (alone,) = _ids(1)

        assert await resolve_event_chain(alone, _fetcher({alone: None})) == alone

    async def test_absorbed_event_resolves_to_its_survivor(self):
        absorbed, survivor = _ids(2)
        chain = {absorbed: survivor, survivor: None}

        assert await resolve_event_chain(absorbed, _fetcher(chain)) == survivor

    async def test_multi_step_chain_reaches_the_final_survivor(self):
        a, b, c, d = _ids(4)
        chain = {a: b, b: c, c: d, d: None}

        assert await resolve_event_chain(a, _fetcher(chain)) == d

    async def test_cycle_stops_without_raising(self):
        a, b = _ids(2)
        # a → b → a → ...; correct cycle detection stops after 2 calls at b,
        # distinct from the 8-call hop-limit backstop.
        fetch = AsyncMock(side_effect=[b, a])

        result = await resolve_event_chain(a, fetch)

        assert result == b
        assert fetch.await_count == 2

    async def test_self_referential_cycle_stops_without_raising(self):
        (a,) = _ids(1)
        # a → a; correct cycle detection stops after a single call.
        fetch = AsyncMock(side_effect=[a])

        result = await resolve_event_chain(a, fetch)

        assert result == a
        assert fetch.await_count == 1

    async def test_cycle_logs_a_warning(self, caplog):
        a, b = _ids(2)
        with caplog.at_level(logging.WARNING):
            await resolve_event_chain(a, _fetcher({a: b, b: a}))

        # Must be the cycle-specific warning, not merely any WARNING record
        # (the post-loop hop-limit warning would also satisfy a looser check).
        assert any(
            "cycles back to" in record.getMessage() for record in caplog.records
        )

    async def test_chain_longer_than_the_hop_limit_stops_and_warns(self, caplog):
        ids = _ids(MAX_MERGE_HOPS + 5)
        chain = {ids[i]: ids[i + 1] for i in range(len(ids) - 1)}
        chain[ids[-1]] = None

        with caplog.at_level(logging.WARNING):
            result = await resolve_event_chain(ids[0], _fetcher(chain))

        # 예외 대신 마지막으로 도달한 id를 돌려준다 (A6) — 피드를 죽이면 안 된다.
        assert result == ids[MAX_MERGE_HOPS]
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    async def test_hop_limit_is_eight_by_default(self):
        assert MAX_MERGE_HOPS == 8

    async def test_stops_calling_the_fetcher_once_the_chain_ends(self):
        absorbed, survivor = _ids(2)
        fetch = AsyncMock(side_effect=[survivor, None])

        assert await resolve_event_chain(absorbed, fetch) == survivor
        assert fetch.await_count == 2


@pytest.mark.asyncio
class TestResolveEvent:
    """얇은 래퍼 — 세션에서 merged_into_id를 읽어 코어에 넘기기만 한다."""

    async def test_walks_the_chain_using_the_session(self):
        absorbed, survivor = _ids(2)
        session = AsyncMock()
        session.scalar = AsyncMock(side_effect=[survivor, None])

        assert await resolve_event(session, absorbed) == survivor
        assert session.scalar.await_count == 2

    async def test_live_event_needs_one_lookup(self):
        (alone,) = _ids(1)
        session = AsyncMock()
        session.scalar = AsyncMock(return_value=None)

        assert await resolve_event(session, alone) == alone
        assert session.scalar.await_count == 1
