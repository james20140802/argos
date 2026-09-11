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
    resolve_event_chains,
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


def _batch_fetcher(chain: dict[uuid.UUID, uuid.UUID | None], calls: list[set]):
    """배치 조회를 흉내낸다. 호출마다 요청 id 집합을 `calls`에 기록한다."""

    async def _fetch(event_ids):
        calls.append(set(event_ids))
        return {event_id: chain.get(event_id) for event_id in event_ids}

    return _fetch


@pytest.mark.asyncio
class TestResolveEventChains:
    """배치 판 — 왕복이 사건 수가 아니라 체인 깊이를 따라야 한다."""

    async def test_many_live_events_take_one_round_trip(self):
        # 기간 전체 재군집은 사건이 수백 개일 수 있다. 사건당 한 번씩 물으면
        # 그래프 계산 전에 직렬 왕복만 수백 번이다.
        ids = _ids(200)
        calls: list[set] = []
        chain = {event_id: None for event_id in ids}

        resolved = await resolve_event_chains(ids, _batch_fetcher(chain, calls))

        assert resolved == {event_id: event_id for event_id in ids}
        assert len(calls) == 1

    async def test_round_trips_follow_chain_depth_not_event_count(self):
        # 생존자가 물어본 집합 **밖**에 있을 때만 라운드가 늘어난다. 그래도
        # 늘어나는 축은 깊이지 사건 수가 아니다.
        a, b, c, *rest = _ids(50)
        calls: list[set] = []
        chain: dict[uuid.UUID, uuid.UUID | None] = {a: b, b: c, c: None}
        chain.update({event_id: None for event_id in rest})
        asked = [a, *rest]  # b·c는 물어본 적 없는 흡수 사건이다

        resolved = await resolve_event_chains(asked, _batch_fetcher(chain, calls))

        assert resolved[a] == c
        assert all(resolved[event_id] == event_id for event_id in rest)
        # a → b → c 를 따라가느라 3번. 함께 물어본 살아 있는 사건 47개는
        # 첫 라운드에 같이 실려 가므로 왕복을 늘리지 않는다.
        assert len(calls) == 3

    async def test_chain_members_already_asked_about_cost_no_extra_round_trip(self):
        # 재군집이 실제로 겪는 모양: 기간 안 문서가 흡수 사건과 생존 사건을
        # 둘 다 참조한다. 둘 다 첫 라운드에 실려 오므로 한 번이면 끝난다.
        a, b, c = _ids(3)
        calls: list[set] = []
        chain: dict[uuid.UUID, uuid.UUID | None] = {a: b, b: c, c: None}

        resolved = await resolve_event_chains(chain.keys(), _batch_fetcher(chain, calls))

        assert resolved == {a: c, b: c, c: c}
        assert len(calls) == 1

    async def test_agrees_with_the_one_at_a_time_resolver(self):
        # 배치가 빨라지는 대신 **다른 답**을 내면 안 된다. 순환·한도 초과처럼
        # 코어가 예외 대신 "마지막 도달 id"를 돌려주는 경우까지 맞춰 본다.
        ids = _ids(MAX_MERGE_HOPS + 4)
        long_chain = {ids[i]: ids[i + 1] for i in range(len(ids) - 1)}
        cycle_a, cycle_b = _ids(2)
        cases = [
            {},
            {ids[0]: None},
            long_chain | {ids[-1]: None},
            {cycle_a: cycle_b, cycle_b: cycle_a},
        ]
        for chain in cases:
            expected = {
                event_id: await resolve_event_chain(event_id, _fetcher(chain))
                for event_id in chain
            }
            assert await resolve_event_chains(chain.keys(), _batch_fetcher(chain, [])) == expected

    async def test_no_ids_asks_nothing(self):
        calls: list[set] = []
        assert await resolve_event_chains([], _batch_fetcher({}, calls)) == {}
        assert calls == []
