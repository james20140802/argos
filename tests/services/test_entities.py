"""엔티티 이름 정규화 + 사건↔엔티티 양방향 조회 테스트 — Postgres 없이 돈다."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from argos.services.entities import (
    build_event_entities_query,
    build_events_for_entity_query,
    list_event_entities,
    list_events_for_entity,
    normalize_entity_name,
)

_EVENT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class TestNormalizeEntityName:
    """최소 정규화만 한다 — 소문자화 + 공백 정리 (A3). 그 이상은 ARG-225 범위."""

    def test_lowercases(self):
        assert normalize_entity_name("Claude Opus") == "claude opus"

    def test_strips_surrounding_whitespace(self):
        assert normalize_entity_name("  Ollama  ") == "ollama"

    def test_collapses_runs_of_whitespace(self):
        assert normalize_entity_name("Qwen3\t\n  32B") == "qwen3 32b"

    def test_same_name_in_different_casing_collapses_to_one_key(self):
        assert normalize_entity_name("PGVECTOR") == normalize_entity_name("pgvector")

    def test_blank_input_yields_empty_key(self):
        assert normalize_entity_name("   ") == ""

    def test_does_not_touch_hyphens_or_internal_punctuation(self):
        # 하이픈/구두점 변형 통합은 형제 이슈 ARG-225 소관이다.
        assert normalize_entity_name("Sonnet-5") == "sonnet-5"


def _compile(stmt):
    """실제 방언으로 컴파일한다. ``literal_binds``는 UUID를 하이픈 없이
    렌더링하므로 쓰지 않는다 — 바인드 값은 ``.params``로 본다."""
    return stmt.compile(dialect=postgresql.dialect())


class TestQueryBuilders:
    def test_event_entities_query_joins_entities_through_link_table(self):
        compiled = _compile(build_event_entities_query(_EVENT_ID))
        sql = str(compiled)
        assert "entities" in sql
        assert "event_entities" in sql
        assert _EVENT_ID in compiled.params.values()

    def test_events_for_entity_query_filters_on_normalized_key(self):
        compiled = _compile(build_events_for_entity_query("claude opus"))
        assert "tech_events" in str(compiled)
        assert "claude opus" in compiled.params.values()


def _session_returning(rows: list) -> AsyncMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)
    return session


@pytest.mark.asyncio
class TestBidirectionalLookup:
    async def test_lists_names_attached_to_an_event(self):
        entity = object()
        session = _session_returning([entity])

        assert await list_event_entities(session, _EVENT_ID) == [entity]

    async def test_finds_events_by_name_regardless_of_casing(self):
        event = object()
        session = _session_returning([event])

        assert await list_events_for_entity(session, "  CLAUDE   Opus ") == [event]

        executed = session.execute.await_args.args[0]
        assert "claude opus" in _compile(executed).params.values()

    async def test_unknown_name_yields_no_events(self):
        session = _session_returning([])

        assert await list_events_for_entity(session, "nonexistent") == []
