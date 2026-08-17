"""사건 근거 문서 조회 테스트 — Postgres 없이 돈다."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from argos.services.events import build_evidence_documents_query, list_evidence_documents

_EVENT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _compile(stmt):
    """실제 방언으로 컴파일한다.

    ``literal_binds``를 쓰면 안 된다 — postgresql UUID가 하이픈 없는 문자열
    (``'11111111111111111111111111111111'``)로 렌더링돼서 ``str(uuid)``와 절대
    일치하지 않는다. 바인드 값은 ``.params``로 따로 본다.
    """
    return stmt.compile(dialect=postgresql.dialect())


class TestBuildEvidenceDocumentsQuery:
    """쿼리 조립부는 세션 없이 검증한다."""

    def test_selects_tech_items_joined_through_link_table(self):
        sql = str(_compile(build_evidence_documents_query(_EVENT_ID)))
        assert "tech_items" in sql
        assert "event_documents" in sql

    def test_filters_by_the_requested_event(self):
        compiled = _compile(build_evidence_documents_query(_EVENT_ID))
        assert _EVENT_ID in compiled.params.values()

    def test_does_not_join_unrelated_tables(self):
        sql = str(_compile(build_evidence_documents_query(_EVENT_ID)))
        assert "feed_events" not in sql
        assert "user_assets" not in sql


@pytest.mark.asyncio
class TestListEvidenceDocuments:
    """async 실행부는 로직 없이 세션에 위임만 한다."""

    async def test_returns_documents_from_the_session(self):
        doc_a, doc_b = object(), object()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [doc_a, doc_b]
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)

        documents = await list_evidence_documents(session, _EVENT_ID)

        assert documents == [doc_a, doc_b]
        session.execute.assert_awaited_once()

    async def test_returns_empty_list_when_event_has_no_evidence(self):
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)

        assert await list_evidence_documents(session, _EVENT_ID) == []
