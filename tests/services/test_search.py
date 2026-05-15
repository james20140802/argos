"""Tests for search_tech_items — pgvector cosine similarity query with filters."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos.services.search import SearchResult, search_tech_items

_TS = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
_EMBEDDING = [0.1] * 768


def _make_row(
    title: str = "Test Item",
    trust_score: float | None = 0.85,
    category: str | None = "Alpha",
    status: str | None = None,
    created_at: datetime = _TS,
) -> SimpleNamespace:
    return SimpleNamespace(
        title=title,
        trust_score=trust_score,
        category=category,
        status=status,
        created_at=created_at,
    )


def _make_session(rows: list) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.fetchall.return_value = rows
    session.execute = AsyncMock(return_value=result)
    return session


@pytest.mark.asyncio
async def test_search_returns_results():
    rows = [
        _make_row("LightRAG v2.0", 0.91, "Alpha", "Keep"),
        _make_row("RAG Fusion", 0.85, "Alpha", None),
    ]
    session = _make_session(rows)

    results = await search_tech_items(session, _EMBEDDING)

    assert len(results) == 2
    assert results[0].title == "LightRAG v2.0"
    assert results[0].trust_score == 0.91
    assert results[0].category == "Alpha"
    assert results[0].status == "Keep"
    assert results[0].created_at == _TS
    assert results[1].title == "RAG Fusion"
    assert results[1].status is None


@pytest.mark.asyncio
async def test_search_result_is_dataclass():
    session = _make_session([_make_row()])
    results = await search_tech_items(session, _EMBEDDING)
    assert isinstance(results[0], SearchResult)


@pytest.mark.asyncio
async def test_search_sql_uses_cosine_operator():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING)

    call_args = session.execute.await_args
    stmt = call_args.args[0]
    sql_str = str(stmt)
    assert "<=>" in sql_str
    assert "tech_items" in sql_str
    assert "user_assets" in sql_str


@pytest.mark.asyncio
async def test_search_sql_passes_embedding_as_cast_vector():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING)

    call_args = session.execute.await_args
    params = call_args.args[1]
    assert "emb" in params
    emb_val = params["emb"]
    assert emb_val.startswith("[") and emb_val.endswith("]")
    floats = [float(x) for x in emb_val[1:-1].split(",")]
    assert len(floats) == 768


@pytest.mark.asyncio
async def test_search_default_limit_10():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING)

    params = session.execute.await_args.args[1]
    assert params["limit"] == 10


@pytest.mark.asyncio
async def test_search_custom_limit():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING, limit=5)

    params = session.execute.await_args.args[1]
    assert params["limit"] == 5


@pytest.mark.asyncio
async def test_search_limit_capped_at_50():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING, limit=100)

    params = session.execute.await_args.args[1]
    assert params["limit"] == 50


@pytest.mark.asyncio
async def test_search_no_category_filter_by_default():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING)

    params = session.execute.await_args.args[1]
    assert "category" not in params


@pytest.mark.asyncio
async def test_search_category_alpha_filter():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING, category="alpha")

    params = session.execute.await_args.args[1]
    assert params["category"] == "Alpha"
    stmt = session.execute.await_args.args[0]
    assert "category" in str(stmt).lower()


@pytest.mark.asyncio
async def test_search_category_mainstream_filter():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING, category="mainstream")

    params = session.execute.await_args.args[1]
    assert params["category"] == "Mainstream"


@pytest.mark.asyncio
async def test_search_category_already_pascal_case():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING, category="Alpha")

    params = session.execute.await_args.args[1]
    assert params["category"] == "Alpha"


@pytest.mark.asyncio
async def test_search_status_all_no_filter():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING, status="all")

    # ua.status appears in SELECT but must NOT appear in the WHERE clause
    stmt_str = str(session.execute.await_args.args[0])
    where_part = stmt_str.upper().split("WHERE", 1)[-1] if "WHERE" in stmt_str.upper() else ""
    assert "UA.STATUS" not in where_part


@pytest.mark.asyncio
async def test_search_status_keep_filter():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING, status="keep")

    stmt_str = str(session.execute.await_args.args[0])
    assert "Keep" in stmt_str


@pytest.mark.asyncio
async def test_search_empty_results():
    session = _make_session([])
    results = await search_tech_items(session, _EMBEDDING)
    assert results == []


@pytest.mark.asyncio
async def test_search_left_outer_join():
    session = _make_session([])
    await search_tech_items(session, _EMBEDDING)

    stmt_str = str(session.execute.await_args.args[0]).upper()
    assert "LEFT" in stmt_str and "JOIN" in stmt_str


@pytest.mark.asyncio
async def test_search_optional_fields_can_be_none():
    row = _make_row(trust_score=None, category=None, status=None)
    session = _make_session([row])
    results = await search_tech_items(session, _EMBEDDING)

    assert results[0].trust_score is None
    assert results[0].category is None
    assert results[0].status is None
