"""Tests for `argos search` CLI subcommand (ARG-100)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


from argos.cli import main
from argos.services.search import SearchResult

_TS = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
_EMBEDDING = [0.1] * 768


def _make_result(**kwargs) -> SearchResult:
    defaults = dict(
        title="LightRAG v2.0",
        trust_score=0.91,
        category="Alpha",
        status="Keep",
        created_at=_TS,
    )
    defaults.update(kwargs)
    return SearchResult(**defaults)


def _make_session_ctx():
    session = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return session, ctx


def _patch_search_stack(results, embed_side_effect=None):
    embed_mock = AsyncMock(return_value=_EMBEDDING)
    if embed_side_effect is not None:
        embed_mock.side_effect = embed_side_effect

    session, session_ctx = _make_session_ctx()
    search_mock = AsyncMock(return_value=results)
    return embed_mock, session, session_ctx, search_mock


# ---------------------------------------------------------------------------
# Basic dispatch — main() calls asyncio.run() internally; must be sync tests
# ---------------------------------------------------------------------------

def test_search_command_returns_zero_on_success(capsys):
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([_make_result()])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        rc = main(["search", "RAG"])

    assert rc == 0


def test_search_calls_embed_with_query():
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([_make_result()])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG"])

    embed_mock.assert_awaited_once_with("RAG")


def test_search_calls_search_tech_items_with_embedding():
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([_make_result()])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG"])

    assert search_mock.await_args.args[1] == _EMBEDDING


# ---------------------------------------------------------------------------
# CLI flags pass-through
# ---------------------------------------------------------------------------

def test_search_default_limit_10():
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG"])

    assert search_mock.await_args.kwargs["limit"] == 10


def test_search_custom_limit():
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG", "--limit", "5"])

    assert search_mock.await_args.kwargs["limit"] == 5


def test_search_negative_limit_rejected_by_cli():
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main(["search", "RAG", "--limit", "-1"])
    assert exc_info.value.code != 0


def test_search_zero_limit_rejected_by_cli():
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main(["search", "RAG", "--limit", "0"])
    assert exc_info.value.code != 0


def test_search_category_flag():
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG", "--category", "alpha"])

    assert search_mock.await_args.kwargs["category"] == "alpha"


def test_search_status_keep_flag():
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG", "--status", "keep"])

    assert search_mock.await_args.kwargs["status"] == "keep"


def test_search_default_status_is_all():
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG"])

    assert search_mock.await_args.kwargs["status"] == "all"


def test_search_default_category_is_none():
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG"])

    assert search_mock.await_args.kwargs["category"] is None


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------

def test_search_empty_results_prints_korean_message(capsys):
    embed_mock, session, session_ctx, search_mock = _patch_search_stack([])

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        rc = main(["search", "no-match-query"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "검색 결과 없음" in out


# ---------------------------------------------------------------------------
# Ollama unavailable
# ---------------------------------------------------------------------------

def test_search_ollama_unavailable_exits_nonzero(capsys):
    import httpx

    embed_mock, session, session_ctx, search_mock = _patch_search_stack(
        [], embed_side_effect=httpx.ConnectError("connection refused")
    )

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        rc = main(["search", "RAG"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "Ollama" in err


def test_search_ollama_unavailable_does_not_call_db():
    import httpx

    embed_mock, session, session_ctx, search_mock = _patch_search_stack(
        [], embed_side_effect=httpx.ConnectError("connection refused")
    )

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG"])

    search_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# Rich table output
# ---------------------------------------------------------------------------

def test_search_results_output_contains_title(capsys):
    results = [_make_result(title="LightRAG v2.0")]
    embed_mock, session, session_ctx, search_mock = _patch_search_stack(results)

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG"])

    out = capsys.readouterr().out
    assert "LightRAG v2.0" in out


def test_search_results_output_contains_trust_score(capsys):
    results = [_make_result(trust_score=0.91)]
    embed_mock, session, session_ctx, search_mock = _patch_search_stack(results)

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG"])

    out = capsys.readouterr().out
    assert "0.91" in out


def test_search_results_none_trust_score_shows_dash(capsys):
    results = [_make_result(trust_score=None)]
    embed_mock, session, session_ctx, search_mock = _patch_search_stack(results)

    with (
        patch("argos.brain.ollama_client.embed", embed_mock),
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.services.search.search_tech_items", search_mock),
    ):
        main(["search", "RAG"])

    out = capsys.readouterr().out
    assert "—" in out or "-" in out
