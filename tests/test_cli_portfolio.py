"""Tests for `argos portfolio` CLI subcommand (ARG-113)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.cli import main
from argos.models.tech_item import CategoryType
from argos.models.user_asset import AssetStatus

_TS_CREATED = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
_TS_UPDATED = datetime(2026, 5, 9, 10, 0, 0, tzinfo=timezone.utc)
_TS_SIGNAL = datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc)


def _make_asset_item(
    title: str = "LangGraph 0.3",
    category: CategoryType = CategoryType.MAINSTREAM,
    trust_score: float | None = 0.8,
    created_at: datetime = _TS_CREATED,
    updated_at: datetime = _TS_UPDATED,
    last_monitored_at: datetime | None = None,
) -> tuple[MagicMock, MagicMock]:
    tech_id = uuid.uuid4()

    item = MagicMock()
    item.id = tech_id
    item.title = title
    item.category = category
    item.trust_score = trust_score
    item.created_at = created_at
    item.updated_at = updated_at
    item.source_url = f"https://example.com/{title.lower().replace(' ', '-')}"

    asset = MagicMock()
    asset.id = uuid.uuid4()
    asset.tech_id = tech_id
    asset.status = AssetStatus.KEEP
    asset.created_at = created_at
    asset.updated_at = updated_at
    asset.last_monitored_at = last_monitored_at

    return asset, item


def _make_session_ctx(portfolio_result=None):
    if portfolio_result is None:
        portfolio_result = []
    session = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return session, ctx


# ---------------------------------------------------------------------------
# Basic dispatch
# ---------------------------------------------------------------------------


def test_portfolio_command_returns_zero_on_success(capsys):
    asset, item = _make_asset_item()
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        rc = main(["portfolio"])

    assert rc == 0


def test_portfolio_command_accepts_config_flag():
    asset, item = _make_asset_item()
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
        patch("argos.cli._apply_config_override", return_value=None) as mock_override,
    ):
        rc = main(["portfolio", "--config", "/some/path.toml"])

    assert rc == 0
    mock_override.assert_called_once()
    assert mock_override.call_args.args[0].config == "/some/path.toml"


def test_portfolio_config_override_error_propagates():
    with patch("argos.cli._apply_config_override", return_value=3):
        rc = main(["portfolio", "--config", "/bad/path.toml"])

    assert rc == 3


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


def test_portfolio_empty_state_prints_friendly_message_and_exits_zero(capsys):
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        rc = main(["portfolio"])

    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip()  # some message was printed


# ---------------------------------------------------------------------------
# Output format — happy path
# ---------------------------------------------------------------------------


def test_portfolio_output_contains_header_with_count(capsys):
    asset, item = _make_asset_item(title="LangGraph 0.3")
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio"])

    out = capsys.readouterr().out
    assert "Keep 포트폴리오" in out
    assert "1개" in out or "1" in out


def test_portfolio_output_contains_title(capsys):
    asset, item = _make_asset_item(title="LightRAG")
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio"])

    out = capsys.readouterr().out
    assert "LightRAG" in out


def test_portfolio_output_contains_kept_date(capsys):
    asset, item = _make_asset_item(created_at=_TS_CREATED)
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio"])

    out = capsys.readouterr().out
    assert "2026-05-01" in out


def test_portfolio_output_shows_dash_when_no_last_signal(capsys):
    asset, item = _make_asset_item(last_monitored_at=None)
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio"])

    out = capsys.readouterr().out
    assert "—" in out or "-" in out


def test_portfolio_output_shows_last_signal_date_when_present(capsys):
    asset, item = _make_asset_item(last_monitored_at=_TS_SIGNAL)
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio"])

    out = capsys.readouterr().out
    assert "2026-05-10" in out


def test_portfolio_shows_category_header_when_items_exist(capsys):
    asset, item = _make_asset_item(category=CategoryType.ALPHA)
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio"])

    out = capsys.readouterr().out
    assert "Alpha" in out


def test_portfolio_does_not_show_empty_category_headers(capsys):
    """Only categories with ≥1 item should show a header."""
    asset, item = _make_asset_item(category=CategoryType.ALPHA)
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio"])

    out = capsys.readouterr().out
    # Alpha has items, Mainstream does not → only Alpha header should appear
    assert "Alpha" in out
    assert "Mainstream" not in out


# ---------------------------------------------------------------------------
# --category filter
# ---------------------------------------------------------------------------


def test_portfolio_category_alpha_calls_fetch_with_alpha_enum(capsys):
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio", "--category", "alpha"])

    call_kwargs = portfolio_mock.await_args.kwargs
    assert call_kwargs.get("category") == CategoryType.ALPHA


def test_portfolio_category_mainstream_calls_fetch_with_mainstream_enum(capsys):
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio", "--category", "mainstream"])

    call_kwargs = portfolio_mock.await_args.kwargs
    assert call_kwargs.get("category") == CategoryType.MAINSTREAM


def test_portfolio_no_category_calls_fetch_with_none():
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio"])

    call_kwargs = portfolio_mock.await_args.kwargs
    assert call_kwargs.get("category") is None


def test_portfolio_invalid_category_rejected():
    with pytest.raises(SystemExit) as exc_info:
        main(["portfolio", "--category", "invalid"])
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# --sort flag
# ---------------------------------------------------------------------------


def test_portfolio_sort_trust_calls_fetch_with_trust():
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio", "--sort", "trust"])

    call_kwargs = portfolio_mock.await_args.kwargs
    assert call_kwargs.get("sort_by") == "trust"


def test_portfolio_sort_date_default():
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        main(["portfolio"])

    call_kwargs = portfolio_mock.await_args.kwargs
    assert call_kwargs.get("sort_by") == "date"


def test_portfolio_invalid_sort_rejected():
    with pytest.raises(SystemExit) as exc_info:
        main(["portfolio", "--sort", "invalid"])
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Both --category and --sort combined
# ---------------------------------------------------------------------------


def test_portfolio_category_and_sort_combined():
    asset, item = _make_asset_item(category=CategoryType.ALPHA)
    session, ctx = _make_session_ctx()
    portfolio_mock = AsyncMock(return_value=[(asset, item)])

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(
            "argos.slack.services.briefing_query.fetch_user_portfolio",
            portfolio_mock,
        ),
    ):
        rc = main(["portfolio", "--category", "alpha", "--sort", "trust"])

    assert rc == 0
    call_kwargs = portfolio_mock.await_args.kwargs
    assert call_kwargs.get("category") == CategoryType.ALPHA
    assert call_kwargs.get("sort_by") == "trust"
