"""Tests for `argos stats` CLI subcommand (ARG-66/ARG-106/ARG-108/ARG-109/ARG-111)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from argos.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_ctx():
    session = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return session, ctx


def _make_stats_data(
    total_items: int = 10,
    github_count: int = 3,
    hn_count: int = 3,
    rss_count: int = 2,
    arxiv_count: int = 2,
    valid_count: int = 7,
    new_saved_count: int = 5,
    keep_count: int = 2,
    pass_count: int = 3,
    unclassified_count: int = 2,
    total_keep_cumulative: int = 15,
    track_alert_count: int = 4,
):
    """Return a dict matching what fetch_stats_summary returns."""
    return {
        "total_items": total_items,
        "github_count": github_count,
        "hn_count": hn_count,
        "rss_count": rss_count,
        "arxiv_count": arxiv_count,
        "valid_count": valid_count,
        "new_saved_count": new_saved_count,
        "keep_count": keep_count,
        "pass_count": pass_count,
        "unclassified_count": unclassified_count,
        "total_keep_cumulative": total_keep_cumulative,
        "track_alert_count": track_alert_count,
    }


STATS_MODULE = "argos.slack.services.stats_query.fetch_stats_summary"


# ===========================================================================
# ARG-106: CLI skeleton + --days + header
# ===========================================================================


def test_stats_command_returns_zero_on_success(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data()
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        rc = main(["stats"])

    assert rc == 0


def test_stats_default_days_is_7(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data()
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats"])

    out = capsys.readouterr().out
    assert "7일" in out


def test_stats_days_30_reflected_in_header(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data()
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats", "--days", "30"])

    out = capsys.readouterr().out
    assert "30일" in out


def test_stats_header_contains_argos_통계(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data()
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats"])

    out = capsys.readouterr().out
    assert "통계" in out


def test_stats_days_zero_exits_nonzero(capsys):
    rc = main(["stats", "--days", "0"])
    assert rc != 0


def test_stats_days_negative_exits_nonzero(capsys):
    rc = main(["stats", "--days", "-5"])
    assert rc != 0


def test_stats_days_zero_prints_error_message(capsys):
    main(["stats", "--days", "0"])
    combined = capsys.readouterr()
    # The function prints an error message to stderr; non-zero exit already verified above
    assert combined.err or combined.out  # something was printed


def test_stats_accepts_config_flag(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data()
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
        patch("argos.cli._apply_config_override", return_value=None),
    ):
        rc = main(["stats", "--config", "/some/path.toml"])

    assert rc == 0


# ===========================================================================
# ARG-108: collection section (per-source counts)
# ===========================================================================


def test_stats_collection_shows_total_item_count(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data(total_items=142, github_count=45, hn_count=38, rss_count=41, arxiv_count=18)
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats"])

    out = capsys.readouterr().out
    assert "142" in out


def test_stats_collection_shows_per_source_breakdown(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data(total_items=142, github_count=45, hn_count=38, rss_count=41, arxiv_count=18)
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats"])

    out = capsys.readouterr().out
    assert "GitHub" in out
    assert "HN" in out
    assert "arXiv" in out


def test_stats_collection_zero_items_safe(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data(
        total_items=0, github_count=0, hn_count=0, rss_count=0, arxiv_count=0,
        valid_count=0, new_saved_count=0, keep_count=0, pass_count=0,
        unclassified_count=0, total_keep_cumulative=0, track_alert_count=0,
    )
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        rc = main(["stats"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "0" in out


def test_stats_days_changes_query_window(capsys):
    """--days N is forwarded to the query function."""
    session, ctx = _make_session_ctx()
    data = _make_stats_data()
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats", "--days", "14"])

    # The query fn should be called with days=14
    call_kwargs = mock_fn.await_args.kwargs
    assert call_kwargs.get("days") == 14 or (mock_fn.await_args.args and 14 in mock_fn.await_args.args)


# ---------------------------------------------------------------------------
# Pure helper: source domain mapper (ARG-108)
# ---------------------------------------------------------------------------


def test_classify_source_github():
    from argos.slack.services.stats_query import classify_source

    assert classify_source("https://github.com/owner/repo") == "GitHub"


def test_classify_source_github_subdomains():
    from argos.slack.services.stats_query import classify_source

    assert classify_source("https://raw.githubusercontent.com/owner/repo/main/README.md") == "GitHub"
    assert classify_source("https://gist.github.com/user/abc123") == "GitHub"


def test_classify_source_hn():
    from argos.slack.services.stats_query import classify_source

    assert classify_source("https://news.ycombinator.com/item?id=12345") == "HN"
    assert classify_source("https://hacker-news.firebaseio.com/v0/item/123.json") == "HN"


def test_classify_source_arxiv():
    from argos.slack.services.stats_query import classify_source

    assert classify_source("https://arxiv.org/abs/2301.07041") == "arXiv"


def test_classify_source_other_maps_to_rss():
    from argos.slack.services.stats_query import classify_source

    assert classify_source("https://someblog.com/post/123") == "RSS"
    assert classify_source("https://techcrunch.com/article/foo") == "RSS"


def test_classify_source_empty_url_maps_to_rss():
    from argos.slack.services.stats_query import classify_source

    assert classify_source("") == "RSS"


# ===========================================================================
# ARG-109: brain/triage section (valid-rate + Keep/Pass)
# ===========================================================================


def test_stats_valid_count_shown(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data(total_items=142, valid_count=67)
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats"])

    out = capsys.readouterr().out
    assert "67" in out


def test_stats_valid_percentage_shown(capsys):
    session, ctx = _make_session_ctx()
    # 67/142 ≈ 47%
    data = _make_stats_data(total_items=142, valid_count=67)
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats"])

    out = capsys.readouterr().out
    assert "47" in out


def test_stats_valid_percentage_division_by_zero_safe(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data(total_items=0, valid_count=0)
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        rc = main(["stats"])

    assert rc == 0
    out = capsys.readouterr().out
    # Should show 0% or "-" — not raise an exception
    assert "0%" in out or "-" in out or "0" in out


def test_stats_keep_pass_unclassified_shown(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data(keep_count=12, pass_count=28, unclassified_count=3)
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats"])

    out = capsys.readouterr().out
    assert "Keep" in out
    assert "Pass" in out
    assert "12" in out
    assert "28" in out


# ---------------------------------------------------------------------------
# Pure helper: safe_pct
# ---------------------------------------------------------------------------


def test_safe_pct_normal():
    from argos.slack.services.stats_query import safe_pct

    assert safe_pct(67, 142) == 47


def test_safe_pct_zero_denominator():
    from argos.slack.services.stats_query import safe_pct

    result = safe_pct(0, 0)
    # Should return 0 or a sentinel, not raise
    assert result == 0 or result is None


def test_safe_pct_full():
    from argos.slack.services.stats_query import safe_pct

    assert safe_pct(142, 142) == 100


# ===========================================================================
# ARG-111: portfolio + Track-alert section
# ===========================================================================


def test_stats_portfolio_cumulative_keep_shown(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data(total_keep_cumulative=31)
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats"])

    out = capsys.readouterr().out
    assert "31" in out
    assert "Keep" in out


def test_stats_track_alert_count_shown(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data(track_alert_count=8)
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats"])

    out = capsys.readouterr().out
    assert "8" in out


def test_stats_track_alert_shows_days(capsys):
    session, ctx = _make_session_ctx()
    data = _make_stats_data(track_alert_count=8)
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        main(["stats", "--days", "7"])

    out = capsys.readouterr().out
    # Should mention the N days somewhere near the track alert section
    assert "7" in out


def test_stats_cumulative_keep_unchanged_by_days(capsys):
    """The cumulative Keep count should be the same regardless of --days."""
    session, ctx = _make_session_ctx()
    data_7 = _make_stats_data(total_keep_cumulative=31, track_alert_count=4)
    data_30 = _make_stats_data(total_keep_cumulative=31, track_alert_count=12)
    mock_fn_7 = AsyncMock(return_value=data_7)
    mock_fn_30 = AsyncMock(return_value=data_30)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn_7),
    ):
        main(["stats", "--days", "7"])
    out_7 = capsys.readouterr().out

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn_30),
    ):
        main(["stats", "--days", "30"])
    out_30 = capsys.readouterr().out

    # Both outputs should show 31 (cumulative Keep)
    assert "31" in out_7
    assert "31" in out_30


def test_stats_empty_db_safe(capsys):
    """Empty DB: total_keep_cumulative=0 and track_alert_count=0 should not crash."""
    session, ctx = _make_session_ctx()
    data = _make_stats_data(
        total_items=0, github_count=0, hn_count=0, rss_count=0, arxiv_count=0,
        valid_count=0, new_saved_count=0, keep_count=0, pass_count=0,
        unclassified_count=0, total_keep_cumulative=0, track_alert_count=0,
    )
    mock_fn = AsyncMock(return_value=data)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=ctx),
        patch(STATS_MODULE, mock_fn),
    ):
        rc = main(["stats"])

    assert rc == 0
