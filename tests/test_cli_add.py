"""Tests for `argos add` CLI subcommand (ARG-107)."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from argos.cli import main
from argos.crawler.add_url import AddUrlResult, AddUrlStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(url: str, status: AddUrlStatus, **kwargs) -> AddUrlResult:
    defaults: dict = {
        "url": url,
        "status": status,
        "tech_item_id": None,
        "reason": None,
    }
    defaults.update(kwargs)
    return AddUrlResult(**defaults)


def _make_session_ctx():
    session = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return session, ctx


def _patch_add_stack(*results):
    """Patch add_url + AsyncSessionLocal. add_url returns *results* in order."""
    session, session_ctx = _make_session_ctx()
    add_mock = AsyncMock(side_effect=list(results))
    return session, session_ctx, add_mock


# ---------------------------------------------------------------------------
# Dispatch / exit code
# ---------------------------------------------------------------------------


def test_add_command_returns_zero_on_created():
    new_id = uuid.uuid4()
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://x.test/a", AddUrlStatus.CREATED, tech_item_id=new_id)
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        rc = main(["add", "https://x.test/a"])

    assert rc == 0


def test_add_command_returns_zero_on_duplicate():
    existing = uuid.uuid4()
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://x.test/a", AddUrlStatus.DUPLICATE, tech_item_id=existing)
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        rc = main(["add", "https://x.test/a"])

    assert rc == 0


def test_add_command_returns_nonzero_on_rejected():
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("ftp://bad", AddUrlStatus.REJECTED, reason="unsupported scheme")
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        rc = main(["add", "ftp://bad"])

    assert rc != 0


def test_add_command_returns_nonzero_on_error():
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://x.test/a", AddUrlStatus.ERROR, reason="fetch failed")
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        rc = main(["add", "https://x.test/a"])

    assert rc != 0


def test_add_command_mixed_results_returns_nonzero():
    """One success + one error → nonzero exit (any failure fails the command)."""
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://x.test/a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
        _make_result("https://x.test/b", AddUrlStatus.ERROR, reason="fetch failed"),
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        rc = main(["add", "https://x.test/a", "https://x.test/b"])

    assert rc != 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_add_command_with_no_urls_errors(capsys):
    """At least one URL must be supplied — either positional or --url."""
    rc = main(["add"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "url" in err.lower() or "argument" in err.lower()


def test_add_command_accepts_multiple_positional_urls():
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
        _make_result("https://b", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        rc = main(["add", "https://a", "https://b"])

    assert rc == 0
    assert add_mock.await_count == 2


def test_add_command_accepts_url_option_repeated():
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
        _make_result("https://b", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        rc = main(["add", "--url", "https://a", "--url", "https://b"])

    assert rc == 0
    assert add_mock.await_count == 2


def test_add_command_mixes_positional_and_url_option():
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
        _make_result("https://b", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        rc = main(["add", "https://a", "--url", "https://b"])

    assert rc == 0
    assert add_mock.await_count == 2


def test_add_command_deduplicates_input_urls():
    """The same URL passed twice should only call add_url once."""
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        rc = main(["add", "https://a", "https://a"])

    assert rc == 0
    assert add_mock.await_count == 1


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------


def test_add_accepts_config_flag():
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://x.test/a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4())
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
        patch("argos.cli._apply_config_override", return_value=None) as mock_override,
    ):
        rc = main(["add", "https://x.test/a", "--config", "/some/path.toml"])

    assert rc == 0
    mock_override.assert_called_once()
    assert mock_override.call_args.args[0].config == "/some/path.toml"


def test_add_config_override_error_propagates():
    with patch("argos.cli._apply_config_override", return_value=3):
        rc = main(["add", "https://x.test/a", "--config", "/bad/path.toml"])

    assert rc == 3


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def test_add_command_prints_url_and_status(capsys):
    new_id = uuid.uuid4()
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result(
            "https://x.test/a", AddUrlStatus.CREATED, tech_item_id=new_id
        )
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        main(["add", "https://x.test/a"])

    out = capsys.readouterr().out
    assert "https://x.test/a" in out
    assert "created" in out.lower()


def test_add_command_prints_reason_for_failures(capsys):
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result(
            "https://x.test/a",
            AddUrlStatus.REJECTED,
            reason="robots.txt disallows",
        )
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        main(["add", "https://x.test/a"])

    out = capsys.readouterr().out
    assert "robots" in out.lower()


def test_add_command_prints_tech_item_id_for_duplicate(capsys):
    existing_id = uuid.uuid4()
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result(
            "https://x.test/a",
            AddUrlStatus.DUPLICATE,
            tech_item_id=existing_id,
        )
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        main(["add", "https://x.test/a"])

    out = capsys.readouterr().out
    # The id (or a short prefix of it) should appear in the output.
    assert str(existing_id)[:8] in out


# ---------------------------------------------------------------------------
# Pass-through to service
# ---------------------------------------------------------------------------


def test_add_command_passes_each_url_to_service():
    session, session_ctx, add_mock = _patch_add_stack(
        _make_result("https://a", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
        _make_result("https://b", AddUrlStatus.CREATED, tech_item_id=uuid.uuid4()),
    )

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.crawler.add_url.add_url", add_mock),
    ):
        main(["add", "https://a", "--url", "https://b"])

    awaited_urls = [call.args[0] for call in add_mock.await_args_list]
    assert awaited_urls == ["https://a", "https://b"]
