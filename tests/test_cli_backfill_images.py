"""Tests for `argos backfill-images` CLI subcommand (ARG-179 / T4).

Mock-level tests — no DB required. Patches `argos.cli.AsyncSessionLocal`
and the internal `_backfill_images` coroutine exactly like `test_cli_add.py`.
"""
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


# ---------------------------------------------------------------------------
# Dispatch / exit code
# ---------------------------------------------------------------------------


def test_backfill_images_dispatches():
    """backfill-images parses, dispatches to _backfill_images, returns EXIT_OK."""
    with patch("argos.cli._backfill_images", new_callable=AsyncMock, return_value=0) as m:
        rc = main(["backfill-images"])
    assert rc == 0
    m.assert_awaited_once()
    # default: refetch flag should be False
    assert m.call_args.kwargs.get("refetch", False) is False


def test_backfill_images_refetch_flag():
    """--refetch is threaded through to _backfill_images(refetch=True)."""
    with patch("argos.cli._backfill_images", new_callable=AsyncMock, return_value=0) as m:
        rc = main(["backfill-images", "--refetch"])
    assert rc == 0
    m.assert_awaited_once()
    assert m.call_args.kwargs.get("refetch") is True


def test_backfill_images_default_no_refetch():
    """Without --refetch the refetch kwarg is False."""
    with patch("argos.cli._backfill_images", new_callable=AsyncMock, return_value=0) as m:
        rc = main(["backfill-images"])
    assert rc == 0
    assert m.call_args.kwargs.get("refetch", False) is False


# ---------------------------------------------------------------------------
# Default path: favicon_for_domain, no network
# ---------------------------------------------------------------------------


def test_backfill_images_default_path_calls_favicon_no_network(capsys):
    """Default path calls favicon_for_domain and never touches the network."""
    session, session_ctx = _make_session_ctx()

    # Simulate one NULL row
    row = (MagicMock(), "https://example.com/article")
    session.execute = AsyncMock(
        side_effect=[
            # first call: SELECT rows with image_url IS NULL
            MagicMock(**{"all.return_value": [row]}),
            # second call: UPDATE result
            MagicMock(rowcount=1),
        ]
    )
    session.commit = AsyncMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch(
            "argos.crawler._og_image.favicon_for_domain",
            return_value="https://example.com/favicon.ico",
        ) as favicon_mock,
        patch("argos.crawler.add_url._fetch_url_content", new_callable=AsyncMock) as fetch_mock,
    ):
        rc = main(["backfill-images"])

    assert rc == 0
    # Positively assert favicon_for_domain was called with the row's source_url —
    # this proves the no-network favicon route actually executed.
    favicon_mock.assert_called_once_with("https://example.com/article")
    # Secondary guard: the network fetch path must NOT have been touched.
    fetch_mock.assert_not_awaited()


def test_backfill_images_refetch_path_calls_fetch(capsys):
    """--refetch path calls _fetch_url_content (the network path)."""
    session, session_ctx = _make_session_ctx()

    row = (MagicMock(), "https://example.com/article")
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(**{"all.return_value": [row]}),
            MagicMock(rowcount=1),
        ]
    )
    session.commit = AsyncMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch(
            "argos.crawler.add_url._fetch_url_content",
            new_callable=AsyncMock,
            return_value={"image_url": "https://example.com/og.png"},
        ) as fetch_mock,
    ):
        rc = main(["backfill-images", "--refetch"])

    assert rc == 0
    fetch_mock.assert_awaited_once_with("https://example.com/article")
