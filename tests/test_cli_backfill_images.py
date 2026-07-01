"""Tests for `argos backfill-images` CLI subcommand (ARG-179 / T4).

Mock-level tests — no DB required. Patches `argos.cli.AsyncSessionLocal`
and the internal `_backfill_images` coroutine exactly like `test_cli_add.py`.
"""
from __future__ import annotations

from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

from argos.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Mirrors the SQLAlchemy Row shape of the SELECT in `_backfill_images`:
# `select(TechItem.id, TechItem.source_url, TechItem.image_url)`. The code
# reads these columns by attribute (r.id / r.source_url / r.image_url), so the
# mock rows must be attribute-accessible — a plain tuple is not.
_Row = namedtuple("_Row", ["id", "source_url", "image_url"])


def _row(source_url: str, image_url: str | None = None):
    return _Row(id=MagicMock(), source_url=source_url, image_url=image_url)


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
    row = _row("https://example.com/article")
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

    row = _row("https://example.com/article")
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(**{"all.return_value": [row]}),
            MagicMock(rowcount=1),
        ]
    )
    session.commit = AsyncMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        # Gate is exercised by its own test below; force-pass it here so this
        # test stays network-free (no real DNS for the SSRF safety check).
        patch(
            "argos.crawler.dynamic_fetcher._is_safe_url",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "argos.crawler.add_url._fetch_url_content",
            new_callable=AsyncMock,
            return_value={"image_url": "https://example.com/og.png"},
        ) as fetch_mock,
    ):
        rc = main(["backfill-images", "--refetch"])

    assert rc == 0
    fetch_mock.assert_awaited_once_with("https://example.com/article")


def test_backfill_images_refetch_skips_unsafe_url(capsys):
    """--refetch must NOT fetch a stored row whose source_url targets a
    private/loopback/metadata host — it re-applies add_url()'s SSRF gate and
    falls back to the favicon instead of issuing the request."""
    session, session_ctx = _make_session_ctx()

    row = _row("http://169.254.169.254/latest/meta-data/")
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
        ) as fetch_mock,
    ):
        rc = main(["backfill-images", "--refetch"])

    assert rc == 0
    # The unsafe URL is link-local — never fetched.
    fetch_mock.assert_not_awaited()
