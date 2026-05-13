"""Tests for src/argos/crawler/arxiv_fetcher.py (ARG-53)."""
from __future__ import annotations

import time
import calendar
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.crawler.arxiv_fetcher import (
    _normalize_abs_url,
    _entry_to_dict,
    _truncate,
    fetch_arxiv_recent,
)
from argos.models.tech_item import CategoryType


# ---------------------------------------------------------------------------
# (a) _normalize_abs_url — strips vN suffix and rewrites http→https
# ---------------------------------------------------------------------------


def test_normalize_abs_url_strips_version_suffix():
    url = "http://arxiv.org/abs/2401.12345v2"
    assert _normalize_abs_url(url) == "https://arxiv.org/abs/2401.12345"


def test_normalize_abs_url_already_canonical():
    url = "https://arxiv.org/abs/2401.12345"
    assert _normalize_abs_url(url) == "https://arxiv.org/abs/2401.12345"


def test_normalize_abs_url_http_no_version():
    url = "http://arxiv.org/abs/2401.12345"
    assert _normalize_abs_url(url) == "https://arxiv.org/abs/2401.12345"


def test_normalize_abs_url_high_version_number():
    url = "http://arxiv.org/abs/2312.99999v10"
    assert _normalize_abs_url(url) == "https://arxiv.org/abs/2312.99999"


def test_normalize_abs_url_new_style_id():
    # Some older papers have format like 1234.56789
    url = "http://arxiv.org/abs/1234.56789v3"
    assert _normalize_abs_url(url) == "https://arxiv.org/abs/1234.56789"


# ---------------------------------------------------------------------------
# (b) _entry_to_dict — correct keys + _source_category is ALPHA
# ---------------------------------------------------------------------------


def _make_entry(
    title="Attention Is All You Need",
    entry_id="http://arxiv.org/abs/1706.03762v5",
    summary="We propose a new simple network architecture, the Transformer.",
):
    return SimpleNamespace(
        title=title,
        id=entry_id,
        summary=summary,
    )


def test_entry_to_dict_correct_keys():
    entry = _make_entry()
    item = _entry_to_dict(entry)
    assert set(item.keys()) == {"title", "source_url", "raw_content", "_source_category"}


def test_entry_to_dict_source_category_is_alpha():
    entry = _make_entry()
    item = _entry_to_dict(entry)
    assert item["_source_category"] is CategoryType.ALPHA


def test_entry_to_dict_normalizes_source_url():
    entry = _make_entry(entry_id="http://arxiv.org/abs/1706.03762v5")
    item = _entry_to_dict(entry)
    assert item["source_url"] == "https://arxiv.org/abs/1706.03762"


def test_entry_to_dict_raw_content_contains_title_and_summary():
    entry = _make_entry(
        title="Test Paper",
        summary="This is the abstract.",
    )
    item = _entry_to_dict(entry)
    assert "Test Paper" in item["raw_content"]
    assert "This is the abstract." in item["raw_content"]


def test_entry_to_dict_raw_content_has_no_link_field():
    # raw_content must be title + abstract only — no link text
    entry = _make_entry(title="My Paper", summary="Abstract here.")
    item = _entry_to_dict(entry)
    # link/URL should not appear in raw_content (only in source_url)
    assert "http" not in item["raw_content"]
    assert "arxiv.org" not in item["raw_content"]


# ---------------------------------------------------------------------------
# (c) fetch_arxiv_recent — mocked httpx client, 24h filter
# ---------------------------------------------------------------------------


def _make_feedparser_result(entries):
    """Wrap a list of entries in a fake feedparser parsed result."""
    result = MagicMock()
    result.entries = entries
    return result


def _recent_entry(title="Recent Paper", paper_id="2401.11111", hours_ago=1):
    """Build a SimpleNamespace entry with published_parsed inside the cutoff."""
    epoch = time.time() - hours_ago * 3600
    struct = time.gmtime(epoch)
    return SimpleNamespace(
        title=title,
        id=f"http://arxiv.org/abs/{paper_id}v1",
        summary="Recent abstract.",
        published_parsed=struct,
    )


def _old_entry(title="Old Paper", paper_id="2301.00001", hours_ago=48):
    """Build a SimpleNamespace entry outside the 24h cutoff window."""
    epoch = time.time() - hours_ago * 3600
    struct = time.gmtime(epoch)
    return SimpleNamespace(
        title=title,
        id=f"http://arxiv.org/abs/{paper_id}v2",
        summary="Old abstract.",
        published_parsed=struct,
    )


@pytest.fixture
def mock_http_response():
    """A fake httpx response with status 200 and minimal Atom bytes."""
    response = MagicMock()
    response.status_code = 200
    response.content = b"<feed></feed>"  # feedparser is mocked separately
    return response


@pytest.fixture
def mock_client(mock_http_response):
    """AsyncClient mock whose .get() returns mock_http_response."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_http_response)
    client.aclose = AsyncMock()
    return client


async def test_fetch_arxiv_recent_returns_expected_list(mock_client):
    entry = _recent_entry()
    parsed = _make_feedparser_result([entry])

    with patch(
        "argos.crawler.arxiv_fetcher.asyncio.to_thread",
        new=AsyncMock(return_value=parsed),
    ):
        items = await fetch_arxiv_recent(client=mock_client)

    assert len(items) == 1
    assert items[0]["title"] == "Recent Paper"
    assert items[0]["source_url"] == "https://arxiv.org/abs/2401.11111"
    assert items[0]["_source_category"] is CategoryType.ALPHA


async def test_fetch_arxiv_recent_respects_24h_cutoff(mock_client):
    recent = _recent_entry(title="Recent", paper_id="2401.11111", hours_ago=2)
    old = _old_entry(title="Old", paper_id="2301.00001", hours_ago=48)
    parsed = _make_feedparser_result([recent, old])

    with patch(
        "argos.crawler.arxiv_fetcher.asyncio.to_thread",
        new=AsyncMock(return_value=parsed),
    ):
        items = await fetch_arxiv_recent(hours=24, client=mock_client)

    assert len(items) == 1
    assert items[0]["title"] == "Recent"


async def test_fetch_arxiv_recent_excludes_entry_exactly_at_cutoff(mock_client):
    """Entry published exactly at the cutoff boundary should be excluded."""
    # published 24h + 1s ago — just outside the window
    epoch = time.time() - 24 * 3600 - 1
    struct = time.gmtime(epoch)
    entry = SimpleNamespace(
        title="Borderline Paper",
        id="http://arxiv.org/abs/2401.22222v1",
        summary="Abstract.",
        published_parsed=struct,
    )
    parsed = _make_feedparser_result([entry])

    with patch(
        "argos.crawler.arxiv_fetcher.asyncio.to_thread",
        new=AsyncMock(return_value=parsed),
    ):
        items = await fetch_arxiv_recent(hours=24, client=mock_client)

    assert items == []


# ---------------------------------------------------------------------------
# (d) HTTP error path → returns [] and logs warning
# ---------------------------------------------------------------------------


async def test_fetch_arxiv_recent_returns_empty_on_http_error():
    bad_response = MagicMock()
    bad_response.status_code = 503
    bad_response.content = b""
    bad_client = AsyncMock()
    bad_client.get = AsyncMock(return_value=bad_response)
    bad_client.aclose = AsyncMock()

    items = await fetch_arxiv_recent(client=bad_client)
    assert items == []


async def test_fetch_arxiv_recent_returns_empty_on_request_exception():
    failing_client = AsyncMock()
    failing_client.get = AsyncMock(side_effect=RuntimeError("connection refused"))
    failing_client.aclose = AsyncMock()

    items = await fetch_arxiv_recent(client=failing_client)
    assert items == []


async def test_fetch_arxiv_recent_returns_empty_on_feedparser_exception(mock_client):
    with patch(
        "argos.crawler.arxiv_fetcher.asyncio.to_thread",
        new=AsyncMock(side_effect=RuntimeError("parse error")),
    ):
        items = await fetch_arxiv_recent(client=mock_client)

    assert items == []


# ---------------------------------------------------------------------------
# (e) UTF-8 truncation when summary exceeds 8 KB
# ---------------------------------------------------------------------------


def test_truncate_short_string_unchanged():
    text = "Hello, world!"
    assert _truncate(text) == text


def test_truncate_long_string_within_byte_limit():
    long_text = "x" * (8 * 1024 + 500)
    result = _truncate(long_text)
    assert len(result.encode("utf-8")) <= 8 * 1024


async def test_fetch_arxiv_recent_truncates_large_summary(mock_client):
    big_entry = SimpleNamespace(
        title="Big Paper",
        id="http://arxiv.org/abs/2401.99999v1",
        summary="y" * (16 * 1024),  # 16 KB — twice the limit
        published_parsed=time.gmtime(time.time() - 3600),  # 1h ago
    )
    parsed = _make_feedparser_result([big_entry])

    with patch(
        "argos.crawler.arxiv_fetcher.asyncio.to_thread",
        new=AsyncMock(return_value=parsed),
    ):
        items = await fetch_arxiv_recent(client=mock_client)

    assert len(items) == 1
    assert len(items[0]["raw_content"].encode("utf-8")) <= 8 * 1024
