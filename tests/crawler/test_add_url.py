"""Tests for the shared `add_url` service module (ARG-105)."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from argos.crawler.add_url import AddUrlResult, AddUrlStatus, add_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> AsyncMock:
    """Return an AsyncMock that satisfies the AsyncSession protocol used here."""
    return AsyncMock()


def _patch_robots(allowed: bool):
    return patch(
        "argos.crawler.add_url.is_robots_allowed",
        new=AsyncMock(return_value=allowed),
    )


def _patch_safe_url(safe: bool):
    return patch(
        "argos.crawler.add_url._is_safe_url",
        new=AsyncMock(return_value=safe),
    )


def _patch_dedup_lookup(*return_values):
    """Patch the existing source_url lookup; supply one value per call.

    add_url calls _find_existing_tech_item_id twice:
      1. Pre-fetch dedup check (None => no duplicate).
      2. Post-save id retrieval (uuid => row in DB).
    """
    return patch(
        "argos.crawler.add_url._find_existing_tech_item_id",
        new=AsyncMock(side_effect=list(return_values)),
    )


def _patch_fetch(content: dict | None, *, raises: Exception | None = None):
    if raises is not None:
        mock = AsyncMock(side_effect=raises)
    else:
        mock = AsyncMock(return_value=content)
    return patch("argos.crawler.add_url._fetch_url_content", new=mock)


def _patch_brain(state: dict, *, raises: Exception | None = None):
    if raises is not None:
        mock = AsyncMock(side_effect=raises)
    else:
        mock = AsyncMock(return_value=state)
    return patch("argos.crawler.add_url.run_brain_pipeline", new=mock)


# ---------------------------------------------------------------------------
# AddUrlResult dataclass surface
# ---------------------------------------------------------------------------


def test_addurl_status_values_are_strings():
    assert AddUrlStatus.CREATED.value == "created"
    assert AddUrlStatus.DUPLICATE.value == "duplicate"
    assert AddUrlStatus.REJECTED.value == "rejected"
    assert AddUrlStatus.ERROR.value == "error"


def test_addurl_result_default_fields():
    r = AddUrlResult(url="https://x.test/a", status=AddUrlStatus.CREATED)
    assert r.url == "https://x.test/a"
    assert r.status is AddUrlStatus.CREATED
    assert r.tech_item_id is None
    assert r.reason is None


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


async def test_rejects_unsupported_scheme() -> None:
    session = _make_session()
    r = await add_url("ftp://example.com/file", session)
    assert r.status is AddUrlStatus.REJECTED
    assert r.reason and "scheme" in r.reason.lower()


async def test_rejects_unparseable_url() -> None:
    session = _make_session()
    r = await add_url("not-a-url", session)
    assert r.status is AddUrlStatus.REJECTED


async def test_rejects_missing_host() -> None:
    session = _make_session()
    r = await add_url("http:///path-only", session)
    assert r.status is AddUrlStatus.REJECTED


async def test_rejects_when_ssrf_guard_blocks() -> None:
    session = _make_session()
    with _patch_safe_url(False):
        r = await add_url("http://10.0.0.1/internal", session)
    assert r.status is AddUrlStatus.REJECTED
    assert r.reason and (
        "ssrf" in r.reason.lower() or "host" in r.reason.lower()
    )


async def test_rejects_when_robots_disallows() -> None:
    session = _make_session()
    with _patch_safe_url(True), _patch_robots(False):
        r = await add_url("https://example.com/page", session)
    assert r.status is AddUrlStatus.REJECTED
    assert r.reason and "robots" in r.reason.lower()


# ---------------------------------------------------------------------------
# Dedup path
# ---------------------------------------------------------------------------


async def test_returns_duplicate_when_source_url_already_in_tech_items() -> None:
    session = _make_session()
    existing_id = uuid.uuid4()
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(existing_id),
    ):
        r = await add_url("https://example.com/page", session)
    assert r.status is AddUrlStatus.DUPLICATE
    assert r.tech_item_id == existing_id


async def test_duplicate_skips_fetch_and_brain() -> None:
    session = _make_session()
    existing_id = uuid.uuid4()

    fetch_mock = AsyncMock()
    brain_mock = AsyncMock()
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(existing_id),
        patch("argos.crawler.add_url._fetch_url_content", fetch_mock),
        patch("argos.crawler.add_url.run_brain_pipeline", brain_mock),
    ):
        await add_url("https://example.com/page", session)

    fetch_mock.assert_not_awaited()
    brain_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# Fetch failure path
# ---------------------------------------------------------------------------


async def test_returns_error_when_fetch_returns_none() -> None:
    session = _make_session()
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        _patch_fetch(None),
    ):
        r = await add_url("https://example.com/page", session)
    assert r.status is AddUrlStatus.ERROR
    assert r.reason and "fetch" in r.reason.lower()


async def test_returns_error_when_fetch_raises() -> None:
    session = _make_session()
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        _patch_fetch(None, raises=RuntimeError("boom")),
    ):
        r = await add_url("https://example.com/page", session)
    assert r.status is AddUrlStatus.ERROR
    assert r.reason and "fetch" in r.reason.lower()


async def test_returns_error_when_fetch_returns_empty_content() -> None:
    session = _make_session()
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        _patch_fetch({"title": "", "raw_content": "", "source_url": "https://example.com/page"}),
    ):
        r = await add_url("https://example.com/page", session)
    # Empty content can't run brain meaningfully; treat as error.
    assert r.status is AddUrlStatus.ERROR


# ---------------------------------------------------------------------------
# Happy path → brain pipeline runs
# ---------------------------------------------------------------------------


async def test_success_returns_created_with_tech_item_id() -> None:
    session = _make_session()
    new_id = uuid.uuid4()
    brain_state = {
        "is_valid": True,
        "saved": True,
        "source_url": "https://example.com/page",
    }
    # Pre-fetch dedup -> None; post-save lookup -> new_id.
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None, new_id),
        _patch_fetch(
            {
                "title": "Demo title",
                "raw_content": "Substantive body text for testing.",
                "source_url": "https://example.com/page",
            }
        ),
        _patch_brain(brain_state),
    ):
        r = await add_url("https://example.com/page", session)

    assert r.status is AddUrlStatus.CREATED
    assert r.tech_item_id == new_id
    session.commit.assert_awaited()


async def test_triage_rejected_returns_rejected_status() -> None:
    session = _make_session()
    brain_state = {
        "is_valid": False,
        "saved": False,
        "source_url": "https://example.com/page",
    }
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        _patch_fetch(
            {
                "title": "t",
                "raw_content": "body",
                "source_url": "https://example.com/page",
            }
        ),
        _patch_brain(brain_state),
    ):
        r = await add_url("https://example.com/page", session)

    assert r.status is AddUrlStatus.REJECTED
    assert r.tech_item_id is None
    assert r.reason and "triage" in r.reason.lower()


async def test_save_failed_returns_error_status() -> None:
    """is_valid=True but saved=False indicates save_node failed."""
    session = _make_session()
    brain_state = {
        "is_valid": True,
        "saved": False,
        "source_url": "https://example.com/page",
    }
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        _patch_fetch(
            {
                "title": "t",
                "raw_content": "body",
                "source_url": "https://example.com/page",
            }
        ),
        _patch_brain(brain_state),
    ):
        r = await add_url("https://example.com/page", session)

    assert r.status is AddUrlStatus.ERROR


async def test_brain_exception_returns_error_status() -> None:
    session = _make_session()
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        _patch_fetch(
            {
                "title": "t",
                "raw_content": "body",
                "source_url": "https://example.com/page",
            }
        ),
        _patch_brain({}, raises=RuntimeError("brain blew up")),
    ):
        r = await add_url("https://example.com/page", session)

    assert r.status is AddUrlStatus.ERROR
    assert r.reason and (
        "brain" in r.reason.lower() or "error" in r.reason.lower()
    )


async def test_success_uses_final_source_url_from_fetch() -> None:
    """If fetcher redirects, the final URL should be used for the result."""
    session = _make_session()
    new_id = uuid.uuid4()
    final_url = "https://example.com/redirected"
    brain_state = {
        "is_valid": True,
        "saved": True,
        "source_url": final_url,
    }
    # 3 lookups: (1) pre-fetch dedup on input URL, (2) post-redirect dedup on
    # final URL, (3) post-save id retrieval.
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None, None, new_id),
        _patch_fetch(
            {
                "title": "t",
                "raw_content": "body",
                "source_url": final_url,
            }
        ),
        _patch_brain(brain_state),
    ):
        r = await add_url("https://example.com/original", session)

    assert r.status is AddUrlStatus.CREATED
    assert r.url == final_url


# ---------------------------------------------------------------------------
# Helper: _find_existing_tech_item_id
# ---------------------------------------------------------------------------


async def test_find_existing_returns_id_when_present() -> None:
    from argos.crawler.add_url import _find_existing_tech_item_id

    session = MagicMock()
    expected_id = uuid.uuid4()
    scalar = MagicMock()
    scalar.scalar_one_or_none = MagicMock(return_value=expected_id)
    session.execute = AsyncMock(return_value=scalar)

    got = await _find_existing_tech_item_id(session, "https://example.com/page")

    assert got == expected_id
    session.execute.assert_awaited_once()


async def test_find_existing_returns_none_when_absent() -> None:
    from argos.crawler.add_url import _find_existing_tech_item_id

    session = MagicMock()
    scalar = MagicMock()
    scalar.scalar_one_or_none = MagicMock(return_value=None)
    session.execute = AsyncMock(return_value=scalar)

    got = await _find_existing_tech_item_id(session, "https://example.com/page")

    assert got is None
