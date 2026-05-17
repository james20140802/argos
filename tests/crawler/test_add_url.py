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


# ---------------------------------------------------------------------------
# SSRF: redirect re-validation (Codex review on PR #67)
#
# httpx in add_url follows redirects transparently, so the SSRF + robots
# checks at the entry of add_url cover only the user-supplied URL. The
# static fetch path must re-validate response.url before trusting the body.
# ---------------------------------------------------------------------------


def _patch_get_with_retry(response: MagicMock):
    return patch(
        "argos.crawler.add_url._get_with_retry",
        new=AsyncMock(return_value=response),
    )


def _make_html_response(final_url: str, body: str = "<html><body><p>x</p></body></html>") -> MagicMock:
    response = MagicMock()
    response.url = final_url
    response.text = body
    response.headers = {"content-type": "text/html"}
    return response


async def test_static_fetch_revalidates_redirect_blocks_private_target() -> None:
    """If the static fetcher's response.url resolves to a private host, add_url
    must reject the fetch rather than trust the body — even though the original
    URL passed SSRF.
    """
    session = _make_session()
    safe_url = AsyncMock(side_effect=[True, False])  # original ok, redirect blocked
    response = _make_html_response("http://10.0.0.1/internal")
    extract = MagicMock(return_value=("Title", "Lots of body text"))
    fetch_dynamic = AsyncMock(return_value=None)
    with (
        patch("argos.crawler.add_url._is_safe_url", new=safe_url),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        _patch_get_with_retry(response),
        patch("argos.crawler.add_url.extract_main_content", new=extract),
        patch("argos.crawler.add_url.fetch_dynamic_page", new=fetch_dynamic),
    ):
        r = await add_url("https://example.com/start", session)

    # The fetch path must fall back / fail without persisting anything.
    assert r.status is AddUrlStatus.ERROR
    # Both safe_url calls happened: pre-fetch + post-redirect.
    assert safe_url.await_count >= 2


async def test_static_fetch_revalidates_redirect_blocks_robots_disallowed_target() -> None:
    """A redirect target that is robots-disallowed must also be rejected after
    fetch, not just at the original URL.
    """
    session = _make_session()
    # robots: original allowed, redirected target disallowed.
    robots = AsyncMock(side_effect=[True, False])
    safe_url = AsyncMock(return_value=True)
    response = _make_html_response("https://forbidden.example.com/blocked")
    extract = MagicMock(return_value=("Title", "Body"))
    fetch_dynamic = AsyncMock(return_value=None)
    with (
        patch("argos.crawler.add_url._is_safe_url", new=safe_url),
        patch("argos.crawler.add_url.is_robots_allowed", new=robots),
        _patch_dedup_lookup(None),
        _patch_get_with_retry(response),
        patch("argos.crawler.add_url.extract_main_content", new=extract),
        patch("argos.crawler.add_url.fetch_dynamic_page", new=fetch_dynamic),
    ):
        r = await add_url("https://example.com/start", session)

    # Redirected to a robots-disallowed URL — surface as REJECTED.
    assert r.status is AddUrlStatus.REJECTED
    assert r.reason and "robots" in r.reason.lower()


async def test_static_fetch_no_redirect_does_not_double_validate() -> None:
    """If response.url == original URL, no extra _is_safe_url call is needed."""
    session = _make_session()
    new_id = uuid.uuid4()
    safe_url = AsyncMock(return_value=True)
    response = _make_html_response("https://example.com/page")
    extract = MagicMock(return_value=("Title", "Lots of body text"))
    brain_state = {
        "is_valid": True,
        "saved": True,
        "source_url": "https://example.com/page",
    }
    with (
        patch("argos.crawler.add_url._is_safe_url", new=safe_url),
        _patch_robots(True),
        _patch_dedup_lookup(None, new_id),
        _patch_get_with_retry(response),
        patch("argos.crawler.add_url.extract_main_content", new=extract),
        _patch_brain(brain_state),
    ):
        r = await add_url("https://example.com/page", session)

    assert r.status is AddUrlStatus.CREATED
    # Only the entry-point SSRF check should have run when there's no redirect.
    assert safe_url.await_count == 1
