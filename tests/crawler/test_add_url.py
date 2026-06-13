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
    """is_valid=True + saved=False AND no row in DB => true save failure."""
    session = _make_session()
    brain_state = {
        "is_valid": True,
        "saved": False,
        "source_url": "https://example.com/page",
    }
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        # Pre-fetch dedup: None. Post-save re-check: None (still no row) => ERROR.
        _patch_dedup_lookup(None, None),
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


async def test_save_skipped_with_existing_row_returns_duplicate() -> None:
    """is_valid=True + saved=False BUT the URL is now present in tech_items =>
    benign race (another worker inserted the same source_url between our
    pre-fetch dedup check and save_node's lookup). Must surface as DUPLICATE,
    not ERROR (otherwise /argos add exits non-zero on a benign duplicate).
    """
    session = _make_session()
    race_winner_id = uuid.uuid4()
    brain_state = {
        "is_valid": True,
        "saved": False,
        "source_url": "https://example.com/page",
    }
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        # Pre-fetch dedup: None. Post-save re-check: race winner's UUID.
        _patch_dedup_lookup(None, race_winner_id),
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

    assert r.status is AddUrlStatus.DUPLICATE
    assert r.tech_item_id == race_winner_id
    assert r.reason and "race" in r.reason.lower()


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
# SSRF: per-hop redirect validation (Codex review on PR #67, follow-up)
#
# Previous fix only blocked ingestion of post-redirect bodies, leaving the
# network request to the private host as an SSRF primitive. The current
# implementation MUST disable httpx auto-redirect and validate each
# `Location` target via `_is_safe_url` BEFORE issuing the next request.
# ---------------------------------------------------------------------------


def _patch_get_with_retry_sequence(*responses):
    """Patch ``_get_with_retry`` with a sequenced AsyncMock so each successive
    call returns the next response. Returns the mock so tests can assert on
    ``.call_args_list`` / ``.await_count``."""
    mock = AsyncMock(side_effect=list(responses))
    return mock, patch("argos.crawler.add_url._get_with_retry", new=mock)


def _make_html_response(
    final_url: str, body: str = "<html><body><p>x</p></body></html>"
) -> MagicMock:
    response = MagicMock()
    response.url = final_url
    response.text = body
    response.status_code = 200
    response.headers = {"content-type": "text/html"}
    return response


def _make_redirect_response(
    location: str, status_code: int = 302
) -> MagicMock:
    """A bare 3xx whose Location header points at ``location``. The body /
    content-type / .url are irrelevant — the safe-fetch loop only reads
    status_code + Location."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"location": location}
    response.text = ""
    return response


async def test_static_fetch_revalidates_redirect_blocks_private_target() -> None:
    """A redirect Location pointing at a private host MUST NOT trigger a second
    network request. The SSRF check has to run BEFORE the next GET.
    """
    session = _make_session()
    # First _is_safe_url is the add_url entry-point check (ok).
    # Second is the per-hop check on the redirect target (blocked).
    safe_url = AsyncMock(side_effect=[True, False])
    redirect = _make_redirect_response("http://10.0.0.1/internal")
    # Only ONE response should ever be consumed — if the SSRF gate fails open
    # and a second request is issued, side_effect=[redirect] will raise
    # StopAsyncIteration and the test will fail loudly.
    get_mock, get_patch = _patch_get_with_retry_sequence(redirect)
    extract = MagicMock(return_value=("Title", "Lots of body text"))
    fetch_dynamic = AsyncMock(return_value=None)
    with (
        patch("argos.crawler.add_url._is_safe_url", new=safe_url),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        get_patch,
        patch("argos.crawler.add_url.extract_main_content", new=extract),
        patch("argos.crawler.add_url.fetch_dynamic_page", new=fetch_dynamic),
    ):
        r = await add_url("https://example.com/start", session)

    # Outcome: REJECTED with an SSRF-specific reason, NO fallback to dynamic.
    assert r.status is AddUrlStatus.REJECTED
    assert r.reason and "ssrf" in r.reason.lower()
    fetch_dynamic.assert_not_awaited()
    # CRITICAL: only the initial request was issued. The private host was
    # never requested.
    assert get_mock.await_count == 1
    called_urls = [call.args[1] for call in get_mock.await_args_list]
    assert "10.0.0.1" not in " ".join(called_urls)


async def test_static_fetch_revalidates_redirect_blocks_robots_disallowed_target() -> None:
    """A redirect to a robots-disallowed host is rejected by ``_get_with_retry``
    on the second hop (it calls ``is_robots_allowed`` before issuing the GET).
    """
    session = _make_session()
    # robots-allowed at the entry point (initial GET succeeds), then the
    # second invocation of _get_with_retry (against the redirect target)
    # raises RobotsDisallowed before issuing any network request.
    from argos.crawler.add_url import RobotsDisallowed as _Robots

    redirect = _make_redirect_response("https://forbidden.example.com/blocked")
    get_mock = AsyncMock(
        side_effect=[redirect, _Robots("https://forbidden.example.com/blocked")]
    )
    safe_url = AsyncMock(return_value=True)
    extract = MagicMock(return_value=("Title", "Body"))
    fetch_dynamic = AsyncMock(return_value=None)
    with (
        patch("argos.crawler.add_url._is_safe_url", new=safe_url),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        patch("argos.crawler.add_url._get_with_retry", new=get_mock),
        patch("argos.crawler.add_url.extract_main_content", new=extract),
        patch("argos.crawler.add_url.fetch_dynamic_page", new=fetch_dynamic),
    ):
        r = await add_url("https://example.com/start", session)

    # Surface as REJECTED with a robots-flavored reason.
    assert r.status is AddUrlStatus.REJECTED
    assert r.reason and "robots" in r.reason.lower()
    fetch_dynamic.assert_not_awaited()


async def test_static_fetch_relative_redirect_resolves_and_validates() -> None:
    """A relative Location like ``/internal`` resolves against the current URL
    and is then SSRF-validated. If the resolved URL is on the same (safe) host
    the chain proceeds; the test asserts the second GET targets the resolved
    absolute URL.
    """
    session = _make_session()
    new_id = uuid.uuid4()
    # First two calls to _is_safe_url: entry-point (ok), per-hop (ok).
    safe_url = AsyncMock(return_value=True)
    redirect = _make_redirect_response("/page")  # relative
    final = _make_html_response("https://example.com/page")
    get_mock = AsyncMock(side_effect=[redirect, final])
    extract = MagicMock(return_value=("Title", "Lots of body text"))
    brain_state = {
        "is_valid": True,
        "saved": True,
        "source_url": "https://example.com/page",
    }
    with (
        patch("argos.crawler.add_url._is_safe_url", new=safe_url),
        _patch_robots(True),
        _patch_dedup_lookup(None, None, new_id),
        patch("argos.crawler.add_url._get_with_retry", new=get_mock),
        patch("argos.crawler.add_url.extract_main_content", new=extract),
        _patch_brain(brain_state),
    ):
        r = await add_url("https://example.com/start", session)

    assert r.status is AddUrlStatus.CREATED
    # Two GETs: original then resolved absolute redirect target.
    assert get_mock.await_count == 2
    called_urls = [call.args[1] for call in get_mock.await_args_list]
    assert called_urls[0] == "https://example.com/start"
    assert called_urls[1] == "https://example.com/page"


async def test_static_fetch_rejects_disallowed_scheme_in_redirect() -> None:
    """A redirect to ``file://`` or similar non-http(s) scheme is rejected by
    the scheme check before SSRF resolution and before any further request.
    """
    session = _make_session()
    safe_url = AsyncMock(return_value=True)
    redirect = _make_redirect_response("file:///etc/passwd")
    get_mock = AsyncMock(side_effect=[redirect])
    fetch_dynamic = AsyncMock(return_value=None)
    with (
        patch("argos.crawler.add_url._is_safe_url", new=safe_url),
        _patch_robots(True),
        _patch_dedup_lookup(None),
        patch("argos.crawler.add_url._get_with_retry", new=get_mock),
        patch("argos.crawler.add_url.fetch_dynamic_page", new=fetch_dynamic),
    ):
        r = await add_url("https://example.com/start", session)

    assert r.status is AddUrlStatus.REJECTED
    assert r.reason and "scheme" in r.reason.lower()
    assert get_mock.await_count == 1
    fetch_dynamic.assert_not_awaited()


async def test_add_url_threads_published_at_from_fetch_to_brain() -> None:
    """_published_at from _fetch_url_content must be forwarded to run_brain_pipeline."""
    import datetime as _dt

    session = _make_session()
    new_id = uuid.uuid4()
    pub = _dt.datetime(2023, 11, 1, 9, 0, 0, tzinfo=_dt.timezone.utc)
    brain_state = {
        "is_valid": True,
        "saved": True,
        "source_url": "https://example.com/page",
    }
    brain_mock = AsyncMock(return_value=brain_state)
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None, new_id),
        _patch_fetch(
            {
                "title": "Article",
                "raw_content": "Substantive body text.",
                "source_url": "https://example.com/page",
                "_published_at": pub,
            }
        ),
        patch("argos.crawler.add_url.run_brain_pipeline", new=brain_mock),
    ):
        r = await add_url("https://example.com/page", session)

    assert r.status is AddUrlStatus.CREATED
    brain_mock.assert_awaited_once()
    assert brain_mock.call_args.kwargs.get("published_at") == pub


async def test_static_fetch_extracts_published_at_from_html() -> None:
    """Static path must parse _published_at from OpenGraph meta so old articles
    are not treated as current in the briefing lookback filter."""
    import datetime as _dt

    session = _make_session()
    new_id = uuid.uuid4()
    pub_html = (
        '<html><head>'
        '<meta property="article:published_time" content="2022-06-15T10:00:00+00:00"/>'
        '</head><body><p>article body</p></body></html>'
    )
    response = _make_html_response("https://example.com/old-article", body=pub_html)
    get_mock = AsyncMock(side_effect=[response])
    extract = MagicMock(return_value=("Old Article", "article body"))
    brain_state = {
        "is_valid": True,
        "saved": True,
        "source_url": "https://example.com/old-article",
    }
    brain_mock = AsyncMock(return_value=brain_state)
    with (
        patch("argos.crawler.add_url._is_safe_url", new=AsyncMock(return_value=True)),
        _patch_robots(True),
        _patch_dedup_lookup(None, new_id),
        patch("argos.crawler.add_url._get_with_retry", new=get_mock),
        patch("argos.crawler.add_url.extract_main_content", new=extract),
        patch("argos.crawler.add_url.run_brain_pipeline", new=brain_mock),
    ):
        r = await add_url("https://example.com/old-article", session)

    assert r.status is AddUrlStatus.CREATED
    expected = _dt.datetime(2022, 6, 15, 10, 0, 0, tzinfo=_dt.timezone.utc)
    assert brain_mock.call_args.kwargs.get("published_at") == expected


async def test_static_fetch_no_redirect_issues_single_request() -> None:
    """A 200 OK response is returned directly with no follow-up GET."""
    session = _make_session()
    new_id = uuid.uuid4()
    safe_url = AsyncMock(return_value=True)
    response = _make_html_response("https://example.com/page")
    get_mock = AsyncMock(side_effect=[response])
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
        patch("argos.crawler.add_url._get_with_retry", new=get_mock),
        patch("argos.crawler.add_url.extract_main_content", new=extract),
        _patch_brain(brain_state),
    ):
        r = await add_url("https://example.com/page", session)

    assert r.status is AddUrlStatus.CREATED
    # Only the entry-point SSRF check should have run when there's no redirect.
    assert safe_url.await_count == 1
    assert get_mock.await_count == 1


# ---------------------------------------------------------------------------
# ARG-152: add_url forwards image_url from fetcher to brain pipeline
# ---------------------------------------------------------------------------


async def test_add_url_forwards_image_url_to_brain_pipeline() -> None:
    """image_url from the fetcher must be forwarded as a kwarg to run_brain_pipeline."""
    session = _make_session()
    new_id = uuid.uuid4()
    brain_state = {
        "is_valid": True,
        "saved": True,
        "source_url": "https://example.com/page",
    }
    brain_mock = AsyncMock(return_value=brain_state)
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None, new_id),
        _patch_fetch(
            {
                "title": "t",
                "raw_content": "body",
                "source_url": "https://example.com/page",
                "image_url": "https://cdn.example.com/cover.jpg",
            }
        ),
        patch("argos.crawler.add_url.run_brain_pipeline", new=brain_mock),
    ):
        r = await add_url("https://example.com/page", session)

    assert r.status is AddUrlStatus.CREATED
    brain_mock.assert_awaited_once()
    kwargs = brain_mock.call_args.kwargs
    assert kwargs.get("image_url") == "https://cdn.example.com/cover.jpg"


async def test_add_url_forwards_none_image_url_when_fetcher_returned_none() -> None:
    """A fetcher dict without image_url surfaces None to run_brain_pipeline."""
    session = _make_session()
    new_id = uuid.uuid4()
    brain_state = {
        "is_valid": True,
        "saved": True,
        "source_url": "https://example.com/page",
    }
    brain_mock = AsyncMock(return_value=brain_state)
    with (
        _patch_safe_url(True),
        _patch_robots(True),
        _patch_dedup_lookup(None, new_id),
        _patch_fetch(
            {
                "title": "t",
                "raw_content": "body",
                "source_url": "https://example.com/page",
            }
        ),
        patch("argos.crawler.add_url.run_brain_pipeline", new=brain_mock),
    ):
        await add_url("https://example.com/page", session)

    brain_mock.assert_awaited_once()
    assert brain_mock.call_args.kwargs.get("image_url") is None


async def test_fetch_url_content_extracts_image_url_via_static_path() -> None:
    """_fetch_url_content's static path calls extract_og_image on response.text."""
    import httpx

    from argos.crawler.add_url import _fetch_url_content

    html = (
        '<html><head>'
        '<meta property="og:image" content="https://cdn.example.com/cover.jpg">'
        '</head><body><article><h1>Headline</h1>'
        '<p>Substantive body text.</p></article></body></html>'
    )

    async def _fake_safe_fetch(url: str):
        return httpx.Response(
            200,
            text=html,
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", url),
        )

    with patch("argos.crawler.add_url._safe_static_fetch", new=_fake_safe_fetch):
        result = await _fetch_url_content("https://example.com/page")

    assert result is not None
    assert result["image_url"] == "https://cdn.example.com/cover.jpg"


async def test_fetch_url_content_image_url_is_none_when_no_og_image() -> None:
    """Static path returns image_url=None when response HTML has no og:image."""
    import httpx

    from argos.crawler.add_url import _fetch_url_content

    html = (
        "<html><body><article><h1>Headline</h1>"
        "<p>Substantive body text.</p></article></body></html>"
    )

    async def _fake_safe_fetch(url: str):
        return httpx.Response(
            200,
            text=html,
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", url),
        )

    with patch("argos.crawler.add_url._safe_static_fetch", new=_fake_safe_fetch):
        result = await _fetch_url_content("https://example.com/page")

    assert result is not None
    assert result.get("image_url") is None


async def test_fetch_url_content_dynamic_fallback_preserves_image_url() -> None:
    """Dynamic-fetcher fallback path forwards image_url from fetch_dynamic_page."""
    from argos.crawler.add_url import _fetch_url_content

    async def _fake_safe_fetch(url: str):
        return None  # force dynamic fallback

    async def _fake_dynamic(url: str):
        return {
            "title": "Dynamic title",
            "source_url": url,
            "raw_content": "Dynamic body content.",
            "image_url": "https://cdn.example.com/spa.jpg",
            "_published_at": None,
        }

    with (
        patch("argos.crawler.add_url._safe_static_fetch", new=_fake_safe_fetch),
        patch("argos.crawler.add_url.fetch_dynamic_page", new=_fake_dynamic),
    ):
        result = await _fetch_url_content("https://example.com/spa")

    assert result is not None
    assert result["image_url"] == "https://cdn.example.com/spa.jpg"
