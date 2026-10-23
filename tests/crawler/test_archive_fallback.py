"""Wayback 보충 (ARG-309 결정 1′=B) — 네트워크 없이 respx로만."""

from __future__ import annotations

import inspect
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx

from argos.crawler import archive_fallback, body_enrich, pipeline
from argos.crawler.archive_fallback import (
    ArchiveFetcher,
    RequestThrottle,
    extract_article_text,
    is_archive_eligible,
    snapshot_url,
)
from argos.crawler.body_backfill import BodyRow, refetch_and_apply

ARTICLE_URL = "https://openai.com/index/deliberative-alignment/"
TS = "20250102030405"
SENTENCE = "Deliberative alignment teaches models to reason about safety specs. "
LONG_ARTICLE = SENTENCE * 20  # > 500 chars
HUGE_ARTICLE = SENTENCE * 100  # 실제 <article>은 readability가 고른 블록보다 훨씬 길다


def _availability(ts: str | None = TS, status: str = "200", available: bool = True) -> dict:
    if ts is None:
        return {"archived_snapshots": {}}
    return {
        "archived_snapshots": {
            "closest": {
                "available": available,
                "status": status,
                "timestamp": ts,
                "url": f"http://web.archive.org/web/{ts}/{ARTICLE_URL}",
            }
        }
    }


def _snapshot_html(article_text: str = HUGE_ARTICLE) -> str:
    # readability가 고를 만한 큰 블록(사이드바 링크 모음)과 실제 <article>을 함께 둔다.
    sidebar = "<div>" + "".join(f"<p>Related story number {i} about other things entirely here.</p>" for i in range(15)) + "</div>"
    return (
        "<html><head><title>Deliberative alignment</title></head><body>"
        f"<main><article><p>{article_text}</p></article></main>{sidebar}</body></html>"
    )


def _fetcher() -> ArchiveFetcher:
    return ArchiveFetcher(throttle=RequestThrottle(0))


# --- 대상 판정 --------------------------------------------------------------


def test_eligibility_only_openai_article_urls():
    assert is_archive_eligible(ARTICLE_URL)
    assert is_archive_eligible("https://www.openai.com/news/some-post")
    assert not is_archive_eligible("https://openai.com/")
    assert not is_archive_eligible("https://huggingface.co/blog/x")
    assert not is_archive_eligible("https://developers.openai.com/blog/x")
    assert not is_archive_eligible("https://community.openai.com/t/x/1")
    assert not is_archive_eligible("https://notopenai.com/x")
    assert archive_fallback.ARCHIVE_FALLBACK_HOSTS == frozenset({"openai.com"})


def test_snapshot_url_uses_raw_id_flag():
    assert snapshot_url(TS, ARTICLE_URL) == f"https://web.archive.org/web/{TS}id_/{ARTICLE_URL}"


# --- 추출 -------------------------------------------------------------------


def test_extract_falls_back_to_article_when_readability_is_short(monkeypatch):
    # readability가 엉뚱한 블록(실측 1,645자)을 고른 상황을 고정한다.
    monkeypatch.setattr(
        archive_fallback, "extract_main_content", lambda html: ("Deliberative alignment", "Related story. " * 20)
    )
    title, body = extract_article_text(_snapshot_html())
    assert title
    assert "Deliberative alignment teaches models" in body
    assert len(body) >= 500


def test_extract_keeps_readability_when_it_is_not_much_shorter(monkeypatch):
    picked = "Picked by readability. " * 100
    monkeypatch.setattr(archive_fallback, "extract_main_content", lambda html: ("t", picked))
    _title, body = extract_article_text(_snapshot_html(SENTENCE * 20))
    assert body == picked


# --- 가져오기 ---------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_uses_availability_then_id_snapshot():
    lookup = respx.get("https://archive.org/wayback/available").mock(
        return_value=httpx.Response(200, json=_availability())
    )
    snap = respx.get(snapshot_url(TS, ARTICLE_URL)).mock(
        return_value=httpx.Response(200, html=_snapshot_html())
    )
    async with _fetcher() as archive:
        result = await archive.fetch(ARTICLE_URL)
    assert lookup.called and snap.called
    assert lookup.calls.last.request.url.params["url"] == ARTICLE_URL
    assert result is not None
    assert result["archived_at"] == TS
    assert "Deliberative alignment teaches models" in result["raw_content"]


@pytest.mark.asyncio
@respx.mock
async def test_no_snapshot_returns_none_and_skips_snapshot_request():
    respx.get("https://archive.org/wayback/available").mock(
        return_value=httpx.Response(200, json=_availability(None))
    )
    async with _fetcher() as archive:
        assert await archive.fetch(ARTICLE_URL) is None
    assert len(respx.calls) == 1


@pytest.mark.asyncio
@respx.mock
async def test_non_200_snapshot_is_ignored():
    respx.get("https://archive.org/wayback/available").mock(
        return_value=httpx.Response(200, json=_availability(status="403"))
    )
    async with _fetcher() as archive:
        assert await archive.fetch(ARTICLE_URL) is None
    assert len(respx.calls) == 1


@pytest.mark.asyncio
@respx.mock
async def test_ineligible_host_makes_no_requests():
    async with _fetcher() as archive:
        assert await archive.fetch("https://huggingface.co/blog/x") is None
        assert await archive.fetch("https://openai.com/") is None
    assert len(respx.calls) == 0


@pytest.mark.asyncio
@respx.mock
async def test_snapshot_redirect_stays_on_archive_hosts():
    respx.get("https://archive.org/wayback/available").mock(
        return_value=httpx.Response(200, json=_availability())
    )
    respx.get(snapshot_url(TS, ARTICLE_URL)).mock(
        return_value=httpx.Response(
            302, headers={"location": f"https://web.archive.org/web/20250102030499id_/{ARTICLE_URL}"}
        )
    )
    respx.get(f"https://web.archive.org/web/20250102030499id_/{ARTICLE_URL}").mock(
        return_value=httpx.Response(200, html=_snapshot_html())
    )
    async with _fetcher() as archive:
        assert await archive.fetch(ARTICLE_URL) is not None


@pytest.mark.asyncio
@respx.mock
async def test_redirect_to_other_host_is_refused():
    respx.get("https://archive.org/wayback/available").mock(
        return_value=httpx.Response(200, json=_availability())
    )
    respx.get(snapshot_url(TS, ARTICLE_URL)).mock(
        return_value=httpx.Response(302, headers={"location": "https://evil.example/x"})
    )
    evil = respx.get("https://evil.example/x").mock(return_value=httpx.Response(200, text="x"))
    async with _fetcher() as archive:
        assert await archive.fetch(ARTICLE_URL) is None
    assert not evil.called


@pytest.mark.asyncio
@respx.mock
async def test_archive_disabled_after_repeated_lookup_failures():
    respx.get("https://archive.org/wayback/available").mock(return_value=httpx.Response(503))
    async with _fetcher() as archive:
        for _ in range(3):
            assert await archive.fetch(ARTICLE_URL) is None
        assert archive.disabled
        before = len(respx.calls)
        assert await archive.fetch(ARTICLE_URL) is None
        assert len(respx.calls) == before


@pytest.mark.asyncio
@respx.mock
async def test_429_disables_archive_immediately_without_retrying():
    route = respx.get("https://archive.org/wayback/available").mock(
        return_value=httpx.Response(429, headers={"retry-after": "60"})
    )
    async with _fetcher() as archive:
        assert await archive.fetch(ARTICLE_URL) is None
        assert archive.disabled
        assert await archive.fetch(ARTICLE_URL) is None
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_robots_disallow_stops_the_request(monkeypatch):
    monkeypatch.setattr(
        "argos.crawler.static_fetcher.is_robots_allowed", AsyncMock(return_value=False)
    )
    async with _fetcher() as archive:
        assert await archive.fetch(ARTICLE_URL) is None
    assert len(respx.calls) == 0


# --- 속도 제한 --------------------------------------------------------------


@pytest.mark.asyncio
async def test_throttle_spaces_requests_by_min_interval():
    now = [100.0]
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)
        now[0] += d

    t = RequestThrottle(1.0, clock=lambda: now[0], sleep=fake_sleep)
    await t.wait()  # 첫 요청은 기다리지 않는다
    await t.wait()
    now[0] += 0.4
    await t.wait()
    assert sleeps == [pytest.approx(1.0), pytest.approx(0.6)]


@pytest.mark.asyncio
@respx.mock
async def test_every_archive_request_goes_through_the_throttle():
    respx.get("https://archive.org/wayback/available").mock(
        return_value=httpx.Response(200, json=_availability())
    )
    respx.get(snapshot_url(TS, ARTICLE_URL)).mock(
        return_value=httpx.Response(200, html=_snapshot_html())
    )
    throttle = RequestThrottle(0)
    waits = 0
    original = throttle.wait

    async def counting_wait():
        nonlocal waits
        waits += 1
        await original()

    throttle.wait = counting_wait  # type: ignore[method-assign]
    async with ArchiveFetcher(throttle=throttle) as archive:
        await archive.fetch(ARTICLE_URL)
    assert waits == len(respx.calls) == 2


# --- refetch-bodies 통합 ------------------------------------------------------


def _row(url: str) -> BodyRow:
    return BodyRow(uuid.uuid4(), "T", url, "T\n\nshort rss summary")


def _session() -> MagicMock:
    s = MagicMock()
    s.execute = AsyncMock()
    s.commit = AsyncMock()
    return s


class _FakeArchive:
    def __init__(self, body: str | None = LONG_ARTICLE):
        self.calls: list[str] = []
        self.disabled = False
        self._body = body

    async def fetch(self, url: str):
        self.calls.append(url)
        return {"raw_content": self._body} if self._body else None


@pytest.mark.asyncio
async def test_refetch_uses_archive_only_for_eligible_hosts_when_live_fails():
    openai = _row(ARTICLE_URL)
    other = _row("https://blocked.example/post")
    archive = _FakeArchive()

    async def live(url):
        return None  # 원 사이트는 막혀 있다

    summary = await refetch_and_apply(
        _session(), [openai, other], fetch=live, archive=archive, concurrency=1
    )
    assert archive.calls == [ARTICLE_URL]
    assert summary.updated == 1
    assert summary.via_archive == 1


@pytest.mark.asyncio
async def test_refetch_prefers_live_page_over_archive():
    archive = _FakeArchive()

    async def live(url):
        return {"raw_content": LONG_ARTICLE}

    summary = await refetch_and_apply(_session(), [_row(ARTICLE_URL)], fetch=live, archive=archive)
    assert summary.updated == 1 and summary.via_archive == 0
    assert archive.calls == []


@pytest.mark.asyncio
async def test_refetch_goes_straight_to_archive_once_host_breaker_is_open():
    rows = [_row(f"https://openai.com/index/post-{i}") for i in range(5)]
    archive = _FakeArchive()
    live_calls: list[str] = []

    async def live(url):
        live_calls.append(url)
        return None

    summary = await refetch_and_apply(_session(), rows, fetch=live, archive=archive, concurrency=1)
    assert len(live_calls) == 2  # 차단기 임계값
    assert len(archive.calls) == 5  # 나머지도 보관본으로는 시도한다
    assert summary.updated == 5 and summary.via_archive == 5 and summary.skipped == 0


@pytest.mark.asyncio
async def test_refetch_without_archive_keeps_blocked_rows_as_missing():
    rows = [_row(f"https://openai.com/index/post-{i}") for i in range(4)]

    async def live(url):
        return None

    summary = await refetch_and_apply(_session(), rows, fetch=live, concurrency=1)
    assert summary.updated == 0 and summary.skipped == 2


@pytest.mark.asyncio
async def test_archive_body_must_still_pass_the_length_check():
    archive = _FakeArchive(body="tiny snapshot")

    async def live(url):
        return None

    summary = await refetch_and_apply(_session(), [_row(ARTICLE_URL)], fetch=live, archive=archive)
    assert summary.updated == 0 and summary.via_archive == 0


# --- 매일 실행 경로에는 절대 없다 (결정 1′ 경계) -----------------------------------


def test_daily_pipeline_never_references_the_archive():
    assert "archive_fallback" not in inspect.getsource(pipeline)
    assert "ArchiveFetcher" not in inspect.getsource(pipeline)
    assert "archive_fallback" not in inspect.getsource(body_enrich)


@pytest.mark.asyncio
async def test_enrich_bodies_never_touches_the_archive(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("daily path must not use the archive")

    monkeypatch.setattr(archive_fallback, "ArchiveFetcher", boom)
    items = [{"source_url": ARTICLE_URL, "raw_content": "T\n\nshort", "image_url": None}]

    async def live(url):
        return None

    report = await body_enrich.enrich_bodies(items, fetch=live)
    assert report.enriched == 0
