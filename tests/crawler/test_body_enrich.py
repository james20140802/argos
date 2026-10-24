from __future__ import annotations

import pytest

from argos.crawler import body_enrich
from argos.crawler._robots import RobotsDisallowed
from argos.crawler.body_enrich import HostBreaker, enrich_bodies

LONG = "Real article sentence. " * 40  # > 500 chars


def _item(url: str, raw: str = "Title\n\nshort rss summary", **extra) -> dict:
    return {"source_url": url, "raw_content": raw, "image_url": None, **extra}


def _fetcher(responses: dict[str, object], calls: list[str]):
    async def fetch(url: str):
        calls.append(url)
        value = responses.get(url)
        if isinstance(value, Exception):
            raise value
        return value

    return fetch


def _page(body: str = LONG, **extra) -> dict:
    return {"title": "Fetched", "raw_content": body, "source_url": "x", **extra}


@pytest.mark.asyncio
async def test_only_short_items_are_fetched():
    calls: list[str] = []
    items = [
        _item("https://a.example/short"),
        _item("https://a.example/long", raw="Title\n\n" + LONG),
    ]
    fetch = _fetcher({"https://a.example/short": _page()}, calls)
    report = await enrich_bodies(items, fetch=fetch)
    assert calls == ["https://a.example/short"]
    assert report.enriched == 1
    assert items[0]["raw_content"] == "Title\n\n" + LONG.strip()
    assert items[1]["raw_content"] == "Title\n\n" + LONG


@pytest.mark.asyncio
async def test_replaces_only_when_longer_and_has_body():
    calls: list[str] = []
    items = [_item("https://a.example/1"), _item("https://b.example/2")]
    fetch = _fetcher(
        {
            "https://a.example/1": _page("error page"),  # 84자짜리 오류 페이지류
            "https://b.example/2": _page("x" * 300),  # 길지만 500 미만
        },
        calls,
    )
    report = await enrich_bodies(items, fetch=fetch)
    assert report.enriched == 0
    assert report.missing == 2
    assert items[0]["raw_content"] == "Title\n\nshort rss summary"
    assert items[1]["raw_content"] == "Title\n\nshort rss summary"


@pytest.mark.asyncio
async def test_fills_image_and_published_at_only_when_empty():
    published = object()
    items = [
        _item("https://a.example/1", _published_at=None),
        _item("https://a.example/2", _published_at="keep", image_url="keep.png"),
    ]
    fetch = _fetcher(
        {
            "https://a.example/1": _page(image_url="new.png", _published_at=published),
            "https://a.example/2": _page(image_url="new.png", _published_at=published),
        },
        [],
    )
    await enrich_bodies(items, fetch=fetch)
    assert items[0]["image_url"] == "new.png"
    assert items[0]["_published_at"] is published
    assert items[1]["image_url"] == "keep.png"
    assert items[1]["_published_at"] == "keep"


@pytest.mark.asyncio
async def test_host_is_skipped_after_two_consecutive_failures():
    calls: list[str] = []
    items = [_item(f"https://blocked.example/{i}") for i in range(5)]
    fetch = _fetcher({}, calls)  # 전부 None (빈 본문)
    report = await enrich_bodies(items, fetch=fetch, concurrency=1)
    assert len(calls) == 2
    assert "blocked.example" in report.skipped_hosts
    assert report.skipped == 3
    assert report.missing == 5


@pytest.mark.asyncio
async def test_success_resets_failure_streak():
    calls: list[str] = []
    urls = [f"https://h.example/{i}" for i in range(4)]
    items = [_item(u) for u in urls]
    fetch = _fetcher({urls[1]: _page()}, calls)  # 실패, 성공, 실패, 실패
    report = await enrich_bodies(items, fetch=fetch, concurrency=1)
    assert calls == urls  # 2연속 실패에 도달하기 전에 끝난다 (마지막 2건이 실패)
    assert report.enriched == 1


@pytest.mark.asyncio
async def test_robots_disallowed_keeps_original_and_counts_as_failure():
    calls: list[str] = []
    items = [_item("https://r.example/1"), _item("https://r.example/2"), _item("https://r.example/3")]
    fetch = _fetcher(
        {
            "https://r.example/1": RobotsDisallowed("https://r.example/1"),
            "https://r.example/2": RobotsDisallowed("https://r.example/2"),
        },
        calls,
    )
    report = await enrich_bodies(items, fetch=fetch, concurrency=1)
    assert len(calls) == 2
    assert all(i["raw_content"] == "Title\n\nshort rss summary" for i in items)
    assert report.enriched == 0


@pytest.mark.asyncio
async def test_unexpected_exception_does_not_escape():
    items = [_item("https://e.example/1")]
    fetch = _fetcher({"https://e.example/1": RuntimeError("boom")}, [])
    report = await enrich_bodies(items, fetch=fetch)
    assert report.enriched == 0 and report.missing == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=1",
        "https://youtu.be/abc",
        "https://twitter.com/x/status/1",
        "https://x.com/x/status/1",
        "https://arxiv.org/abs/2401.00001",
    ],
)
async def test_excluded_hosts_are_not_fetched(url):
    calls: list[str] = []
    items = [_item(url)]
    report = await enrich_bodies(items, fetch=_fetcher({url: _page()}, calls))
    assert calls == []
    assert report.attempted == 0


@pytest.mark.asyncio
async def test_max_attempts_caps_fetches():
    calls: list[str] = []
    items = [_item(f"https://m{i}.example/p") for i in range(5)]
    await enrich_bodies(items, fetch=_fetcher({}, calls), max_attempts=2, concurrency=1)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_report_counts():
    items = [
        _item("https://a.example/ok"),
        _item("https://b.example/bad"),
        _item("https://c.example/long", raw="T\n\n" + LONG),
    ]
    report = await enrich_bodies(
        items, fetch=_fetcher({"https://a.example/ok": _page()}, [])
    )
    assert (report.attempted, report.enriched, report.missing) == (2, 1, 1)


@pytest.mark.asyncio
async def test_default_fetch_resolves_module_attribute(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        body_enrich, "fetch_article", _fetcher({"https://a.example/1": _page()}, calls)
    )
    report = await enrich_bodies([_item("https://a.example/1")])
    assert calls == ["https://a.example/1"] and report.enriched == 1


def test_breaker_threshold():
    b = HostBreaker(threshold=2)
    b.record("h", ok=False)
    assert not b.is_open("h")
    b.record("h", ok=False)
    assert b.is_open("h")
    b.record("other", ok=True)
    assert not b.is_open("other")


def test_accept_replacement_rules():
    from argos.crawler.body_enrich import accept_replacement

    old_bad = "Title\n\nweâ\x80\x99re " + "z" * 600
    new_ok = "Title\n\nwe’re " + "z" * 550
    assert accept_replacement(old_bad, new_ok, "Title")  # 깨진 행은 짧아져도 교체
    assert not accept_replacement("Title\n\n" + LONG, "Title\n\n" + "y" * 100, "Title")
    assert not accept_replacement("Title\n\nshort", "Title\n\n" + "w\x85" * 400, "Title")
