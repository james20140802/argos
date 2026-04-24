from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from argos.crawler import pipeline


@pytest.fixture
def patched_static(mocker):
    gh = [
        {"title": "gh-1", "source_url": "https://github.com/a/b", "raw_content": "x"},
        {"title": "gh-2", "source_url": "https://github.com/c/d", "raw_content": "y"},
    ]
    hn = [
        {"title": "hn-1", "source_url": "https://hn.com/1", "raw_content": "p"},
        {"title": "hn-2", "source_url": "https://hn.com/2", "raw_content": "q"},
    ]
    mocker.patch(
        "argos.crawler.pipeline.fetch_github_trending",
        new=AsyncMock(return_value=gh),
    )
    mocker.patch(
        "argos.crawler.pipeline.fetch_hackernews_top",
        new=AsyncMock(return_value=hn),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )
    return gh, hn


async def test_run_static_pipeline_combines_sources(patched_static) -> None:
    gh, hn = patched_static
    session = AsyncMock()

    result = await pipeline.run_static_pipeline(session)

    assert len(result) == len(gh) + len(hn)
    urls = {item["source_url"] for item in result}
    for source in (*gh, *hn):
        assert source["source_url"] in urls


async def test_run_full_crawl_without_dynamic_urls_equals_static(patched_static) -> None:
    session = AsyncMock()

    static_result = await pipeline.run_static_pipeline(session)
    full_result = await pipeline.run_full_crawl(session, dynamic_urls=None)

    assert full_result == static_result


async def test_run_static_pipeline_isolates_source_failures(mocker) -> None:
    hn = [
        {"title": "hn-1", "source_url": "https://hn.com/1", "raw_content": "p"},
    ]
    mocker.patch(
        "argos.crawler.pipeline.fetch_github_trending",
        new=AsyncMock(side_effect=RuntimeError("github is down")),
    )
    mocker.patch(
        "argos.crawler.pipeline.fetch_hackernews_top",
        new=AsyncMock(return_value=hn),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )

    result = await pipeline.run_static_pipeline(AsyncMock())

    assert result == hn


async def test_run_full_crawl_appends_dynamic_results(patched_static, mocker) -> None:
    dynamic_item = {
        "title": "dyn",
        "source_url": "https://example.com/article",
        "raw_content": "body",
    }
    mocker.patch(
        "argos.crawler.pipeline.fetch_dynamic_page",
        new=AsyncMock(return_value=dynamic_item),
    )
    session = AsyncMock()

    result = await pipeline.run_full_crawl(
        session, dynamic_urls=["https://example.com/article"]
    )

    gh, hn = patched_static
    assert len(result) == len(gh) + len(hn) + 1
    assert result[-1] == dynamic_item


async def test_run_static_pipeline_reraises_cancelled_error(mocker) -> None:
    mocker.patch(
        "argos.crawler.pipeline.fetch_github_trending",
        new=AsyncMock(side_effect=asyncio.CancelledError()),
    )
    mocker.patch(
        "argos.crawler.pipeline.fetch_hackernews_top",
        new=AsyncMock(return_value=[]),
    )
    mocker.patch(
        "argos.crawler.pipeline.filter_duplicate_urls",
        new=AsyncMock(side_effect=lambda _session, items: items),
    )

    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_static_pipeline(AsyncMock())


async def test_run_full_crawl_reraises_cancelled_error_from_dynamic(patched_static, mocker) -> None:
    mocker.patch(
        "argos.crawler.pipeline.fetch_dynamic_page",
        new=AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await pipeline.run_full_crawl(
            AsyncMock(), dynamic_urls=["https://example.com/x"]
        )
