from __future__ import annotations

import asyncio

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.crawler.user_agents import random_user_agent
from argos.models.tech_item import TechItem

_HN_CONCURRENCY = 8


async def fetch_github_trending(
    client: httpx.AsyncClient,
    language: str | None = None,
) -> list[dict]:
    url = f"https://github.com/trending/{language}" if language else "https://github.com/trending"
    response = await client.get(url, headers={"User-Agent": random_user_agent()})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    items: list[dict] = []

    for article in soup.select("article.Box-row"):
        anchor = article.select_one("h2.h3.lh-condensed a")
        if not anchor:
            continue
        href = (anchor.get("href") or "").strip()
        if not href.startswith("/"):
            continue

        source_url = f"https://github.com{href}"
        title = anchor.get_text(strip=True)
        description_tag = article.select_one("p.col-9.color-fg-muted") or article.find("p")
        raw_content = description_tag.get_text(strip=True) if description_tag else title
        items.append({"title": title, "source_url": source_url, "raw_content": raw_content})

    return items


async def fetch_hackernews_top(
    client: httpx.AsyncClient,
    limit: int = 30,
) -> list[dict]:
    response = await client.get(
        "https://hacker-news.firebaseio.com/v0/topstories.json",
        headers={"User-Agent": random_user_agent()},
    )
    response.raise_for_status()
    top_ids: list[int] = response.json()[:limit]

    semaphore = asyncio.Semaphore(_HN_CONCURRENCY)

    async def _fetch_item(item_id: int) -> dict | None:
        async with semaphore:
            try:
                r = await client.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json",
                    headers={"User-Agent": random_user_agent()},
                )
                r.raise_for_status()
            except httpx.HTTPError:
                return None
            data = r.json() or {}
            url = data.get("url") or f"https://news.ycombinator.com/item?id={item_id}"
            title = data.get("title", "")
            text = data.get("text", "")
            raw_content = f"{title} {text}".strip() if text else title
            return {"title": title, "source_url": url, "raw_content": raw_content}

    results = await asyncio.gather(*(_fetch_item(i) for i in top_ids))
    return [item for item in results if item is not None]


async def filter_duplicate_urls(
    session: AsyncSession,
    items: list[dict],
) -> list[dict]:
    if not items:
        return []

    candidate_urls = [item["source_url"] for item in items]
    stmt = select(TechItem.source_url).where(TechItem.source_url.in_(candidate_urls))
    result = await session.execute(stmt)
    existing_urls: set[str] = set(result.scalars().all())

    deduped: list[dict] = []
    seen: set[str] = set(existing_urls)
    for item in items:
        url = item["source_url"]
        if url in seen:
            continue
        seen.add(url)
        deduped.append(item)
    return deduped
