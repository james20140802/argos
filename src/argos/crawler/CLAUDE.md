# Argos Crawler — `src/argos/crawler/`

> Scope: static + dynamic fetchers (Epic 2). Loads when working under
> `src/argos/crawler/`. Root conventions still apply — see repo-root `CLAUDE.md`.
> This file adds only crawler-layer specifics.

## Fetchers

- **Static**: GitHub / HN / arXiv / RSS — `static_fetcher.py`, `arxiv_fetcher.py`,
  `rss_fetcher.py` (+ `httpx`, `readability-lxml`).
- **Dynamic / SPA**: `dynamic_fetcher.py`, `spa_fetcher.py` (Playwright).
- Support: `add_url.py`, `pipeline.py`, `user_agents.py`, `_robots.py`,
  `_og_image.py`, `_html_utils.py`.

Freshly crawled items land in the **`crawl_queue`** staging table (daily-limit
throttle) before the brain pipeline processes them.

## Conventions

- **Rate limiting is mandatory.** User-Agent rotation (`user_agents.py`), exponential
  backoff, and **respect `robots.txt`**.
- **Robots allowlist is deliberately tiny.** `_robots.py` carves out a small set of
  vendor-published **public-API** hosts (currently `hacker-news.firebaseio.com`) whose
  generic `robots.txt` would falsely block documented public endpoints. **Do not
  expand it without a documented public-API contract.**
- **Async-first (inherited).** Fetchers are async (`httpx`); never block the loop.
