"""Epic-2 크롤러 패키지. 정적/동적 수집기와 파이프라인 오케스트레이터를 노출한다."""

from argos.crawler.dynamic_fetcher import fetch_dynamic_page
from argos.crawler.pipeline import run_full_crawl, run_static_pipeline
from argos.crawler.static_fetcher import (
    fetch_github_trending,
    fetch_hackernews_top,
    filter_duplicate_urls,
)
from argos.crawler.user_agents import USER_AGENTS, random_user_agent

__all__ = [
    "USER_AGENTS",
    "fetch_dynamic_page",
    "fetch_github_trending",
    "fetch_hackernews_top",
    "filter_duplicate_urls",
    "random_user_agent",
    "run_full_crawl",
    "run_static_pipeline",
]
