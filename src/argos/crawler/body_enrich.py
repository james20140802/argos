"""본문이 짧은 항목은 기사 페이지에서 본문을 가져와 보강한다 (ARG-309).

RSS 피드는 제목과 한 줄 요약만 줘서 ``raw_content``가 얇다. 오늘 처리할
배치에 한해 기사 페이지를 열어 본문으로 교체한다. 가져오기 경로는
``argos add``와 같다 (robots·SSRF·리다이렉트 재검사·UA 순환·지수 백오프).

봇 차단(Cloudflare 챌린지 등)을 푸는 시도는 하지 않는다. 같은 호스트가
연속 :data:`HOST_FAILURE_THRESHOLD`회 실패하면 그 실행에서는 그 호스트를
건너뛴다.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from argos.brain.titles import derive_title
from argos.crawler.add_url import fetch_article
from argos.crawler.body_quality import BODY_MIN_CHARS, body_text, has_body, is_mojibake

__all__ = [
    "EXCLUDED_HOSTS",
    "EnrichReport",
    "HOST_FAILURE_THRESHOLD",
    "HostBreaker",
    "accept_replacement",
    "build_raw_content",
    "enrich_bodies",
    "host_of",
    "is_excluded_host",
]

logger = logging.getLogger(__name__)

HOST_FAILURE_THRESHOLD = 2
_CONCURRENCY = 3  # crawler.pipeline._DYNAMIC_CONCURRENCY와 같은 값

#: 영상·트윗은 본문이 없는 게 정상이고, arXiv는 초록이 본문이다.
EXCLUDED_HOSTS = frozenset({"youtube.com", "youtu.be", "twitter.com", "x.com", "arxiv.org"})

FetchFn = Callable[[str], Awaitable[dict | None]]


def host_of(url: str) -> str:
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def is_excluded_host(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in EXCLUDED_HOSTS)


class HostBreaker:
    """호스트별 연속 실패 차단기. 성공하면 연속 횟수가 0으로 돌아간다."""

    def __init__(self, threshold: int = HOST_FAILURE_THRESHOLD) -> None:
        self.threshold = threshold
        self._streak: dict[str, int] = {}

    def record(self, host: str, *, ok: bool) -> None:
        self._streak[host] = 0 if ok else self._streak.get(host, 0) + 1

    def is_open(self, host: str) -> bool:
        return self._streak.get(host, 0) >= self.threshold


@dataclass
class EnrichReport:
    attempted: int = 0
    enriched: int = 0
    skipped: int = 0  # 차단기로 건너뛴 건수
    missing: int = 0  # 끝난 뒤에도 본문이 없는 항목 수 (배치 전체 기준)
    skipped_hosts: set[str] = field(default_factory=set)


def build_raw_content(title: str, body: str) -> str:
    return f"{title}\n\n{body}" if title else body


def accept_replacement(
    old_raw: str | None,
    new_raw: str,
    title: str | None,
    min_chars: int = BODY_MIN_CHARS,
) -> bool:
    """새 본문이 기존보다 낫다고 볼 수 있을 때만 교체한다.

    - 새 본문은 반드시 ``min_chars`` 이상이고 문자가 깨지지 않아야 한다
      (오류 페이지 같은 짧은 글로 덮어쓰지 않는다).
    - 기존보다 길거나, 기존이 깨져 있으면 교체한다.
    """
    # 깨짐은 본문만 본다: 저장된 제목(예전 깨진 추출의 첫 줄)이 깨져 있어도
    # 깨끗한 새 본문으로 고칠 수 있어야 한다. body_text는 공백을 접어 \x85 같은
    # 깨짐 흔적까지 지우므로, 깨짐은 제목만 뗀 원문에서 본다.
    if not has_body(new_raw, title, min_chars) or is_mojibake(_strip_title(new_raw, title)):
        return False
    if is_mojibake(_strip_title(old_raw, title)):
        return True
    return len(body_text(new_raw, title)) > len(body_text(old_raw, title))


def _strip_title(raw: str | None, title: str | None) -> str:
    text = (raw or "").strip()
    wanted = (title or "").strip()
    if wanted and text.startswith(wanted):
        return text[len(wanted):]
    return text


def _title_of(item: dict, fetched: dict | None) -> str:
    raw = (item.get("raw_content") or "").strip()
    if "\n\n" in raw:
        first = derive_title(raw)
        if first:
            return first
    return (fetched or {}).get("title") or ""


async def enrich_bodies(
    items: list[dict],
    *,
    min_chars: int = BODY_MIN_CHARS,
    concurrency: int = _CONCURRENCY,
    max_attempts: int | None = None,
    fetch: FetchFn | None = None,
) -> EnrichReport:
    """본문이 짧은 항목의 ``raw_content``를 기사 페이지 본문으로 교체한다 (제자리 수정)."""
    fetch_fn: FetchFn = fetch or fetch_article
    report = EnrichReport()
    breaker = HostBreaker()
    sem = asyncio.Semaphore(max(1, concurrency))

    targets = [
        it
        for it in items
        if (it.get("source_url") or "").strip()
        and not is_excluded_host(host_of(it["source_url"]))
        and not has_body(it.get("raw_content"), None, min_chars)
    ]

    async def _one(item: dict) -> None:
        url = item["source_url"]
        host = host_of(url)
        async with sem:
            if max_attempts is not None and report.attempted >= max_attempts:
                return
            if breaker.is_open(host):
                report.skipped += 1
                report.skipped_hosts.add(host)
                return
            report.attempted += 1
            fetched: dict[str, Any] | None = None
            try:
                fetched = await fetch_fn(url)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 보강 실패가 실행을 멈추면 안 된다
                logger.info("body enrich failed for %s: %r", url, exc)

            body = ((fetched or {}).get("raw_content") or "").strip()
            title = _title_of(item, fetched)
            candidate = build_raw_content(title, body) if body else ""
            if candidate and accept_replacement(
                item.get("raw_content"), candidate, title, min_chars
            ):
                item["raw_content"] = candidate
                if not item.get("image_url") and fetched.get("image_url"):
                    item["image_url"] = fetched["image_url"]
                if not item.get("_published_at") and fetched.get("_published_at"):
                    item["_published_at"] = fetched["_published_at"]
                report.enriched += 1
                breaker.record(host, ok=True)
            else:
                breaker.record(host, ok=False)

    await asyncio.gather(*(_one(it) for it in targets))
    report.missing = sum(1 for it in items if not has_body(it.get("raw_content"), None, min_chars))
    return report
