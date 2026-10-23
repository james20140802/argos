"""저장된 기사의 본문 부족 집계와 다시 가져오기 (ARG-309).

* :func:`compute_stats` — ``argos body-stats``가 쓰는 순수 집계.
* ``refetch-bodies`` 쪽 로직은 같은 모듈 아래쪽에 있다.

"본문 없음"은 길이만으로 판정한다 (:data:`BODY_MIN_CHARS`, 컬럼 없음).
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.near_duplicate import simhash
from argos.brain.simhash_storage import to_storage
from argos.crawler import body_enrich
from argos.crawler.body_enrich import (
    HostBreaker,
    accept_replacement,
    build_raw_content,
    host_of,
    is_excluded_host,
)
from argos.crawler.body_quality import BODY_MIN_CHARS, body_text, is_mojibake
from argos.models.tech_item import TechItem


@dataclass(frozen=True)
class BodyRow:
    id: uuid.UUID
    title: str
    source_url: str
    raw_content: str


@dataclass
class HostStats:
    host: str
    total: int = 0
    short: int = 0
    mojibake: int = 0
    median_len: int = 0

    @property
    def short_pct(self) -> float:
        return 100.0 * self.short / self.total if self.total else 0.0


@dataclass
class BodyStats:
    total: int
    short: int
    mojibake: int
    median_len: int
    by_host: list[HostStats]

    @property
    def short_pct(self) -> float:
        return 100.0 * self.short / self.total if self.total else 0.0


async def load_rows(session: AsyncSession) -> list[BodyRow]:
    """모든 tech_items의 (id, title, source_url, raw_content). 읽기 전용."""
    result = await session.execute(
        select(TechItem.id, TechItem.title, TechItem.source_url, TechItem.raw_content)
    )
    return [
        BodyRow(r.id, r.title or "", r.source_url or "", r.raw_content or "")
        for r in result.all()
    ]


def compute_stats(rows: list[BodyRow], min_chars: int = BODY_MIN_CHARS) -> BodyStats:
    lengths: list[int] = []
    per_host: dict[str, list[int]] = {}
    host_moji: dict[str, int] = {}
    mojibake = 0
    for row in rows:
        length = len(body_text(row.raw_content, row.title))
        lengths.append(length)
        host = host_of(row.source_url) or "(unknown)"
        per_host.setdefault(host, []).append(length)
        if is_mojibake(row.raw_content):
            mojibake += 1
            host_moji[host] = host_moji.get(host, 0) + 1

    by_host = [
        HostStats(
            host=host,
            total=len(ls),
            short=sum(1 for n in ls if n < min_chars),
            mojibake=host_moji.get(host, 0),
            median_len=int(statistics.median(ls)),
        )
        for host, ls in per_host.items()
    ]
    by_host.sort(key=lambda h: (-h.total, h.host))
    return BodyStats(
        total=len(rows),
        short=sum(1 for n in lengths if n < min_chars),
        mojibake=mojibake,
        median_len=int(statistics.median(lengths)) if lengths else 0,
        by_host=by_host,
    )


# ---------------------------------------------------------------------------
# refetch-bodies
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_REFETCH_CONCURRENCY = 3
_COMMIT_CHUNK = 50


def _matches_host(url: str, wanted: str) -> bool:
    wanted = wanted.lower().removeprefix("www.")
    host = host_of(url)
    return host == wanted or host.endswith("." + wanted)


def select_targets(
    rows: list[BodyRow],
    *,
    host: str | None = None,
    limit: int | None = None,
    min_chars: int = BODY_MIN_CHARS,
) -> list[BodyRow]:
    """다시 가져올 행: 본문이 짧거나 문자가 깨진 행 (영상·트윗·arXiv 호스트 제외)."""
    targets: list[BodyRow] = []
    for row in rows:
        if not row.source_url.strip() or is_excluded_host(host_of(row.source_url)):
            continue
        if host and not _matches_host(row.source_url, host):
            continue
        if len(body_text(row.raw_content, row.title)) < min_chars or is_mojibake(row.raw_content):
            targets.append(row)
    return targets[:limit] if limit else targets


def count_by_host(targets: list[BodyRow]) -> list[tuple[str, int]]:
    counts = Counter(host_of(r.source_url) or "(unknown)" for r in targets)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class RefetchSummary:
    updated: int = 0
    unchanged: int = 0  # 가져왔지만 더 나은 본문을 얻지 못해 그대로 둔 행 (본문 없음)
    skipped: int = 0  # 호스트 차단기로 건너뛴 행
    skipped_hosts: set[str] = field(default_factory=set)


FetchFn = Callable[[str], Awaitable["dict | None"]]


async def _try_live(url: str, row: BodyRow, fetch: FetchFn) -> str | None:
    """기사 페이지에서 가져온 후보 raw_content (교체 가능할 때만), 아니면 None."""
    try:
        fetched: dict[str, Any] | None = await fetch(url)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.info("refetch failed for %s: %r", url, exc)
        return None
    body = ((fetched or {}).get("raw_content") or "").strip()
    if not body:
        return None
    candidate = build_raw_content(row.title, body)
    return candidate if accept_replacement(row.raw_content, candidate, row.title) else None


async def refetch_and_apply(
    session: AsyncSession,
    targets: list[BodyRow],
    *,
    fetch: FetchFn | None = None,
    concurrency: int = _REFETCH_CONCURRENCY,
    chunk_size: int = _COMMIT_CHUNK,
    on_progress: Callable[[int, int], None] | None = None,
) -> RefetchSummary:
    """대상 행을 다시 가져와 ``raw_content``와 ``simhash``만 갱신한다 (LLM 없음).

    새 본문이 500자 이상이면서 기존보다 길 때(또는 기존이 깨졌을 때)만 쓴다.
    ``chunk_size``건마다 커밋해서 중간에 멈춰도 끝낸 만큼은 남는다.
    """
    fetch_fn: FetchFn = fetch or body_enrich.fetch_article
    summary = RefetchSummary()
    breaker = HostBreaker()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(row: BodyRow) -> tuple[BodyRow, str | None]:
        host = host_of(row.source_url)
        async with sem:
            if breaker.is_open(host):
                summary.skipped += 1
                summary.skipped_hosts.add(host)
                return row, None
            candidate = await _try_live(row.source_url, row, fetch_fn)
            breaker.record(host, ok=candidate is not None)
            return row, candidate

    done = 0
    for start in range(0, len(targets), chunk_size):
        chunk = targets[start : start + chunk_size]
        results = await asyncio.gather(*(_one(r) for r in chunk))
        for row, new_raw in results:
            if new_raw is None:
                continue
            await session.execute(
                update(TechItem)
                .where(TechItem.id == row.id)
                .values(raw_content=new_raw, simhash=to_storage(simhash(new_raw)))
            )
            summary.updated += 1
        await session.commit()
        done += len(chunk)
        if on_progress is not None:
            on_progress(done, len(targets))

    summary.unchanged = len(targets) - summary.updated - summary.skipped
    return summary
