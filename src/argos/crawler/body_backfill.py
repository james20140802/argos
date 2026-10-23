"""저장된 기사의 본문 부족 집계와 다시 가져오기 (ARG-309).

* :func:`compute_stats` — ``argos body-stats``가 쓰는 순수 집계.
* ``refetch-bodies`` 쪽 로직은 같은 모듈 아래쪽에 있다.

"본문 없음"은 길이만으로 판정한다 (:data:`BODY_MIN_CHARS`, 컬럼 없음).
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from argos.crawler.body_enrich import host_of
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
