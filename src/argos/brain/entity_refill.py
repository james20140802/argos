"""평가용 '메모리 이름 채우기' — 크롤 날짜별 배치 추출 (ARG-301).

추출기는 문서빈도를 **호출한 배치 안에서만** 센다. 운영의 이름 채우기는 크롤
날짜 단위로 돌기 때문에, 평가도 같은 묶음으로 추출해야 같은 이름이 나온다.
이 모듈은 DB에 쓰지 않는다 — 읽어서 메모리에서만 이름을 만든다.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass
from itertools import groupby
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain import entity_extraction
from argos.brain.entity_extraction import ExtractedName

_ROWS_SQL = text(
    """
    SELECT id, title, raw_content, created_at::date AS crawl_date
    FROM tech_items
    ORDER BY created_at, id
    """
)


@dataclass(frozen=True)
class RefillRow:
    tech_item_id: uuid.UUID
    title: str | None
    raw_content: str | None
    crawl_date: datetime.date


async def load_refill_rows(session: AsyncSession) -> list[RefillRow]:
    """추출 대상 전부. 순서(created_at, id)가 곧 묶음 안 순서라 결과가 결정적이다."""
    result = await session.execute(_ROWS_SQL)
    return [
        RefillRow(
            tech_item_id=row.id,
            title=row.title,
            raw_content=row.raw_content,
            crawl_date=row.crawl_date,
        )
        for row in result.all()
    ]


def extract_by_crawl_date(
    rows: Sequence[RefillRow],
) -> dict[uuid.UUID, list[ExtractedName]]:
    """크롤 날짜 오름차순으로 묶어 묶음마다 추출기를 한 번 부른다."""
    # sorted는 안정 정렬 — 같은 날짜 안에서는 입력 순서가 유지된다.
    ordered = sorted(rows, key=lambda row: row.crawl_date)
    found: dict[uuid.UUID, list[ExtractedName]] = {}
    for _, group in groupby(ordered, key=lambda row: row.crawl_date):
        batch = list(group)
        # 모듈 속성으로 불러야 테스트가 추출기를 바꿔 끼울 수 있다.
        names = entity_extraction.extract_names([row.raw_content or "" for row in batch])
        for row, row_names in zip(batch, names):
            found[row.tech_item_id] = row_names
    return found
