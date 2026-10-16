"""메모리 이름 채우기 — 크롤 날짜별 배치 추출 (ARG-301). DB 없이 돈다."""

from __future__ import annotations

import datetime
import uuid

from argos.brain import entity_extraction
from argos.brain.entity_refill import RefillRow, extract_by_crawl_date


def _row(content, day) -> RefillRow:
    return RefillRow(
        tech_item_id=uuid.uuid4(),
        title="t",
        raw_content=content,
        crawl_date=datetime.date(2026, 10, day),
    )


def _fake_extractor(calls, title_calls=None):
    def fake(documents, *, titles=None):
        calls.append(list(documents))
        if title_calls is not None:
            title_calls.append(None if titles is None else list(titles))
        return [
            [entity_extraction.ExtractedName(canonical=doc.lower(), surface=doc)]
            for doc in documents
        ]

    return fake


def test_extract_by_crawl_date_batches_per_date(monkeypatch):
    rows = [_row("a", 2), _row("b", 1), _row("c", 2)]
    calls: list[list[str]] = []
    monkeypatch.setattr(entity_extraction, "extract_names", _fake_extractor(calls))

    result = extract_by_crawl_date(rows)

    assert calls == [["b"], ["a", "c"]]
    assert set(result) == {r.tech_item_id for r in rows}


def test_extract_by_crawl_date_is_deterministic(monkeypatch):
    rows = [_row("a", 2), _row("b", 1), _row("c", 2)]
    monkeypatch.setattr(entity_extraction, "extract_names", _fake_extractor([]))

    assert extract_by_crawl_date(rows) == extract_by_crawl_date(rows)


def test_none_raw_content_becomes_empty_string(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(entity_extraction, "extract_names", _fake_extractor(calls))

    extract_by_crawl_date([_row(None, 1)])

    assert calls == [[""]]


def test_titles_are_passed_alongside_documents(monkeypatch):
    title_calls: list = []
    rows = [_row("a", 1), _row("b", 1)]
    monkeypatch.setattr(
        entity_extraction, "extract_names", _fake_extractor([], title_calls)
    )

    extract_by_crawl_date(rows)

    assert title_calls == [["t", "t"]]


# --- plan_refill / apply_refill (ARG-304) — DB 없이 ---------------------------

import pytest  # noqa: E402

from argos.brain import entity_refill  # noqa: E402
from argos.brain.entity_extraction import ExtractedName  # noqa: E402


@pytest.mark.asyncio
async def test_plan_refill_reports_added_and_removed(monkeypatch):
    changed, same, fresh = (uuid.uuid4() for _ in range(3))
    rows = [
        RefillRow(changed, "Changed", "x", datetime.date(2026, 10, 1)),
        RefillRow(same, "Same", "x", datetime.date(2026, 10, 1)),
        RefillRow(fresh, None, "x", datetime.date(2026, 10, 1)),
    ]

    async def fake_rows(session):
        return rows

    def fake_extract(_rows):
        return {
            changed: [ExtractedName("frontier", "Frontier")],
            same: [ExtractedName("claude", "Claude")],
            fresh: [],
        }

    async def fake_stored(session, ids):
        return {changed: frozenset({"we", "openai"}), same: frozenset({"claude"})}

    monkeypatch.setattr(entity_refill, "load_refill_rows", fake_rows)
    monkeypatch.setattr(entity_refill, "extract_by_crawl_date", fake_extract)
    monkeypatch.setattr(entity_refill, "raw_names_for_documents", fake_stored)

    plan, extracted = await entity_refill.plan_refill(None)

    assert plan.total == 3
    assert [c.tech_item_id for c in plan.changes] == [changed]
    change = plan.changes[0]
    assert change.added == ("frontier",)
    assert change.removed == ("openai", "we")
    assert (plan.links_added, plan.links_removed) == (1, 2)
    assert extracted[same][0].canonical == "claude"
