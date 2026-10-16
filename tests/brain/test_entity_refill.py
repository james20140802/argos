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


title_calls: list = []


def _fake_extractor(calls):
    def fake(documents, *, titles=None):
        calls.append(list(documents))
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
    title_calls.clear()
    rows = [_row("a", 1), _row("b", 1)]
    monkeypatch.setattr(entity_extraction, "extract_names", _fake_extractor([]))

    extract_by_crawl_date(rows)

    assert title_calls == [["t", "t"]]
