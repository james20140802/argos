"""``argos body-stats`` — 본문 500자 미만 비율 집계 (ARG-309). DB 없이."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos import cli
from argos.crawler import body_backfill
from argos.crawler.body_backfill import BodyRow, compute_stats

LONG = "word " * 200  # 1000자 내외


def _row(host: str, body: str, title: str = "T") -> BodyRow:
    return BodyRow(
        id=uuid.uuid4(),
        title=title,
        source_url=f"https://{host}/p/{uuid.uuid4().hex[:6]}",
        raw_content=f"{title}\n\n{body}",
    )


def test_compute_stats_overall():
    rows = [
        _row("a.com", "short"),
        _row("a.com", LONG),
        _row("b.com", "tiny"),
        _row("b.com", "weâ\x80\x99re " + LONG),
    ]
    stats = compute_stats(rows)
    assert stats.total == 4
    assert stats.short == 2
    assert stats.short_pct == pytest.approx(50.0)
    assert stats.mojibake == 1
    assert stats.median_len > 0


def test_compute_stats_median_is_pure_body_length():
    rows = [_row("a.com", "x" * n) for n in (10, 20, 30)]
    assert compute_stats(rows).median_len == 20


def test_compute_stats_by_host_sorted_by_total_desc_and_strips_www():
    rows = [
        _row("www.a.com", "s"),
        _row("a.com", LONG),
        _row("a.com", "s"),
        _row("b.com", "s"),
    ]
    stats = compute_stats(rows)
    hosts = [h.host for h in stats.by_host]
    assert hosts == ["a.com", "b.com"]
    a = stats.by_host[0]
    assert (a.total, a.short) == (3, 2)


def test_compute_stats_empty():
    stats = compute_stats([])
    assert stats.total == 0 and stats.short_pct == 0.0 and stats.median_len == 0


def _hosts(out: str) -> set[str]:
    """출력 각 줄의 첫 칸 — 호스트 표의 호스트 열."""
    return {line.split()[0] for line in out.splitlines() if line.strip()}


@pytest.fixture
def wired(monkeypatch):
    rows = [_row("a.com", "short"), _row("a.com", LONG), _row("b.com", "tiny")]
    session = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: ctx)
    monkeypatch.setattr(body_backfill, "load_rows", AsyncMock(return_value=rows))
    return session


def test_body_stats_prints_overall_numbers(wired, capsys):
    assert cli.main(["body-stats"]) == 0
    out = capsys.readouterr().out
    assert "전체 3건" in out
    assert "본문 500자 미만 2건 (66.7%)" in out
    assert "문자 깨짐 0건" in out
    assert not {"a.com"} & _hosts(out)  # --by-host 없이는 호스트 표 없음


def test_body_stats_by_host_table(wired, capsys):
    assert cli.main(["body-stats", "--by-host"]) == 0
    out = capsys.readouterr().out
    assert {"a.com", "b.com"} <= _hosts(out)
    a_line = next(line for line in out.splitlines() if line.strip().startswith("a.com"))
    assert " 2" in a_line and "50.0%" in a_line


def test_body_stats_limit_caps_host_rows(wired, capsys):
    assert cli.main(["body-stats", "--by-host", "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert {"a.com", "b.com"} & _hosts(out) == {"a.com"}
