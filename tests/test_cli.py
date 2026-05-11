from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from argos.cli import main, _format_duration
from argos.crawler.pipeline import PipelineSummary


# ---------------------------------------------------------------------------
# _format_duration helper
# ---------------------------------------------------------------------------

def test_format_duration_seconds_only():
    assert _format_duration(45.9) == "45s"


def test_format_duration_minutes_and_seconds():
    assert _format_duration(83.0) == "1m 23s"


def test_format_duration_zero():
    assert _format_duration(0) == "0s"


# ---------------------------------------------------------------------------
# _run / argos run — summary output
# ---------------------------------------------------------------------------

def _make_summary(**kwargs):
    defaults = {
        "crawled_total": 45,
        "per_source": {"github_trending": 25, "hackernews": 20},
        "triage_pass": 12,
        "saved_new": 8,
        "genealogy_skipped": 0,
        "duration_seconds": 83.0,
    }
    defaults.update(kwargs)
    return PipelineSummary(**defaults)


def _make_mock_states(n: int = 2):
    return [
        {
            "is_valid": True,
            "saved": True,
            "source_url": f"https://example.com/{i}",
            "raw_text": "",
            "extracted_info": None,
            "related_tech_ids": [],
            "succession_result": None,
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_run_prints_summary_block(capsys, monkeypatch) -> None:
    summary = _make_summary()
    states = _make_mock_states(2)

    async def fake_session_context():
        return None

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=mock_session),
        patch(
            "argos.cli.run_full_pipeline",
            new=AsyncMock(return_value=(states, summary)),
        ),
    ):
        from argos.cli import _run
        rc = await _run([])

    assert rc == 0
    captured = capsys.readouterr().out
    assert "argos run 완료" in captured
    assert "45개" in captured
    assert "GitHub: 25" in captured
    assert "HN: 20" in captured
    assert "트리아지 통과: 12개" in captured
    assert "신규 저장: 8개" in captured
    # Duration line is present (exact value varies by wall clock in real run,
    # but in this mock the format_duration uses elapsed from monotonic, so just
    # check the line exists)
    assert "소요 시간:" in captured


@pytest.mark.asyncio
async def test_run_empty_crawl_shows_zero_counts(capsys) -> None:
    summary = _make_summary(crawled_total=0, per_source={}, triage_pass=0, saved_new=0)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=mock_session),
        patch(
            "argos.cli.run_full_pipeline",
            new=AsyncMock(return_value=([], summary)),
        ),
    ):
        from argos.cli import _run
        await _run([])

    captured = capsys.readouterr().out
    assert "크롤링: 0개" in captured
    assert "트리아지 통과: 0개" in captured
    assert "신규 저장: 0개" in captured


@pytest.mark.asyncio
async def test_run_prints_genealogy_skipped_line_when_nonzero(capsys) -> None:
    summary = _make_summary(genealogy_skipped=4)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=mock_session),
        patch(
            "argos.cli.run_full_pipeline",
            new=AsyncMock(return_value=([], summary)),
        ),
    ):
        from argos.cli import _run
        await _run([])

    captured = capsys.readouterr().out
    assert "족보 분석 스킵: 4개 (DB 부족)" in captured


@pytest.mark.asyncio
async def test_run_omits_genealogy_line_when_zero(capsys) -> None:
    summary = _make_summary(genealogy_skipped=0)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=mock_session),
        patch(
            "argos.cli.run_full_pipeline",
            new=AsyncMock(return_value=([], summary)),
        ),
    ):
        from argos.cli import _run
        await _run([])

    captured = capsys.readouterr().out
    assert "족보 분석 스킵" not in captured


def test_main_run_subcommand_exits_zero(monkeypatch) -> None:
    summary = _make_summary()
    states = _make_mock_states(1)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=mock_session),
        patch(
            "argos.cli.run_full_pipeline",
            new=AsyncMock(return_value=(states, summary)),
        ),
    ):
        rc = main(["run"])

    assert rc == 0
