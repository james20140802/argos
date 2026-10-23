"""``argos refetch-bodies`` — 미리보기 -> 확인 -> 쓰기 (ARG-309). DB 없이."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos import cli
from argos.brain.near_duplicate import simhash
from argos.brain.simhash_storage import to_storage
from argos.crawler import body_backfill
from argos.crawler.body_backfill import BodyRow, refetch_and_apply, select_targets

LONG = "Real article sentence. " * 40


def _row(url: str, body: str = "short", title: str = "T") -> BodyRow:
    return BodyRow(
        id=uuid.uuid4(), title=title, source_url=url, raw_content=f"{title}\n\n{body}"
    )


def _page(body: str = LONG) -> dict:
    return {"title": "x", "raw_content": body, "source_url": "x"}


def _session() -> MagicMock:
    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


# --- 대상 선별 -------------------------------------------------------------


def test_select_targets_picks_short_and_garbled_only():
    short = _row("https://a.com/1")
    full = _row("https://a.com/2", LONG)
    garbled = _row("https://a.com/3", "weâ\x80\x99re " + LONG)
    video = _row("https://www.youtube.com/watch?v=1")
    no_url = _row("")
    got = select_targets([short, full, garbled, video, no_url])
    assert got == [short, garbled]


def test_select_targets_host_filter_and_limit():
    rows = [_row("https://www.a.com/1"), _row("https://a.com/2"), _row("https://b.com/3")]
    assert [r.source_url for r in select_targets(rows, host="a.com")] == [
        "https://www.a.com/1",
        "https://a.com/2",
    ]
    assert len(select_targets(rows, limit=2)) == 2


# --- 쓰기 ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refetch_updates_only_improved_rows_with_simhash():
    ok = _row("https://a.com/ok")
    bad = _row("https://b.com/bad")
    session = _session()

    async def fetch(url):
        return _page() if url.endswith("/ok") else _page("error page")

    summary = await refetch_and_apply(session, [ok, bad], fetch=fetch, chunk_size=10)

    assert (summary.updated, summary.unchanged, summary.skipped) == (1, 1, 0)
    assert session.execute.await_count == 1
    stmt = session.execute.await_args.args[0]
    params = stmt.compile().params
    new_raw = "T\n\n" + LONG.strip()
    assert params["raw_content"] == new_raw
    assert params["simhash"] == to_storage(simhash(new_raw))
    assert set(params) >= {"raw_content", "simhash"}
    assert not ({"embedding", "summary", "digest"} & set(params))
    session.commit.assert_awaited()


@pytest.mark.asyncio
async def test_refetch_replaces_garbled_row_even_when_not_longer():
    garbled = _row("https://a.com/1", "weâ\x80\x99re " + "z" * 800)
    session = _session()

    async def fetch(url):
        return _page("we’re " + "z" * 600)

    summary = await refetch_and_apply(session, [garbled], fetch=fetch)
    assert summary.updated == 1
    assert "we’re" in session.execute.await_args.args[0].compile().params["raw_content"]


@pytest.mark.asyncio
async def test_refetch_host_breaker_skips_blocked_host():
    rows = [_row(f"https://blocked.com/{i}") for i in range(5)]
    calls: list[str] = []

    async def fetch(url):
        calls.append(url)
        return None

    summary = await refetch_and_apply(_session(), rows, fetch=fetch, concurrency=1)
    assert len(calls) == 2
    assert summary.skipped == 3 and summary.unchanged == 2
    assert "blocked.com" in summary.skipped_hosts


@pytest.mark.asyncio
async def test_refetch_commits_per_chunk_and_reports_progress():
    rows = [_row(f"https://a.com/{i}") for i in range(5)]
    session = _session()
    seen: list[tuple[int, int]] = []

    async def fetch(url):
        return _page()

    await refetch_and_apply(
        session, rows, fetch=fetch, chunk_size=2, on_progress=lambda d, t: seen.append((d, t))
    )
    assert session.commit.await_count == 3
    assert seen == [(2, 5), (4, 5), (5, 5)]


# --- CLI -------------------------------------------------------------------


@pytest.fixture
def wired(monkeypatch):
    rows = [
        _row("https://a.com/1"),
        _row("https://a.com/2"),
        _row("https://b.com/1"),
        _row("https://c.com/ok", LONG),
    ]
    session = _session()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: ctx)
    monkeypatch.setattr(body_backfill, "load_rows", AsyncMock(return_value=rows))
    apply = AsyncMock(return_value=body_backfill.RefetchSummary(updated=2, unchanged=1))
    monkeypatch.setattr(body_backfill, "refetch_and_apply", apply)
    return session, apply


def test_dry_run_prints_count_by_host_and_writes_nothing(wired, capsys, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("dry-run must not ask"))
    assert cli.main(["refetch-bodies", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "다시 가져올 기사 3건" in out
    assert "a.com 2" in out and "b.com 1" in out
    apply.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_declined_confirmation_writes_nothing(wired, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert cli.main(["refetch-bodies"]) != 0
    apply.assert_not_awaited()


def test_yes_skips_prompt_and_applies(wired, capsys, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("--yes must not ask"))
    assert cli.main(["refetch-bodies", "--yes"]) == 0
    apply.assert_awaited_once()
    out = capsys.readouterr().out
    assert "갱신 2" in out and "그대로 1" in out


def test_host_and_limit_flags(wired, capsys):
    session, apply = wired
    assert cli.main(["refetch-bodies", "--dry-run", "--host", "a.com", "--limit", "1"]) == 0
    assert "다시 가져올 기사 1건" in capsys.readouterr().out


def test_nothing_to_refetch(wired, capsys, monkeypatch):
    monkeypatch.setattr(body_backfill, "load_rows", AsyncMock(return_value=[]))
    assert cli.main(["refetch-bodies", "--yes"]) == 0
    assert "다시 가져올 기사가 없습니다" in capsys.readouterr().out


def test_openai_rows_trigger_archive_note_and_archive_argument(wired, capsys):
    session, apply = wired
    body_backfill.load_rows.return_value = [  # type: ignore[attr-defined]
        _row("https://openai.com/index/a"),
        _row("https://openai.com/index/b"),
        _row("https://a.com/1"),
    ]
    assert cli.main(["refetch-bodies", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "openai.com 기사 2건" in out and "Internet Archive" in out
    assert apply.await_args.kwargs["archive"] is not None


def test_no_archive_flag_disables_fallback(wired, capsys):
    session, apply = wired
    body_backfill.load_rows.return_value = [_row("https://openai.com/index/a")]  # type: ignore[attr-defined]
    assert cli.main(["refetch-bodies", "--yes", "--no-archive"]) == 0
    assert "Internet Archive" not in capsys.readouterr().out
    assert "archive" not in apply.await_args.kwargs


def test_summary_reports_archive_count(wired, capsys):
    session, apply = wired
    apply.return_value = body_backfill.RefetchSummary(updated=3, unchanged=0, via_archive=3)
    body_backfill.load_rows.return_value = [_row("https://openai.com/index/a")]  # type: ignore[attr-defined]
    assert cli.main(["refetch-bodies", "--yes"]) == 0
    assert "보관본에서 가져옴 3건" in capsys.readouterr().out
