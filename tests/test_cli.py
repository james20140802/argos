from __future__ import annotations

import plistlib
from unittest.mock import AsyncMock, patch

import pytest

from argos.cli import _format_duration, main
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


# ---------------------------------------------------------------------------
# `--config <path>` regression coverage for run/brief (ARG-51 P1).
#
# `scheduler.reload_schedule(..., config_path=path)` renders plists whose
# `ProgramArguments` include `--config <path>`. The run/brief argparse
# surface must accept that flag or every scheduled launchd job would crash
# with argparse exit 2 ("unrecognized arguments: --config /path").
# ---------------------------------------------------------------------------


def test_run_parser_accepts_config_flag(tmp_path, monkeypatch) -> None:
    """`argos run --config <path>` must NOT die with argparse SystemExit(2)."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("")  # Empty TOML → UserConfig defaults apply.

    summary = _make_summary()
    states = _make_mock_states(0)

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
        rc = main(["run", "--config", str(cfg_file)])

    assert rc == 0


def test_brief_parser_accepts_config_flag(tmp_path) -> None:
    """`argos brief --config <path>` must NOT die with argparse SystemExit(2)."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("")

    with patch(
        "argos.slack.briefing.dispatch_daily_briefing",
        new=AsyncMock(return_value=None),
    ):
        rc = main(["brief", "--config", str(cfg_file)])

    assert rc == 0


def test_apply_config_override_reloads_settings_user(tmp_path) -> None:
    """`_apply_config_override` should swap `settings.user` with values from the override file."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        # A non-default value we can read back to prove the file was honored.
        "[briefing]\n"
        'time = "09:42"\n'
    )

    import argparse

    from argos import cli as cli_mod

    args = argparse.Namespace(config=str(cfg_file))
    cli_mod._apply_config_override(args)

    assert cli_mod.settings.user.briefing.time == "09:42"


def test_plist_program_arguments_roundtrip_through_run_parser(
    tmp_path, monkeypatch
) -> None:
    """Round-trip: the `--config <path>` shape the scheduler writes into plists
    is exactly the shape `argos run` accepts.

    Renders a real plist via `render_run_plist`, parses its
    `ProgramArguments`, strips the leading argos binary path, and feeds the
    remainder to `main()`. Failure mode under the original bug: argparse
    SystemExit(2).
    """
    # Force the binary lookup to a sentinel path so the test is hermetic.
    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(
        "argos.scheduler.shutil.which", lambda name: str(fake_bin)
    )

    from argos.scheduler import render_run_plist

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")

    xml = render_run_plist(time="06:00", config_path=cfg_path)
    parsed = plistlib.loads(xml.encode("utf-8"))
    program_args = parsed["ProgramArguments"]

    # Sanity: the scheduler put `--config <cfg_path>` after the subcommand.
    assert program_args[0] == str(fake_bin)
    assert program_args[1] == "run"
    assert "--config" in program_args
    assert str(cfg_path) in program_args

    # Drop the binary (argv[0] for the real exec) and run the rest through
    # the CLI parser. The handler is fully mocked so we only exercise argparse.
    cli_argv = program_args[1:]

    summary = _make_summary()
    states = _make_mock_states(0)
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
        rc = main(cli_argv)

    assert rc == 0


def test_plist_program_arguments_roundtrip_through_brief_parser(
    tmp_path, monkeypatch
) -> None:
    """Same round-trip check for the brief plist."""
    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(
        "argos.scheduler.shutil.which", lambda name: str(fake_bin)
    )

    from argos.scheduler import render_brief_plist

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")

    xml = render_brief_plist(
        time="07:00",
        weekdays=["Mon", "Tue", "Wed", "Thu", "Fri"],
        config_path=cfg_path,
    )
    parsed = plistlib.loads(xml.encode("utf-8"))
    program_args = parsed["ProgramArguments"]

    assert program_args[1] == "brief"
    assert "--config" in program_args

    cli_argv = program_args[1:]

    with patch(
        "argos.slack.briefing.dispatch_daily_briefing",
        new=AsyncMock(return_value=None),
    ):
        rc = main(cli_argv)

    assert rc == 0
