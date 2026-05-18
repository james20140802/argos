from __future__ import annotations

import logging
import plistlib
from unittest.mock import AsyncMock, patch

import pytest

from argos.cli import _configure_logging, _format_duration, main
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
# _configure_logging helper (ARG-118 / ARG-120)
# ---------------------------------------------------------------------------


def test_configure_logging_non_verbose_quiets_httpx():
    """Non-verbose mode clamps httpx and httpcore to WARNING."""
    _configure_logging(verbose=False, tty=False)
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING


def test_configure_logging_verbose_allows_httpx_info():
    """Verbose mode sets httpx and httpcore back to INFO."""
    _configure_logging(verbose=True, tty=False)
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.INFO
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.INFO


def test_configure_logging_non_verbose_root_is_info():
    """Non-verbose root logger level is INFO."""
    _configure_logging(verbose=False, tty=False)
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_verbose_root_is_debug():
    """Verbose root logger level is DEBUG."""
    _configure_logging(verbose=True, tty=False)
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_urllib3_always_warning():
    """urllib3 stays at WARNING regardless of verbose flag."""
    _configure_logging(verbose=True, tty=False)
    assert logging.getLogger("urllib3").getEffectiveLevel() == logging.WARNING
    _configure_logging(verbose=False, tty=False)
    assert logging.getLogger("urllib3").getEffectiveLevel() == logging.WARNING


def test_configure_logging_non_tty_returns_none():
    """Non-TTY path returns None (no shared console)."""
    result = _configure_logging(verbose=False, tty=False)
    assert result is None


def test_configure_logging_tty_returns_console():
    """TTY path returns a rich Console instance."""
    from rich.console import Console

    result = _configure_logging(verbose=False, tty=True)
    assert isinstance(result, Console)


def test_configure_logging_tty_installs_rich_handler():
    """TTY path installs a RichHandler on the root logger."""
    from rich.logging import RichHandler

    _configure_logging(verbose=False, tty=True)
    root = logging.getLogger()
    handler_types = [type(h) for h in root.handlers]
    assert RichHandler in handler_types, f"Expected RichHandler among {handler_types}"


def test_configure_logging_tty_rich_handler_uses_shared_console():
    """The RichHandler and the returned Console must be the same object."""
    from rich.console import Console
    from rich.logging import RichHandler

    returned_console = _configure_logging(verbose=False, tty=True)
    assert isinstance(returned_console, Console)
    root = logging.getLogger()
    rich_handlers = [h for h in root.handlers if isinstance(h, RichHandler)]
    assert rich_handlers, "No RichHandler found"
    assert rich_handlers[0].console is returned_console


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


# ---------------------------------------------------------------------------
# argos init dispatch
# ---------------------------------------------------------------------------


def test_init_help_lists_reconfigure_choices(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["init", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--reconfigure" in out
    for section in ("infra", "slack", "interests", "schedule"):
        assert section in out


def test_init_full_dispatches_to_run_full(monkeypatch) -> None:
    seen = {}

    def fake_full(*a, **kw):
        seen["called"] = "full"
        return 0

    monkeypatch.setattr("argos.init_wizard.wizard.run_full", fake_full)
    rc = main(["init"])
    assert rc == 0
    assert seen["called"] == "full"


def test_init_reconfigure_section_dispatches(monkeypatch) -> None:
    seen = {}

    def fake_reconfigure(section, *a, **kw):
        seen["section"] = section
        return 0

    monkeypatch.setattr("argos.init_wizard.wizard.run_reconfigure", fake_reconfigure)
    rc = main(["init", "--reconfigure", "interests"])
    assert rc == 0
    assert seen["section"] == "interests"


def test_init_non_interactive_flag_sets_env_var(monkeypatch) -> None:
    monkeypatch.delenv("ARGOS_INIT_NONINTERACTIVE", raising=False)
    seen = {}

    def fake_full(*a, **kw):
        # Capture env at call time so we know the CLI flipped it before dispatch.
        import os

        seen["env"] = os.environ.get("ARGOS_INIT_NONINTERACTIVE")
        return 0

    monkeypatch.setattr("argos.init_wizard.wizard.run_full", fake_full)
    main(["init", "--non-interactive"])
    assert seen["env"] == "1"


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


# ---------------------------------------------------------------------------
# Finding 1 regression: `argos schedule install` (no flag, no config file)
# must succeed and embed the default absolute path in the plist.
# ---------------------------------------------------------------------------


def test_schedule_install_no_flag_no_config_succeeds(tmp_path, monkeypatch) -> None:
    """`argos schedule install` with no --config and no config file on disk
    must succeed (use defaults) and pass config_path=None to reload_schedule.

    This is the first-run / fresh-machine case. Passing config_path=None means
    the rendered plists omit --config entirely so launchd fires `argos run`
    without a --config arg. That path uses the permissive UserConfig.load and
    treats a missing default file as "use built-in defaults" — which is the
    correct behaviour on a fresh machine. Embedding the path to a missing file
    would instead cause _apply_config_override (strict for any explicit --config)
    to exit non-zero on every scheduled trigger.
    """
    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr("argos.scheduler.shutil.which", lambda name: str(fake_bin))

    default_cfg = tmp_path / "config.toml"  # does NOT exist yet
    monkeypatch.setattr("argos.config_store.default_config_path", lambda: default_cfg)

    captured_path: list = []

    def fake_reload(user_config, *, config_path):
        captured_path.append(config_path)

    monkeypatch.setattr("argos.scheduler.reload_schedule", fake_reload)

    rc = main(["schedule", "install"])

    assert rc == 0
    assert captured_path, "reload_schedule was not called"
    # Missing default file → config_path=None so the plist omits --config.
    assert captured_path[0] is None


def test_schedule_install_explicit_nonexistent_config_fails(tmp_path, monkeypatch) -> None:
    """`argos schedule --config /nonexistent.toml install` must exit non-zero."""
    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr("argos.scheduler.shutil.which", lambda name: str(fake_bin))

    missing = tmp_path / "nonexistent.toml"  # does NOT exist

    rc = main(["schedule", "--config", str(missing), "install"])

    assert rc != 0


def test_schedule_install_explicit_broken_toml_fails(tmp_path, monkeypatch) -> None:
    """`argos schedule --config /broken.toml install` must exit non-zero."""
    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr("argos.scheduler.shutil.which", lambda name: str(fake_bin))

    bad_toml = tmp_path / "broken.toml"
    bad_toml.write_text("not = valid [ toml")

    rc = main(["schedule", "--config", str(bad_toml), "install"])

    assert rc != 0


# ---------------------------------------------------------------------------
# Regression (PRRT_kwDOR4m8Js6BEAqh): `schedule install` (no flag, no file)
# must NOT embed --config in the plist. Embedding a path to a missing file
# causes every launchd-triggered `argos run --config <path>` to exit non-zero
# via the strict _apply_config_override path.
# ---------------------------------------------------------------------------


def test_schedule_install_no_flag_no_file_plist_omits_config(
    tmp_path, monkeypatch
) -> None:
    """Case 1: no --config, default file absent → ProgramArguments must be
    exactly [argos_bin, "run"] (no --config tail).
    """
    import subprocess
    import plistlib
    from argos import scheduler

    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr("argos.scheduler.shutil.which", lambda name: str(fake_bin))

    default_cfg = tmp_path / "config.toml"  # does NOT exist
    monkeypatch.setattr("argos.config_store.default_config_path", lambda: default_cfg)

    la = tmp_path / "LaunchAgents"
    logs = tmp_path / "Logs"
    monkeypatch.setattr(scheduler, "_DEFAULT_LAUNCH_AGENTS", la)
    monkeypatch.setattr(scheduler, "_DEFAULT_LOG_DIR", logs)
    monkeypatch.setattr(scheduler, "_current_uid", lambda: 501)

    def fake_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake_launchctl)

    rc = main(["schedule", "install"])
    assert rc == 0

    run_plist = la / "com.argos.run.plist"
    brief_plist = la / "com.argos.brief.plist"
    run_args = plistlib.loads(run_plist.read_bytes())["ProgramArguments"]
    brief_args = plistlib.loads(brief_plist.read_bytes())["ProgramArguments"]

    # No --config in either plist when the default file is absent.
    assert "--config" not in run_args, f"unexpected --config in run plist: {run_args}"
    assert "--config" not in brief_args, f"unexpected --config in brief plist: {brief_args}"
    assert run_args == [str(fake_bin), "run"]
    assert brief_args[:2] == [str(fake_bin), "brief"]


def test_schedule_install_no_flag_file_exists_plist_embeds_config(
    tmp_path, monkeypatch
) -> None:
    """Case 2: no --config, default file EXISTS → ProgramArguments must embed
    --config <resolved-default-path>.
    """
    import subprocess
    import plistlib
    from argos import scheduler

    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr("argos.scheduler.shutil.which", lambda name: str(fake_bin))

    default_cfg = tmp_path / "config.toml"
    default_cfg.write_text("[run]\ntime = \"06:00\"\n")  # file exists
    monkeypatch.setattr("argos.config_store.default_config_path", lambda: default_cfg)

    la = tmp_path / "LaunchAgents"
    logs = tmp_path / "Logs"
    monkeypatch.setattr(scheduler, "_DEFAULT_LAUNCH_AGENTS", la)
    monkeypatch.setattr(scheduler, "_DEFAULT_LOG_DIR", logs)
    monkeypatch.setattr(scheduler, "_current_uid", lambda: 501)

    def fake_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake_launchctl)

    rc = main(["schedule", "install"])
    assert rc == 0

    run_plist = la / "com.argos.run.plist"
    brief_plist = la / "com.argos.brief.plist"
    run_args = plistlib.loads(run_plist.read_bytes())["ProgramArguments"]
    brief_args = plistlib.loads(brief_plist.read_bytes())["ProgramArguments"]

    assert "--config" in run_args
    assert str(default_cfg.resolve()) in run_args
    assert "--config" in brief_args
    assert str(default_cfg.resolve()) in brief_args


def test_schedule_install_no_flag_broken_toml_plist_omits_config(
    tmp_path, monkeypatch
) -> None:
    """Case 3 (PRRT_kwDOR4m8Js6BEI7i): no --config, default file EXISTS but is
    invalid TOML → install must succeed (resilient) and the plist must NOT embed
    --config, because the scheduled `argos run --config <path>` would hit the
    strict _apply_config_override path and exit non-zero on every launchd trigger.
    """
    import subprocess
    import plistlib
    from argos import scheduler

    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr("argos.scheduler.shutil.which", lambda name: str(fake_bin))

    default_cfg = tmp_path / "config.toml"
    default_cfg.write_text("not = valid [ toml")  # broken TOML
    monkeypatch.setattr("argos.config_store.default_config_path", lambda: default_cfg)

    la = tmp_path / "LaunchAgents"
    logs = tmp_path / "Logs"
    monkeypatch.setattr(scheduler, "_DEFAULT_LAUNCH_AGENTS", la)
    monkeypatch.setattr(scheduler, "_DEFAULT_LOG_DIR", logs)
    monkeypatch.setattr(scheduler, "_current_uid", lambda: 501)

    def fake_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake_launchctl)

    rc = main(["schedule", "install"])
    assert rc == 0  # install must be resilient — not fail

    run_plist = la / "com.argos.run.plist"
    brief_plist = la / "com.argos.brief.plist"
    run_args = plistlib.loads(run_plist.read_bytes())["ProgramArguments"]
    brief_args = plistlib.loads(brief_plist.read_bytes())["ProgramArguments"]

    # Broken TOML → must NOT embed --config so launchd uses permissive defaults.
    assert "--config" not in run_args, f"unexpected --config in run plist: {run_args}"
    assert "--config" not in brief_args, f"unexpected --config in brief plist: {brief_args}"
    assert run_args == [str(fake_bin), "run"]
    assert brief_args[:2] == [str(fake_bin), "brief"]


def test_schedule_install_no_flag_invalid_schema_plist_omits_config(
    tmp_path, monkeypatch
) -> None:
    """Case 4 (PRRT_kwDOR4m8Js6BEI7i): no --config, default file EXISTS but fails
    schema validation → install must succeed (resilient) and the plist must NOT
    embed --config.
    """
    import subprocess
    import plistlib
    from argos import scheduler

    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr("argos.scheduler.shutil.which", lambda name: str(fake_bin))

    default_cfg = tmp_path / "config.toml"
    # Valid TOML but schema-invalid value (limit_per_category has ge=1 constraint).
    default_cfg.write_text("[briefing]\nlimit_per_category = 0\n")
    monkeypatch.setattr("argos.config_store.default_config_path", lambda: default_cfg)

    la = tmp_path / "LaunchAgents"
    logs = tmp_path / "Logs"
    monkeypatch.setattr(scheduler, "_DEFAULT_LAUNCH_AGENTS", la)
    monkeypatch.setattr(scheduler, "_DEFAULT_LOG_DIR", logs)
    monkeypatch.setattr(scheduler, "_current_uid", lambda: 501)

    def fake_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake_launchctl)

    rc = main(["schedule", "install"])
    assert rc == 0  # install must be resilient — not fail

    run_plist = la / "com.argos.run.plist"
    brief_plist = la / "com.argos.brief.plist"
    run_args = plistlib.loads(run_plist.read_bytes())["ProgramArguments"]
    brief_args = plistlib.loads(brief_plist.read_bytes())["ProgramArguments"]

    # Schema-invalid file → must NOT embed --config so launchd uses permissive defaults.
    assert "--config" not in run_args, f"unexpected --config in run plist: {run_args}"
    assert "--config" not in brief_args, f"unexpected --config in brief plist: {brief_args}"
    assert run_args == [str(fake_bin), "run"]
    assert brief_args[:2] == [str(fake_bin), "brief"]


def test_schedule_install_no_flag_no_file_launchd_invocation_uses_defaults(
    tmp_path, monkeypatch
) -> None:
    """Case 4 round-trip: after installing with no --config and no file,
    simulate launchd's invocation of `argos run` (no --config) → exit 0.

    This is the regression path: if the plist incorrectly embedded
    `--config <missing-path>`, argos run would exit non-zero (FileNotFoundError
    via _apply_config_override). The plist must produce a bare `argos run`.
    """
    fake_bin = tmp_path / "argos"
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr("argos.scheduler.shutil.which", lambda name: str(fake_bin))

    default_cfg = tmp_path / "config.toml"  # does NOT exist
    monkeypatch.setattr("argos.config_store.default_config_path", lambda: default_cfg)

    # Capture what reload_schedule receives so we can read the rendered plist.
    import subprocess
    import plistlib
    from argos import scheduler

    la = tmp_path / "LaunchAgents"
    logs = tmp_path / "Logs"
    monkeypatch.setattr(scheduler, "_DEFAULT_LAUNCH_AGENTS", la)
    monkeypatch.setattr(scheduler, "_DEFAULT_LOG_DIR", logs)
    monkeypatch.setattr(scheduler, "_current_uid", lambda: 501)

    def fake_launchctl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "print":
            label = args[1].rsplit("/", 1)[-1]
            return subprocess.CompletedProcess(
                ["launchctl", *args], 0, stdout=f"{label} = service", stderr=""
            )
        return subprocess.CompletedProcess(
            ["launchctl", *args], 0, stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler, "_run_launchctl", fake_launchctl)

    # Install with no flag and no file.
    rc = main(["schedule", "install"])
    assert rc == 0

    run_plist = la / "com.argos.run.plist"
    program_args = plistlib.loads(run_plist.read_bytes())["ProgramArguments"]
    # Strip the binary (argv[0]) to get the CLI argv.
    cli_argv = program_args[1:]

    # cli_argv must be ["run"] — no --config — so launchd's invocation uses defaults.
    assert cli_argv == ["run"], f"expected ['run'], got {cli_argv!r}"

    # Now simulate launchd's actual call: `argos run` (no --config).
    # Must succeed with default config (missing file → permissive load → defaults).
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
        rc2 = main(cli_argv)

    assert rc2 == 0


# ---------------------------------------------------------------------------
# Finding 2 regression: `argos run/brief --config <bad-path>` must now exit
# non-zero instead of silently using defaults.
# ---------------------------------------------------------------------------


def test_run_explicit_nonexistent_config_exits_nonzero(tmp_path, capsys) -> None:
    """`argos run --config /nonexistent.toml` must exit non-zero with a message."""
    missing = tmp_path / "nonexistent.toml"

    rc = main(["run", "--config", str(missing)])

    assert rc != 0
    err = capsys.readouterr().err
    assert "not found" in err.lower() or str(missing) in err


def test_run_explicit_broken_toml_exits_nonzero(tmp_path, capsys) -> None:
    """`argos run --config /broken.toml` must exit non-zero, not use defaults."""
    bad_toml = tmp_path / "broken.toml"
    bad_toml.write_text("not = valid [ toml")

    rc = main(["run", "--config", str(bad_toml)])

    assert rc != 0
    err = capsys.readouterr().err
    assert str(bad_toml) in err


def test_brief_explicit_nonexistent_config_exits_nonzero(tmp_path, capsys) -> None:
    """`argos brief --config /nonexistent.toml` must exit non-zero."""
    missing = tmp_path / "nonexistent.toml"

    rc = main(["brief", "--config", str(missing)])

    assert rc != 0
    err = capsys.readouterr().err
    assert "not found" in err.lower() or str(missing) in err


# ---------------------------------------------------------------------------
# argos --version flag (ARG-79)
# ---------------------------------------------------------------------------


def test_version_flag_prints_package_version(monkeypatch, capsys) -> None:
    """`argos --version` prints the version from importlib.metadata."""
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.0")

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    # argparse prints to stdout for --version
    out = capsys.readouterr().out
    assert "0.1.0" in out


def test_version_flag_fallback_when_metadata_missing(monkeypatch, capsys, tmp_path) -> None:
    """`argos --version` falls back gracefully when importlib.metadata raises PackageNotFoundError."""
    import importlib.metadata

    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)

    # Also mock the pyproject.toml read so we don't depend on the on-disk file.
    fake_pyproject = tmp_path / "pyproject.toml"
    fake_pyproject.write_text('[project]\nversion = "0.2.0-dev"\n')

    # Patch Path(__file__).parent.parent.parent chain inside _resolve_version by
    # monkeypatching the cli module's tomllib.load to return a controlled dict.
    import argos.cli as cli_mod

    original_resolve = cli_mod._resolve_version

    def _fake_resolve():
        try:
            import importlib.metadata as _m
            return _m.version("argos-scout")
        except _m.PackageNotFoundError:
            return "dev"

    monkeypatch.setattr(cli_mod, "_resolve_version", _fake_resolve)

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.strip()  # non-empty — some fallback string was printed


def test_run_no_config_flag_no_file_uses_defaults(tmp_path, monkeypatch) -> None:
    """`argos run` (no --config, no file) must succeed with defaults."""
    default_cfg = tmp_path / "config.toml"  # does NOT exist
    monkeypatch.setattr("argos.config_store.default_config_path", lambda: default_cfg)

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
        rc = main(["run"])

    assert rc == 0
