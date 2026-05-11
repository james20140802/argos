from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from argos import config_store
from argos.crawler.pipeline import run_full_pipeline
from argos.database import AsyncSessionLocal

# argos config subcommand exit codes (also documented in `argos config --help`).
EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_UNKNOWN_KEY = 2
EXIT_VALIDATION = 3
EXIT_SECRET = 4


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


async def _run(dynamic_urls: list[str] | None) -> int:
    start = time.monotonic()
    async with AsyncSessionLocal() as session:
        results, summary = await run_full_pipeline(session, dynamic_urls=dynamic_urls or None)
    elapsed = time.monotonic() - start

    # Build per-source breakdown string
    source_parts = []
    if "github_trending" in summary.per_source:
        source_parts.append(f"GitHub: {summary.per_source['github_trending']}")
    if "hackernews" in summary.per_source:
        source_parts.append(f"HN: {summary.per_source['hackernews']}")
    for src, cnt in summary.per_source.items():
        if src not in ("github_trending", "hackernews"):
            source_parts.append(f"{src}: {cnt}")
    source_detail = f" ({', '.join(source_parts)})" if source_parts else ""

    print("✅ argos run 완료")
    print("─────────────────────────────")
    print(f"크롤링: {summary.crawled_total}개{source_detail}")
    print(f"트리아지 통과: {summary.triage_pass}개")
    print(f"신규 저장: {summary.saved_new}개")
    if summary.genealogy_skipped > 0:
        print(f"족보 분석 스킵: {summary.genealogy_skipped}개 (DB 부족)")
    print(f"소요 시간: {_format_duration(elapsed)}")
    return 0


def _resolve_config_path(args: argparse.Namespace) -> Path:
    override = getattr(args, "config", None)
    if override:
        return Path(override).expanduser()
    return config_store.default_config_path()


def _cmd_config_path(args: argparse.Namespace) -> int:
    print(_resolve_config_path(args))
    return EXIT_OK


def _cmd_config_get(args: argparse.Namespace) -> int:
    path = _resolve_config_path(args)
    try:
        value = config_store.get_value(path, args.key)
    except config_store.SecretKeyError:
        print(
            f"Refusing to read secret value {args.key!r} via CLI — use environment variables "
            "or the config file directly.",
            file=sys.stderr,
        )
        return EXIT_SECRET
    except config_store.UnknownKeyError:
        print(f"Unknown config key: {args.key}", file=sys.stderr)
        return EXIT_UNKNOWN_KEY
    if isinstance(value, list):
        print(",".join(str(v) for v in value))
    else:
        print(value)
    return EXIT_OK


def _cmd_config_set(args: argparse.Namespace) -> int:
    path = _resolve_config_path(args)
    try:
        config_store.set_value(path, args.key, args.value)
    except config_store.SecretKeyError:
        print(
            f"Refusing to set secret value {args.key!r} via CLI — use environment variables "
            "or the config file directly.",
            file=sys.stderr,
        )
        return EXIT_SECRET
    except config_store.UnknownKeyError:
        print(f"Unknown config key: {args.key}", file=sys.stderr)
        return EXIT_UNKNOWN_KEY
    except (ValidationError, ValueError) as exc:
        first = str(exc).strip().splitlines()[0]
        print(f"Invalid value for {args.key}: {first}", file=sys.stderr)
        return EXIT_VALIDATION
    except OSError as exc:
        print(f"Could not write config file {path}: {exc}", file=sys.stderr)
        return EXIT_GENERIC
    print(f"{args.key} = {args.value}")
    return EXIT_OK


def _cmd_config_list(args: argparse.Namespace) -> int:
    path = _resolve_config_path(args)
    rows = config_store.list_entries(path)
    if not rows:
        return EXIT_OK
    key_width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key.ljust(key_width)} | {value}")
    return EXIT_OK


def _build_init_parser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``argos init`` subcommand."""
    from argos.init_wizard.wizard import RECONFIGURE_SECTIONS

    init_p = sub.add_parser(
        "init",
        help="Interactive bootstrap (Postgres + Ollama + Slack + schedule + healthcheck)",
        description=(
            "Walk a 6-step wizard that sets up Argos end-to-end on a fresh machine.\n\n"
            "Use --reconfigure to re-run only one section (the wizard always tails "
            "with a healthcheck). Set ARGOS_INIT_NONINTERACTIVE=1 (or pass "
            "--non-interactive) to take every default silently — useful for CI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    init_p.add_argument(
        "--reconfigure",
        choices=list(RECONFIGURE_SECTIONS),
        default=None,
        help="Re-run only the named section instead of the full wizard",
    )
    init_p.add_argument(
        "--non-interactive",
        action="store_true",
        help="Take every default silently (sets ARGOS_INIT_NONINTERACTIVE=1)",
    )


def _cmd_init(args: argparse.Namespace) -> int:
    from argos.init_wizard.wizard import run_full, run_reconfigure

    if getattr(args, "non_interactive", False):
        os.environ["ARGOS_INIT_NONINTERACTIVE"] = "1"

    section = getattr(args, "reconfigure", None)
    if section:
        return run_reconfigure(section)
    return run_full()


def _build_config_parser(sub: argparse._SubParsersAction) -> None:
    config_p = sub.add_parser(
        "config",
        help="Read or update ~/.config/argos/config.toml",
        description=(
            "Manage the user-level Argos config file.\n\n"
            "Exit codes:\n"
            "  0  success\n"
            "  1  generic error (I/O, etc.)\n"
            "  2  unknown config key\n"
            "  3  validation failure\n"
            "  4  secret rejection (use env vars / edit the file directly)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    config_p.add_argument(
        "--config",
        default=None,
        help="Path to config.toml (defaults to ~/.config/argos/config.toml)",
    )
    actions = config_p.add_subparsers(dest="config_action", required=True)

    actions.add_parser("path", help="Print the resolved config file path")

    get_p = actions.add_parser("get", help="Print the value at a dotted key")
    get_p.add_argument("key", help="Dotted key (e.g. briefing.time, interests.topics)")

    set_p = actions.add_parser("set", help="Update the value at a dotted key")
    set_p.add_argument("key")
    set_p.add_argument("value")

    actions.add_parser("list", help="List all config keys (secrets masked)")


def _dispatch_config(args: argparse.Namespace) -> int:
    action = args.config_action
    if action == "path":
        return _cmd_config_path(args)
    if action == "get":
        return _cmd_config_get(args)
    if action == "set":
        return _cmd_config_set(args)
    if action == "list":
        return _cmd_config_list(args)
    return EXIT_GENERIC


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="argos")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full crawl → brain → save pipeline")
    run_p.add_argument(
        "--url",
        action="append",
        default=[],
        help="Extra dynamic URL to fetch (repeatable)",
    )
    run_p.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("slack", help="Start the Slack bot (Socket Mode)")

    brief_p = sub.add_parser("brief", help="Dispatch today's briefing to Slack")
    brief_p.add_argument("--channel", default=None, help="Override target Slack channel ID")

    _build_config_parser(sub)
    _build_init_parser(sub)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "run":
        return asyncio.run(_run(args.url))
    if args.command == "slack":
        from argos.main import main as slack_main

        asyncio.run(slack_main())
        return 0
    if args.command == "brief":
        from argos.slack.briefing import dispatch_daily_briefing

        ts = asyncio.run(dispatch_daily_briefing(channel=args.channel))
        if ts:
            print(f"Briefing sent: ts={ts}")
        else:
            print("No items today — briefing skipped")
        return 0
    if args.command == "config":
        return _dispatch_config(args)
    if args.command == "init":
        return _cmd_init(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
