from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover — py<3.11 fallback
    import tomli as tomllib  # type: ignore[no-reuse-import]

from pydantic import ValidationError

from argos import config_store
from argos.config import UserConfig, settings
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


def _print_run_summary(summary, elapsed: float) -> None:
    import sys

    source_parts = []
    if "github_trending" in summary.per_source:
        source_parts.append(f"GitHub: {summary.per_source['github_trending']}")
    if "hackernews" in summary.per_source:
        source_parts.append(f"HN: {summary.per_source['hackernews']}")
    for src, cnt in summary.per_source.items():
        if src not in ("github_trending", "hackernews"):
            source_parts.append(f"{src}: {cnt}")
    source_detail = f" ({', '.join(source_parts)})" if source_parts else ""

    queue_total = summary.queue_selected + summary.queue_remaining
    queue_detail = (
        f"{summary.queue_selected}개 / {queue_total}개 (잔여: {summary.queue_remaining}개)"
        if queue_total
        else f"{summary.queue_selected}개"
    )

    if sys.stdout.isatty():
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="bold cyan", no_wrap=True)
            table.add_column()

            table.add_row("신규 크롤링", f"{summary.crawled_total}개{source_detail}")
            table.add_row("일일 처리", queue_detail)
            if summary.preflight_filtered:
                table.add_row("사전 필터 제거", f"{summary.preflight_filtered}개")
            table.add_row("트리아지 통과", f"{summary.triage_pass}개")
            table.add_row("신규 저장", f"{summary.saved_new}개")
            if summary.trust_skipped:
                table.add_row("신뢰도 낮음 스킵", f"{summary.trust_skipped}개")
            if summary.genealogy_skipped:
                table.add_row("족보 분석 스킵", f"{summary.genealogy_skipped}개 (DB 부족)")
            table.add_row("소요 시간", _format_duration(elapsed))

            console.print()
            console.print("✅ [bold green]argos run 완료[/bold green]")
            console.print(table)
            return
        except ImportError:
            pass

    # Non-TTY / Rich unavailable fallback
    print("✅ argos run 완료")
    print("─────────────────────────────")
    print(f"신규 크롤링: {summary.crawled_total}개{source_detail}")
    print(f"일일 처리: {queue_detail}")
    if summary.preflight_filtered:
        print(f"사전 필터 제거: {summary.preflight_filtered}개")
    print(f"트리아지 통과: {summary.triage_pass}개")
    print(f"신규 저장: {summary.saved_new}개")
    if summary.trust_skipped:
        print(f"신뢰도 낮음 스킵: {summary.trust_skipped}개")
    if summary.genealogy_skipped:
        print(f"족보 분석 스킵: {summary.genealogy_skipped}개 (DB 부족)")
    print(f"소요 시간: {_format_duration(elapsed)}")


async def _dispatch_succession_alerts(alerts: list, session) -> None:
    """Forward succession alerts to the Slack dispatcher (ARG-104).

    Defined as a thin indirection so callers (and unit tests) have a single
    seam to patch.  The full Slack implementation lives in ARG-104; this
    placeholder only logs in environments where Slack creds are not wired up,
    which keeps `argos run` working out-of-the-box.
    """
    if not alerts:
        return
    try:
        from argos.slack.app import build_app
        from argos.slack.services.track_check import post_track_update
    except ImportError as exc:
        logging.getLogger(__name__).warning(
            "succession alerts (%d) skipped — Slack layer unavailable: %r",
            len(alerts), exc,
        )
        return

    try:
        app = build_app()
    except Exception as exc:  # noqa: BLE001
        # build_app() raises when Slack tokens / signing secret are absent.
        # Surface a single warning and move on — succession alerts are
        # advisory and must never block `argos run`.
        logging.getLogger(__name__).warning(
            "succession alerts (%d) skipped — Slack app build failed: %r",
            len(alerts), exc,
        )
        return

    channel = settings.user.slack.channel_id
    if not channel:
        logging.getLogger(__name__).warning(
            "succession alerts (%d) skipped — no Slack channel configured",
            len(alerts),
        )
        return

    try:
        await post_track_update(app, channel, alerts, session)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "succession alert dispatch failed: %r", exc
        )


async def _dispatch_signal_matches(new_item_ids: list, session) -> None:
    """Forward signal matches to the Slack dispatcher (ARG-117).

    Calls ``match_signals`` to find Keep-ed assets similar to the newly-saved
    TechItems, then dispatches Slack messages via ``post_signal_update``.
    Mirrors the pattern of :func:`_dispatch_succession_alerts`:

    - ``build_app()`` failure → warning log, skip (no Slack creds).
    - Empty channel → warning log, skip.
    - Dispatcher exception → warning log, skip (never blocks the run).
    """
    if not new_item_ids:
        return
    try:
        from argos.slack.app import build_app
        from argos.slack.services.track_check import match_signals, post_signal_update
    except ImportError as exc:
        logging.getLogger(__name__).warning(
            "signal match dispatch skipped — Slack layer unavailable: %r", exc
        )
        return

    # Validate Slack readiness before the expensive pgvector scan so we skip
    # the DB work entirely when dispatch is impossible.
    try:
        app = build_app()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "signal match dispatch skipped — Slack app build failed: %r", exc,
        )
        return

    channel = settings.user.slack.channel_id
    if not channel:
        logging.getLogger(__name__).warning(
            "signal match dispatch skipped — no Slack channel configured",
        )
        return

    try:
        matches = await match_signals(session, new_item_ids)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "signal match query failed: %r", exc
        )
        return

    if not matches:
        return

    logging.getLogger(__name__).info(
        "signal match: %d match(es) found for %d new item(s)",
        len(matches), len(new_item_ids),
    )

    try:
        await post_signal_update(app, channel, matches, session)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "signal match dispatch failed: %r", exc
        )


async def _run(dynamic_urls: list[str] | None) -> int:
    from argos.progress import ProgressReporter

    start = time.monotonic()
    progress = ProgressReporter()
    with progress:
        async with AsyncSessionLocal() as session:
            results, summary = await run_full_pipeline(
                session,
                dynamic_urls=dynamic_urls or None,
                progress=progress,
            )
            # ARG-103: forward succession alerts to the Slack dispatcher (ARG-104).
            # Pass the same session so post_track_update can write track_history
            # rows transactionally with the rest of the run.
            alerts = getattr(summary, "succession_alerts", None) or []
            if alerts:
                await _dispatch_succession_alerts(alerts, session)
                # commit any track_history rows written by the dispatcher
                try:
                    await session.commit()
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "commit after succession alert dispatch failed",
                        exc_info=True,
                    )

            # ARG-117: signal-match dispatch — compare newly-saved items against
            # Keep-ed assets using pgvector cosine similarity.
            new_item_ids = [
                s["saved_item_id"]
                for s in results
                if s.get("saved") and s.get("saved_item_id")
            ]
            if new_item_ids:
                await _dispatch_signal_matches(new_item_ids, session)
                try:
                    await session.commit()
                except Exception:  # noqa: BLE001
                    logging.getLogger(__name__).warning(
                        "commit after signal match dispatch failed",
                        exc_info=True,
                    )
    elapsed = time.monotonic() - start
    _print_run_summary(summary, elapsed)
    return 0


def _resolve_config_path(args: argparse.Namespace) -> Path:
    override = getattr(args, "config", None)
    if override:
        return Path(override).expanduser()
    return config_store.default_config_path()


def _apply_config_override(args: argparse.Namespace) -> int | None:
    """If ``--config <path>`` was passed, reload ``settings.user`` from it.

    The scheduled launchd jobs invoke ``argos run --config <path>`` and
    ``argos brief --config <path>`` (see ``scheduler.reload_schedule``).
    Without this, the runtime would silently fall back to defaults /
    ``~/.config/argos/config.toml`` even though the operator explicitly
    pointed at a different file.

    When ``--config`` is explicit, uses :meth:`UserConfig.load_strict` so
    TOML/schema errors surface cleanly instead of silently falling back to
    defaults.  Returns ``EXIT_GENERIC`` on error so the caller can propagate
    it; returns ``None`` on success (or when no override was supplied).
    """
    override = getattr(args, "config", None)
    if not override:
        return None
    path = Path(override).expanduser()
    try:
        settings.user = UserConfig.load_strict(path=path)
    except FileNotFoundError:
        print(f"Config file not found: {path}", file=sys.stderr)
        return EXIT_GENERIC
    except tomllib.TOMLDecodeError as exc:
        print(f"Invalid TOML in {path}: {exc}", file=sys.stderr)
        return EXIT_GENERIC
    except ValidationError as exc:
        first = str(exc).strip().splitlines()[0]
        print(f"Invalid config in {path}: {first}", file=sys.stderr)
        return EXIT_GENERIC
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Could not read config file {path}: {exc}", file=sys.stderr)
        return EXIT_GENERIC
    return None


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


def _cmd_config_migrate_env(args: argparse.Namespace) -> int:
    """Migrate a repo-root .env to the XDG location atomically.

    Copies the source file to ``~/.config/argos/.env`` (or the
    ``XDG_CONFIG_HOME``-derived path), preserving ``0600`` permissions, then
    renames the source to ``<source>.bak`` so it no longer shadows the XDG
    copy at runtime.
    """
    import shutil
    import stat as _stat

    dest = config_store.default_env_path()

    # Resolve source path: --from flag overrides; default is cwd/.env.
    from_arg = getattr(args, "from_path", None)
    src = Path(from_arg).expanduser() if from_arg else Path(".env").resolve()

    if not src.exists():
        print(f"Source .env not found: {src}", file=sys.stderr)
        return EXIT_GENERIC

    # Idempotency guard: if the destination already exists and is newer than
    # the source, skip the migration to avoid overwriting a more recent file.
    if dest.exists():
        src_mtime = src.stat().st_mtime
        dest_mtime = dest.stat().st_mtime
        if dest_mtime >= src_mtime:
            print(
                f"XDG .env already exists and is up-to-date: {dest}\n"
                "Nothing to migrate.  Delete the destination first if you "
                "want to force a re-migration."
            )
            return EXIT_OK

    # Atomic copy: write to a temp sibling of the destination, then replace.
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        shutil.copy2(src, tmp)
        os.chmod(tmp, 0o600)
        os.replace(tmp, dest)
        # Ensure final destination is 0600 even if it pre-existed with looser perms.
        os.chmod(dest, 0o600)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # Rename source to <source>.bak so it no longer shadows the XDG copy.
    bak = Path(str(src) + ".bak")
    try:
        os.rename(src, bak)
    except OSError as exc:
        # Migration already complete; just warn about the bak rename failure.
        print(
            f"Warning: could not rename {src} to {bak}: {exc}\n"
            f"The .env was copied to {dest} but the original was not renamed.",
            file=sys.stderr,
        )
        return EXIT_OK

    print(f"Migrated: {src} -> {dest}")
    print(f"Original backed up at: {bak}")
    print(f"Delete {bak} when you're sure the migration worked.")
    # Verify destination permissions.
    mode = _stat.S_IMODE(dest.stat().st_mode)
    if mode != 0o600:  # pragma: no cover - defensive; chmod above should guarantee this
        print(
            f"Warning: destination permissions are {oct(mode)}, expected 0600.  "
            f"Run: chmod 600 {dest}",
            file=sys.stderr,
        )
    return EXIT_OK


def _build_doctor_parser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``argos doctor`` subcommand."""
    doctor_p = sub.add_parser(
        "doctor",
        help="Run pre-flight health probes (Docker, Ollama, Python, macOS)",
        description=(
            "Run a read-only structured check of every prerequisite Argos needs.\n\n"
            "Probes: Docker daemon, Ollama installed, required models pulled\n"
            "(qwen3:8b, qwen3:32b, nomic-embed-text), Python version,\n"
            "macOS version (warn-only). Prints a table and exits 0 only when no probe FAILs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doctor_p.add_argument(
        "--config",
        default=None,
        help="Path to config.toml (defaults to ~/.config/argos/config.toml)",
    )


def _cmd_doctor(args: argparse.Namespace) -> int:
    from argos.doctor import (
        check_docker,
        check_macos_version,
        check_ollama_installed,
        check_ollama_models,
        check_python_version,
        check_uv_installed,
        print_doctor_table,
    )

    rc = _apply_config_override(args)
    if rc is not None:
        return rc

    rows = [
        check_docker(),
        check_ollama_installed(),
        *check_ollama_models(ollama_host=settings.user.ollama.host),
        check_python_version(),
        check_macos_version(),
        check_uv_installed(),
    ]

    print_doctor_table(rows)
    failures = sum(1 for _, status, _ in rows if status == "FAIL")
    return 0 if failures == 0 else 1


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

    migrate_env_p = actions.add_parser(
        "migrate-env",
        help="Move repo-root .env to ~/.config/argos/.env (XDG location)",
        description=(
            "Copies the repo-root .env to the XDG location "
            "(${XDG_CONFIG_HOME:-~/.config}/argos/.env) atomically with 0600 "
            "permissions, then renames the source to <source>.bak so it no "
            "longer shadows the XDG copy at runtime.\n\n"
            "Idempotent: if the destination already exists and is newer than "
            "the source, the command prints a message and exits 0 without "
            "modifying anything."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    migrate_env_p.add_argument(
        "--from",
        dest="from_path",
        default=None,
        metavar="PATH",
        help="Source .env path (default: ./.env in the current directory)",
    )


def _positive_int(value: str) -> int:
    """Argparse type that rejects non-positive integers."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer")
    if n < 1:
        raise argparse.ArgumentTypeError(f"--limit must be ≥ 1, got {n}")
    return n


def _build_search_parser(
    sub: argparse._SubParsersAction,
    common: argparse.ArgumentParser,
) -> None:
    """Wire the ``argos search`` subcommand."""
    search_p = sub.add_parser(
        "search",
        help="Search collected tech_items by semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Embeds the query via nomic-embed-text and returns the most similar\n"
            "tech_items ranked by cosine distance.\n\n"
            "Example:\n  argos search \"RAG\" --category alpha --status keep"
        ),
        parents=[common],
    )
    search_p.add_argument("query", help="Natural-language search query")
    search_p.add_argument(
        "--limit",
        type=_positive_int,
        default=10,
        metavar="N",
        help="Max results to return, must be ≥ 1 (default: 10, max: 50)",
    )
    search_p.add_argument(
        "--category",
        choices=["alpha", "mainstream"],
        default=None,
        help="Filter to a specific category",
    )
    search_p.add_argument(
        "--status",
        choices=["keep", "all"],
        default="all",
        help="Filter by asset status: keep|all (default: all)",
    )


async def _search(query: str, limit: int, category: str | None, status: str) -> int:
    from argos.brain.ollama_client import embed as ollama_embed
    from argos.services.search import search_tech_items

    try:
        embedding = await ollama_embed(query)
    except Exception as e:
        print(f"❌ Ollama 연결 실패: {e}", file=sys.stderr)
        print("Ollama가 실행 중인지 확인하세요: ollama serve", file=sys.stderr)
        return 1

    async with AsyncSessionLocal() as session:
        results = await search_tech_items(
            session,
            embedding,
            limit=limit,
            category=category,
            status=status,
        )

    if not results:
        print("검색 결과 없음.")
        return 0

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(show_header=True)
        table.add_column("title", max_width=40, no_wrap=False)
        table.add_column("trust", justify="right")
        table.add_column("category")
        table.add_column("status")
        table.add_column("date")

        for r in results:
            trust_str = f"{r.trust_score:.2f}" if r.trust_score is not None else "—"
            table.add_row(
                r.title,
                trust_str,
                r.category or "—",
                r.status or "—",
                r.created_at.strftime("%Y-%m-%d"),
            )

        console.print(table)
    except ImportError:
        print(f"{'title':<40} {'trust':>5}  {'category':<12}  {'status':<10}  date")
        print("─" * 80)
        for r in results:
            trust_str = f"{r.trust_score:.2f}" if r.trust_score is not None else "—"
            print(
                f"{r.title[:38]:<40} {trust_str:>5}  "
                f"{(r.category or '—'):<12}  {(r.status or '—'):<10}  "
                f"{r.created_at.strftime('%Y-%m-%d')}"
            )

    return 0


def _build_portfolio_parser(
    sub: argparse._SubParsersAction,
    common: argparse.ArgumentParser,
) -> None:
    """Wire the ``argos portfolio`` subcommand (ARG-113)."""
    portfolio_p = sub.add_parser(
        "portfolio",
        help="Display your Keep portfolio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "List all tech_items you have marked Keep, grouped by category.\n\n"
            "Example:\n"
            "  argos portfolio\n"
            "  argos portfolio --category alpha\n"
            "  argos portfolio --sort trust"
        ),
        parents=[common],
    )
    portfolio_p.add_argument(
        "--category",
        choices=["alpha", "mainstream"],
        default=None,
        help="Filter to a specific category: alpha|mainstream",
    )
    portfolio_p.add_argument(
        "--sort",
        choices=["date", "trust"],
        default="date",
        dest="sort",
        help="Sort order: date (default, newest first) | trust (highest trust first)",
    )


async def _portfolio(category: str | None, sort: str) -> int:
    from argos.models.tech_item import CategoryType
    from argos.slack.services.briefing_query import fetch_user_portfolio

    category_enum: CategoryType | None = None
    if category is not None:
        if category.lower() == "alpha":
            category_enum = CategoryType.ALPHA
        else:
            category_enum = CategoryType.MAINSTREAM

    async with AsyncSessionLocal() as session:
        results = await fetch_user_portfolio(
            session,
            category=category_enum,
            sort_by=sort,  # type: ignore[arg-type]
        )

    if not results:
        print("Keep된 자산이 없습니다. Slack에서 항목을 Keep해 포트폴리오를 만들어보세요.")
        return 0

    # Group by category
    from argos.models.tech_item import CategoryType as CT

    grouped: dict[CT, list] = {CT.MAINSTREAM: [], CT.ALPHA: []}
    for asset, item in results:
        cat = item.category if item.category in grouped else CT.MAINSTREAM
        grouped[cat].append((asset, item))

    total = len(results)

    try:
        from rich.console import Console
        from rich.markup import escape

        console = Console()
        console.print(f"\n[bold]# Keep 포트폴리오 (총 {total}개)[/bold]\n")

        for cat in (CT.MAINSTREAM, CT.ALPHA):
            items = grouped[cat]
            if not items:
                continue
            label = "Mainstream" if cat == CT.MAINSTREAM else "Alpha"
            console.print(f"[bold cyan]\\[{label}][/bold cyan]")
            for asset, item in items:
                kept_date = asset.created_at.strftime("%Y-%m-%d")
                last_signal = (
                    asset.last_monitored_at.strftime("%Y-%m-%d")
                    if asset.last_monitored_at
                    else "—"
                )
                console.print(f"• {escape(item.title):<30}  kept {kept_date}  last_signal {last_signal}")
            console.print()

    except ImportError:
        print(f"\n# Keep 포트폴리오 (총 {total}개)\n")
        for cat in (CT.MAINSTREAM, CT.ALPHA):
            items = grouped[cat]
            if not items:
                continue
            label = "Mainstream" if cat == CT.MAINSTREAM else "Alpha"
            print(f"[{label}]")
            for asset, item in items:
                kept_date = asset.created_at.strftime("%Y-%m-%d")
                last_signal = (
                    asset.last_monitored_at.strftime("%Y-%m-%d")
                    if asset.last_monitored_at
                    else "—"
                )
                print(f"• {item.title:<30}  kept {kept_date}  last_signal {last_signal}")
            print()

    return 0


def _build_add_parser(
    sub: argparse._SubParsersAction,
    common: argparse.ArgumentParser,
) -> None:
    """Wire the ``argos add <URL>`` subcommand (ARG-107)."""
    add_p = sub.add_parser(
        "add",
        help="Manually add a URL to the brain pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Fetch a URL and feed it through the same brain pipeline (triage →\n"
            "embed → genealogist → save) used by the daily crawler.  Useful as an\n"
            "escape hatch for items the automated crawler missed.\n\n"
            "Multiple URLs can be supplied positionally or via repeated --url.\n"
            "Each URL is validated (scheme/SSRF/robots) and deduplicated against\n"
            "tech_items before fetching.\n\n"
            "Examples:\n"
            "  argos add https://example.com/post\n"
            "  argos add https://a.test/1 https://a.test/2\n"
            "  argos add --url https://a.test/1 --url https://a.test/2\n\n"
            "Exit codes:\n"
            "  0  every URL ended up created or duplicate (idempotent success)\n"
            "  1  at least one URL was rejected or errored"
        ),
        parents=[common],
    )
    add_p.add_argument(
        "urls",
        nargs="*",
        metavar="URL",
        help="URL(s) to add (also accepts --url; positional and --url may be combined)",
    )
    add_p.add_argument(
        "--url",
        dest="extra_urls",
        action="append",
        default=[],
        metavar="URL",
        help="Additional URL to add (repeatable)",
    )


async def _add(urls: list[str]) -> int:
    from argos.crawler import add_url as add_url_module
    from argos.crawler.add_url import AddUrlStatus

    if not urls:
        # argparse with nargs='*' allows zero positionals; we want at least one.
        print("argos add: at least one URL is required", file=sys.stderr)
        return EXIT_GENERIC

    async with AsyncSessionLocal() as session:
        results = []
        for url in urls:
            result = await add_url_module.add_url(url, session)
            results.append(result)

    _print_add_results(results)

    failure = any(
        r.status in (AddUrlStatus.REJECTED, AddUrlStatus.ERROR) for r in results
    )
    return EXIT_GENERIC if failure else EXIT_OK


def _print_add_results(results: list) -> None:
    """Render the per-URL result table to stdout."""
    from argos.crawler.add_url import AddUrlStatus

    _STATUS_STYLE: dict[AddUrlStatus, str] = {
        AddUrlStatus.CREATED: "green",
        AddUrlStatus.DUPLICATE: "cyan",
        AddUrlStatus.REJECTED: "yellow",
        AddUrlStatus.ERROR: "red",
    }

    def _fmt_id(tid) -> str:
        if tid is None:
            return "—"
        s = str(tid)
        return s[:8] + "…" if len(s) > 9 else s

    if sys.stdout.isatty():
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(show_header=True)
            table.add_column("URL", max_width=60, no_wrap=False)
            table.add_column("status")
            table.add_column("tech_item_id")
            table.add_column("reason", max_width=40, no_wrap=False)

            for r in results:
                style = _STATUS_STYLE.get(r.status, "")
                status_cell = (
                    f"[{style}]{r.status.value}[/{style}]" if style else r.status.value
                )
                table.add_row(
                    r.url,
                    status_cell,
                    _fmt_id(r.tech_item_id),
                    r.reason or "",
                )
            console.print(table)
            return
        except ImportError:
            pass

    # Plain fallback (non-TTY or Rich unavailable).
    print(f"{'URL':<60} {'status':<10} {'tech_item_id':<10} reason")
    print("─" * 110)
    for r in results:
        print(
            f"{r.url[:58]:<60} {r.status.value:<10} {_fmt_id(r.tech_item_id):<10} "
            f"{r.reason or ''}"
        )


def _build_schedule_parser(sub: argparse._SubParsersAction) -> None:
    schedule_p = sub.add_parser(
        "schedule",
        help="Install/remove the launchd jobs for `argos run` and `argos brief`",
        description=(
            "Manage the macOS launchd schedule for Argos.\n\n"
            "Actions:\n"
            "  install    Render + bootstrap both plists from the current config.\n"
            "  uninstall  Bootout both plists (no error if already absent).\n"
            "  status     Print loaded/not-loaded for both labels."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    schedule_p.add_argument(
        "--config",
        default=None,
        help="Path to config.toml (defaults to ~/.config/argos/config.toml)",
    )
    actions = schedule_p.add_subparsers(dest="schedule_action", required=True)
    actions.add_parser("install", help="Install + bootstrap both launchd jobs")
    actions.add_parser("uninstall", help="Bootout both launchd jobs")
    actions.add_parser("status", help="Show loaded/not-loaded for both labels")


def _cmd_schedule_install(args: argparse.Namespace) -> int:
    from argos.scheduler import SchedulerError, reload_schedule

    explicit_config = bool(getattr(args, "config", None))
    path = _resolve_config_path(args)

    if explicit_config:
        # launchd runs jobs with its own cwd, so any relative path baked into
        # the plist's ProgramArguments would resolve differently (or not at all)
        # at scheduled-run time.  When the operator explicitly supplied a path,
        # resolve strictly — a missing file is a hard error here because the
        # resulting plists would embed a broken path.
        try:
            path = path.resolve(strict=True)
        except FileNotFoundError:
            print(f"Config file not found: {path}", file=sys.stderr)
            return EXIT_GENERIC
        # Refuse to silently fall back to defaults on parse/validation errors —
        # the operator pointed at a specific file and the plists must reflect
        # *its* settings, not whatever the defaults happen to be.
        try:
            user_config = UserConfig.load_strict(path=path)
        except tomllib.TOMLDecodeError as exc:
            print(f"Invalid TOML in {path}: {exc}", file=sys.stderr)
            return EXIT_GENERIC
        except ValidationError as exc:
            first = str(exc).strip().splitlines()[0]
            print(f"Invalid config in {path}: {first}", file=sys.stderr)
            return EXIT_GENERIC
        except (OSError, UnicodeDecodeError) as exc:
            print(f"Could not read config file {path}: {exc}", file=sys.stderr)
            return EXIT_GENERIC
        embed_config_path: Path | None = path
    else:
        # No explicit --config: the default path may not exist yet (fresh
        # machine / first run).  Resolve without strict so we always get an
        # absolute path, and load with the swallow-default behavior so a
        # missing file is treated as "use defaults".
        path = path.resolve()
        user_config = UserConfig.load(path=path)
        # Embed --config in the plist ONLY when the default file is both present
        # AND parseable/valid.  If the file exists but is broken TOML or fails
        # schema validation, UserConfig.load silently fell back to defaults above
        # but the scheduled `argos run --config <path>` would hit the strict
        # _apply_config_override path and exit non-zero on every launchd trigger.
        # Probe with load_strict; any failure → omit --config so the plist runs
        # bare `argos run` which uses the permissive load path (same as defaults).
        embed_config_path: Path | None = None
        if path.is_file():
            try:
                UserConfig.load_strict(path=path)  # probe only — result discarded
                embed_config_path = path
            except (
                FileNotFoundError,
                tomllib.TOMLDecodeError,
                ValidationError,
                OSError,
                UnicodeDecodeError,
            ):
                embed_config_path = None  # invalid file → don't embed --config

    try:
        # Plumb the resolved config path through so the generated plists
        # invoke `argos run --config <path>` / `argos brief --config <path>`
        # with the same settings the install command just validated.
        # embed_config_path is None when the default file is absent — renderers
        # then omit the --config arg so the scheduled job uses the same
        # permissive default-load path as a bare `argos run`.
        reload_schedule(user_config, config_path=embed_config_path)
    except SchedulerError as exc:
        print(f"Scheduler error: {exc}", file=sys.stderr)
        return EXIT_GENERIC
    print("Scheduled: com.argos.run, com.argos.brief, com.argos.brief-weekly")
    return EXIT_OK


def _cmd_schedule_uninstall(_args: argparse.Namespace) -> int:
    from argos.scheduler import SchedulerError, bootout_plist

    failures: list[str] = []
    for label in ("com.argos.run", "com.argos.brief", "com.argos.brief-weekly"):
        try:
            bootout_plist(label)
            print(f"Unloaded: {label}")
        except SchedulerError as exc:
            failures.append(f"{label}: {exc}")
            print(f"Failed to unload {label}: {exc}", file=sys.stderr)
    return EXIT_GENERIC if failures else EXIT_OK


def _cmd_schedule_status(_args: argparse.Namespace) -> int:
    from argos.scheduler import is_loaded

    for label in ("com.argos.run", "com.argos.brief", "com.argos.brief-weekly"):
        state = "loaded" if is_loaded(label) else "not loaded"
        print(f"{label}: {state}")
    return EXIT_OK


def _dispatch_schedule(args: argparse.Namespace) -> int:
    action = args.schedule_action
    if action == "install":
        return _cmd_schedule_install(args)
    if action == "uninstall":
        return _cmd_schedule_uninstall(args)
    if action == "status":
        return _cmd_schedule_status(args)
    return EXIT_GENERIC


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
    if action == "migrate-env":
        return _cmd_config_migrate_env(args)
    return EXIT_GENERIC


def _resolve_version() -> str:
    """Return the installed package version, falling back gracefully for dev installs."""
    import importlib.metadata

    try:
        return importlib.metadata.version("argos-scout")
    except importlib.metadata.PackageNotFoundError:
        pass
    # Editable install without dist-info: try reading pyproject.toml directly.
    try:
        _here = Path(__file__).parent.parent.parent  # src/argos -> src -> repo root
        _pyproject = _here / "pyproject.toml"
        with open(_pyproject, "rb") as _f:
            _data = tomllib.load(_f)
        return _data.get("project", {}).get("version", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="argos")
    parser.add_argument(
        "--version",
        action="version",
        version=f"argos {_resolve_version()}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared `--config` flag for runtime subcommands. The scheduled launchd
    # jobs pass `--config <path>` (rendered by `scheduler.reload_schedule`),
    # so the parsers MUST accept it or the jobs crash with argparse exit 2.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default=None,
        help="Path to config.toml (defaults to ~/.config/argos/config.toml)",
    )

    run_p = sub.add_parser(
        "run",
        help="Run the full crawl → brain → save pipeline",
        parents=[common],
    )
    run_p.add_argument(
        "--url",
        action="append",
        default=[],
        help="Extra dynamic URL to fetch (repeatable)",
    )
    run_p.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("slack", help="Start the Slack bot (Socket Mode)", parents=[common])

    brief_p = sub.add_parser(
        "brief",
        help="Dispatch today's briefing to Slack",
        parents=[common],
    )
    brief_p.add_argument("--channel", default=None, help="Override target Slack channel ID")
    brief_p.add_argument(
        "--weekly",
        action="store_true",
        help="Send the weekly Keep portfolio summary instead of the daily briefing",
    )

    _build_add_parser(sub, common)
    _build_config_parser(sub)
    _build_doctor_parser(sub)
    _build_init_parser(sub)
    _build_portfolio_parser(sub, common)
    _build_schedule_parser(sub)
    _build_search_parser(sub, common)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "run":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        return asyncio.run(_run(args.url))
    if args.command == "slack":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        from argos.main import main as slack_main

        asyncio.run(slack_main())
        return 0
    if args.command == "brief":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        if getattr(args, "weekly", False):
            from argos.slack.briefing import dispatch_weekly_briefing

            ts = asyncio.run(dispatch_weekly_briefing(channel=args.channel))
            if ts:
                print(f"Weekly briefing sent: ts={ts}")
            else:
                print("Weekly briefing dispatch returned no ts")
            return 0
        from argos.slack.briefing import dispatch_daily_briefing

        ts = asyncio.run(dispatch_daily_briefing(channel=args.channel))
        if ts:
            print(f"Briefing sent: ts={ts}")
        else:
            print("No items today — briefing skipped")
        return 0
    if args.command == "config":
        return _dispatch_config(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "schedule":
        return _dispatch_schedule(args)
    if args.command == "search":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        return asyncio.run(_search(args.query, args.limit, args.category, args.status))
    if args.command == "add":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        # Combine positional URLs and --url options, preserving order, dedup.
        all_urls: list[str] = []
        seen: set[str] = set()
        for u in [*args.urls, *args.extra_urls]:
            if u in seen:
                continue
            seen.add(u)
            all_urls.append(u)
        return asyncio.run(_add(all_urls))
    if args.command == "portfolio":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        return asyncio.run(_portfolio(args.category, args.sort))
    return 1


if __name__ == "__main__":
    sys.exit(main())
