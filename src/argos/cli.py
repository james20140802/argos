from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

try:
    import tomllib
except ImportError:  # pragma: no cover — py<3.11 fallback
    import tomli as tomllib  # type: ignore[no-reuse-import]

from pydantic import ValidationError

from argos import backup, config_store
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


def _print_run_summary(summary, elapsed: float, failed: bool = False) -> None:
    import sys

    # A failed run must NOT leave the success marker in the log — `argos status`
    # keys on it. Emit an explicit failure header on the SAME (stdout) stream so
    # the status parser sees a deterministic, correctly-ordered outcome marker.
    header = "❌ argos run 실패 — 트리아지 인프라 오류" if failed else "✅ argos run 완료"

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
            if failed:
                console.print("❌ [bold red]argos run 실패 — 트리아지 인프라 오류[/bold red]")
            else:
                console.print("✅ [bold green]argos run 완료[/bold green]")
            console.print(table)
            return
        except ImportError:
            pass

    # Non-TTY / Rich unavailable fallback — this is the launchd path that writes
    # run.log, so the failure header here is what `argos status` keys on.
    print(header)
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


async def _run(
    dynamic_urls: list[str] | None,
    *,
    verbose: bool = False,
    console: "Console | None" = None,
) -> int:
    """Run the full crawl → brain → save pipeline.

    Parameters
    ----------
    dynamic_urls:
        Extra URLs to fetch (forwarded to the crawler).
    verbose:
        When True, set root log level to DEBUG (shows httpx INFO etc.).
        When False (default), TTY runs use WARNING (Rich bar handles feedback)
        and non-TTY runs use INFO so ProgressReporter._emit lines remain
        visible in launchd/CI/redirected-stdout contexts.
    console:
        Optional Rich Console to use for both the progress bar and the
        RichHandler. Injection point for tests; production callers leave this
        as None so a Console is constructed automatically.
    """
    from rich.console import Console as _Console
    from rich.logging import RichHandler

    from argos.progress import ProgressReporter

    # Build or reuse a single Rich Console so that RichHandler and Progress
    # share the same Live display. This is the documented fix for log output
    # corrupting/duplicating the Rich progress bar (ARG-114).
    shared_console = console if console is not None else _Console()

    # In non-TTY contexts (launchd/CI), ProgressReporter falls back to
    # logger.info() for in-flight progress. Keep INFO visible there; suppress
    # to WARNING only when the Rich bar is active so its noise doesn't appear.
    if verbose:
        log_level = logging.DEBUG
    elif shared_console.is_terminal:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=shared_console, show_path=False)],
        force=True,
    )

    start = time.monotonic()
    # Use the console's own TTY detection rather than sys.stdout.isatty() so
    # that test-injected force_terminal consoles correctly enable the Rich
    # progress bar. Production callers with a real Console get the same result.
    progress = ProgressReporter(tty=shared_console.is_terminal, console=shared_console)
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

            # ARG-210: 교차 검증 — 최근 아이템의 다른-도메인 유사 수 갱신 + 재합성.
            from argos.brain.corroboration import update_corroboration
            try:
                updated = await update_corroboration(session)
                await session.commit()
                if updated:
                    logging.getLogger(__name__).info(
                        "corroboration: %d item(s) recomputed", updated
                    )
            except Exception:  # noqa: BLE001
                # Roll back so a corroboration failure leaves the session
                # usable — otherwise the pending-rollback state would make the
                # feed_score rescore below fail too, coupling two independent
                # best-effort steps (whole-branch review finding).
                await session.rollback()
                logging.getLogger(__name__).warning(
                    "corroboration update failed", exc_info=True
                )

            # ARG-212: feed_score 일괄 재계산 (corroboration_count 채운 뒤).
            from argos.brain.feed_ranking import recompute_feed_scores
            try:
                scored = await recompute_feed_scores(session)
                await session.commit()
                if scored:
                    logging.getLogger(__name__).info(
                        "feed_score: %d item(s) rescored", scored
                    )
            except Exception:  # noqa: BLE001
                await session.rollback()
                logging.getLogger(__name__).warning(
                    "feed_score recompute failed", exc_info=True
                )
    elapsed = time.monotonic() - start
    run_failed = any(s.get("triage_error") for s in results)
    _print_run_summary(summary, elapsed, failed=run_failed)
    if run_failed:
        logging.getLogger(__name__).warning(
            "argos run: Ollama infra error during triage; crawl queue preserved "
            "for retry, reporting non-zero exit."
        )
        return 1
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


def _build_backup_parser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``argos backup`` subcommand (ARG-192)."""
    backup_p = sub.add_parser(
        "backup",
        help="Dump the Postgres DB via `docker exec ... pg_dump -Fc`",
        description=(
            "Create a custom-format (pg_dump -Fc) dump of the Argos database by "
            "shelling out to `docker exec <container> pg_dump`. The dump lands in "
            "the XDG data directory (~/.local/share/argos/backups/) with a "
            "timestamped filename. Restore it later with `argos restore <dump>`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    backup_p.add_argument(
        "--container",
        default=backup.DEFAULT_CONTAINER_NAME,
        help=f"Docker container name to exec into (default: {backup.DEFAULT_CONTAINER_NAME})",
    )
    backup_p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to write the dump into (default: ~/.local/share/argos/backups)",
    )
    backup_p.add_argument(
        "--keep",
        type=int,
        default=None,
        metavar="N",
        help="After a successful backup, delete older dumps beyond the N most recent",
    )


def _cmd_backup(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    try:
        dest = backup.create_backup(
            container=args.container,
            output_dir=output_dir,
            keep=args.keep,
        )
    except backup.BackupError as exc:
        print(f"backup 실패: {exc}", file=sys.stderr)
        return EXIT_GENERIC
    print(f"✅ backup 완료: {dest}")
    return EXIT_OK


def _build_restore_parser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``argos restore`` subcommand (ARG-192)."""
    restore_p = sub.add_parser(
        "restore",
        help="Restore a pg_dump archive into the Postgres DB (DESTRUCTIVE)",
        description=(
            "Restore a dump created by `argos backup` via `docker exec ... pg_restore`.\n\n"
            "DESTRUCTIVE: by default this drops and recreates existing objects "
            "(--clean --if-exists) before restoring, overwriting current data in the "
            "target database. Prompts for confirmation unless --yes is passed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    restore_p.add_argument("dump", help="Path to a dump file created by `argos backup`")
    restore_p.add_argument(
        "--container",
        default=backup.DEFAULT_CONTAINER_NAME,
        help=f"Docker container name to exec into (default: {backup.DEFAULT_CONTAINER_NAME})",
    )
    restore_p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    restore_p.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip pg_restore --clean --if-exists (restore additively instead of overwriting)",
    )


def _cmd_restore(args: argparse.Namespace) -> int:
    dump_path = Path(args.dump).expanduser()
    if not dump_path.exists():
        print(f"restore 실패: 파일을 찾을 수 없습니다: {dump_path}", file=sys.stderr)
        return EXIT_GENERIC

    if not args.yes:
        print(
            f"⚠️  '{args.container}' 컨테이너의 DB를 '{dump_path}' 내용으로 덮어씁니다. "
            "기존 데이터는 삭제됩니다."
        )
        reply = input("계속하시겠습니까? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("취소되었습니다.")
            return EXIT_GENERIC

    try:
        backup.restore_backup(
            dump_path,
            container=args.container,
            clean=not args.no_clean,
        )
    except backup.BackupError as exc:
        print(f"restore 실패: {exc}", file=sys.stderr)
        return EXIT_GENERIC
    print(f"✅ restore 완료: {dump_path}")
    return EXIT_OK


def _build_doctor_parser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``argos doctor`` subcommand."""
    doctor_p = sub.add_parser(
        "doctor",
        help="Run pre-flight health probes (Docker, Ollama, Postgres, alembic, VRAM, Python, macOS)",
        description=(
            "Run a read-only structured check of every prerequisite Argos needs.\n\n"
            "Probes: Docker daemon, Ollama installed, required models pulled\n"
            "(qwen3:8b, qwen3:32b, nomic-embed-text), Python version,\n"
            "macOS version (warn-only), uv installed, Postgres reachable,\n"
            "alembic migrations up to date, VRAM headroom (warn-only).\n"
            "Prints a table and exits 0 only when no probe FAILs."
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
        check_alembic_head,
        check_docker,
        check_macos_version,
        check_ollama_installed,
        check_ollama_models,
        check_postgres_reachable,
        check_python_version,
        check_uv_installed,
        check_vram_headroom,
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
        check_postgres_reachable(),
        check_alembic_head(),
        check_vram_headroom(ollama_host=settings.user.ollama.host),
    ]

    print_doctor_table(rows)
    failures = sum(1 for _, status, _ in rows if status == "FAIL")
    return 0 if failures == 0 else 1


def _build_status_parser(sub: argparse._SubParsersAction) -> None:
    """Wire the top-level ``argos status`` subcommand (ARG-221).

    Distinct from ``argos schedule status`` (launchd load state); this
    summarises the last scheduled run/brief results from their logs.
    """
    sub.add_parser(
        "status",
        help="Summarise the last scheduled run/brief results from their logs",
        description=(
            "Read ~/Library/Logs/argos/{run,brief,brief-weekly}.log and print the "
            "last result, last success time, and processed counts for each job — "
            "no manual log tailing needed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def _cmd_status(_args: argparse.Namespace) -> int:
    from argos.status import collect_status, render_status

    print(render_status(collect_status()))
    return 0


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


def _tcp_port(value: str) -> int:
    """Argparse type that accepts a valid TCP port (1..65535)."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer")
    if n < 1 or n > 65535:
        raise argparse.ArgumentTypeError(f"--port must be in 1..65535, got {n}")
    return n


def _build_web_parser(
    sub: argparse._SubParsersAction,
    common: argparse.ArgumentParser,
) -> None:
    """Wire the ``argos web`` subcommand (ARG-133)."""
    web_p = sub.add_parser(
        "web",
        help="Start the Argos web app (FastAPI + uvicorn)",
        description=(
            "Run the Argos web layer locally with uvicorn.\n\n"
            "Host/port default to [web].host / [web].port from config "
            "(127.0.0.1:8765). Use --host / --port to override per-invocation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    web_p.add_argument("--host", default=None, help="Override config [web].host")
    web_p.add_argument(
        "--port",
        type=_tcp_port,
        default=None,
        help="Override config [web].port",
    )


def _cmd_web(args: argparse.Namespace) -> int:
    import uvicorn
    from argos.web.app import build_web_app

    rc = _apply_config_override(args)
    if rc is not None:
        return rc

    host = args.host or settings.user.web.host
    port = args.port if args.port is not None else settings.user.web.port

    # Thread the active config path (default or --config) into the app so the
    # settings page reads/writes the same file the running daemon uses.
    app = build_web_app(config_path=_resolve_config_path(args))
    uvicorn.run(app, host=host, port=port)
    return 0


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


# ---------------------------------------------------------------------------
# argos stats  (ARG-66)
# ---------------------------------------------------------------------------


def _build_stats_parser(
    sub: argparse._SubParsersAction,
    common: argparse.ArgumentParser,
) -> None:
    """Wire the ``argos stats`` subcommand (ARG-66)."""
    stats_p = sub.add_parser(
        "stats",
        help="Show collection-status dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Display a summary dashboard: collection counts, brain/triage\n"
            "results, and portfolio + Track-alert statistics.\n\n"
            "Example:\n"
            "  argos stats\n"
            "  argos stats --days 30"
        ),
        parents=[common],
    )
    stats_p.add_argument(
        "--days",
        type=int,
        default=7,
        metavar="N",
        help="Look-back window in days (default: 7; must be a positive integer)",
    )


async def _stats(days: int) -> int:
    """Render the stats dashboard for the last *days* days."""
    from argos.slack.services.stats_query import fetch_stats_summary, safe_pct

    async with AsyncSessionLocal() as session:
        data = await fetch_stats_summary(session, days=days)

    total = data["total_items"]
    github = data["github_count"]
    hn = data["hn_count"]
    rss = data["rss_count"]
    arxiv = data["arxiv_count"]
    valid = data["valid_count"]
    new_saved = data["new_saved_count"]
    keep = data["keep_count"]
    pass_ = data["pass_count"]
    unclassified = data["unclassified_count"]
    cumulative_keep = data["total_keep_cumulative"]
    track_alerts = data["track_alert_count"]

    pct = safe_pct(valid, total)
    pct_str = f"{pct}%" if total > 0 else "0%"

    print(f"📊 Argos 통계 (최근 {days}일)")
    print()
    print(f"수집:      {total}개  (GitHub {github} / HN {hn} / RSS {rss} / arXiv {arxiv})")
    print()
    print(f"유효:      {valid}개  ({pct_str})")
    print(f"저장(신규): {new_saved}개")
    print(f"Keep:      {keep}개  |  Pass: {pass_}개  |  미분류: {unclassified}개")
    print()
    print(f"포트폴리오: 총 {cumulative_keep}개 Keep")
    print(f"Track 알림: 지난 {days}일 {track_alerts}건")

    return 0


def _build_backfill_images_parser(
    sub: argparse._SubParsersAction,
    common: argparse.ArgumentParser,
) -> None:
    """Wire the ``argos backfill-images`` subcommand (ARG-179)."""
    bf_p = sub.add_parser(
        "backfill-images",
        help="Fill image_url for items that have none (favicon by default)",
        parents=[common],
        description=(
            "Fill tech_items.image_url where it is NULL. Default: derive a "
            "domain favicon with no network call. --refetch: re-crawl the "
            "source_url and apply the full image fallback chain (slow). "
            "Existing non-NULL image_url values are never overwritten."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bf_p.add_argument(
        "--refetch",
        action="store_true",
        help="Re-crawl source_url to recover og/body images (network, slow)",
    )
    bf_p.add_argument(
        "--upgrade-favicons",
        action="store_true",
        help=(
            "Re-crawl rows whose image_url is a bare /favicon.ico and replace it "
            "with a real og/twitter/body image when one is found (network, slow). "
            "Rows that still resolve to a favicon are left untouched. Implies "
            "--refetch."
        ),
    )


def _cmd_backfill_images(args: argparse.Namespace) -> int:
    return asyncio.run(
        _backfill_images(
            refetch=bool(getattr(args, "refetch", False)),
            upgrade_favicons=bool(getattr(args, "upgrade_favicons", False)),
        )
    )


async def _refetch_image_url(source_url: str) -> str | None:
    """Re-crawl a source URL and resolve its best image (slow path).

    Stored rows are re-fetched here directly. ``_fetch_url_content`` validates
    redirect hops but NOT its initial URL — ``add_url()`` normally runs the
    parse/scheme/SSRF gate before calling it. A legacy/HN/RSS row whose
    ``source_url`` is a private, link-local, loopback, or metadata host would
    otherwise be requested before any SSRF check runs. Re-apply the same gate
    here and fall back to the favicon (no network) for unsafe/unparseable URLs.
    """
    from argos.crawler._og_image import favicon_for_domain
    from argos.crawler.add_url import _fetch_url_content, _parse_and_validate
    from argos.crawler.dynamic_fetcher import _is_safe_url

    cleaned, reason = _parse_and_validate(source_url)
    if cleaned is None or not await _is_safe_url(cleaned):
        logging.getLogger(__name__).warning(
            "backfill --refetch skipping unsafe URL %s (%s); falling back to favicon",
            source_url,
            reason or "failed SSRF safety check",
        )
        return favicon_for_domain(source_url)

    try:
        data = await _fetch_url_content(cleaned)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "backfill --refetch failed for %s: %r (falling back to favicon)",
            source_url,
            exc,
        )
        data = None
    if data and data.get("image_url"):
        return data["image_url"]

    return favicon_for_domain(source_url)


def _is_favicon(url: str | None) -> bool:
    """True when ``url`` is a bare domain favicon (the lowest-priority cover).

    Thin re-export of :func:`argos.crawler._og_image.is_favicon_url` — the single
    source of truth shared with the cover templates so the backfill and the
    render-time branch agree on query-string favicons (``/favicon.ico?v=2``).
    """
    from argos.crawler._og_image import is_favicon_url

    return is_favicon_url(url)


async def _backfill_images(
    refetch: bool = False, upgrade_favicons: bool = False
) -> int:
    """Fill (or upgrade) ``tech_items.image_url``.

    Two selection modes:

    * Default (``upgrade_favicons=False``): target rows where ``image_url`` is
      NULL. ``refetch=False`` derives a domain favicon with no network call;
      ``refetch=True`` re-crawls ``source_url`` and applies the full chain.
      Only NULL rows are touched and the UPDATE is guarded by
      ``image_url IS NULL`` so a value set between SELECT and UPDATE is never
      clobbered.

    * Upgrade (``upgrade_favicons=True``, implies ``refetch``): target rows
      whose ``image_url`` path is ``/favicon.ico`` (bare, or with a cache-busting
      ``?query``/``#fragment``) — the earliest crawls and the no-network backfill
      persisted these, and because the default mode only fills NULLs they were
      otherwise stuck on the favicon forever. Each
      is re-crawled; the row is overwritten **only** when a real og/twitter/body
      image is recovered (the UPDATE is guarded by the old favicon value).
      Rows that still resolve to a favicon are left as-is.
    """
    from sqlalchemy import func, or_, select, update

    from argos.crawler._og_image import favicon_for_domain
    from argos.models.tech_item import TechItem

    if upgrade_favicons:
        refetch = True
        # A favicon cover is the path ".../favicon.ico" — but a stored URL may
        # carry a cache-busting query or fragment (``/favicon.ico?v=2``). Match
        # the path end followed by end-of-string, "?", or "#" so those rows are
        # selected too, keeping the SQL predicate in step with is_favicon_url()
        # and the cover templates. Otherwise a query-string favicon would stay
        # stuck forever (never NULL, never selected for upgrade).
        favicon_match = or_(
            TechItem.image_url.like("%/favicon.ico"),
            TechItem.image_url.like("%/favicon.ico?%"),
            TechItem.image_url.like("%/favicon.ico#%"),
        )
        selector = favicon_match
        guard = favicon_match
        noun = "favicon"
    else:
        selector = TechItem.image_url.is_(None)
        guard = TechItem.image_url.is_(None)
        noun = "NULL"

    # Re-crawls are network-bound and independent, so resolve each batch
    # concurrently (bounded) and commit per batch. Batched commits make
    # progress durable and visible — the previous single end-of-run commit
    # discarded everything if the (long) job was interrupted.
    concurrency = 1 if not refetch else 8
    batch_size = 50

    filled = 0
    async with AsyncSessionLocal() as session:
        # Newest first: the feed/portfolio surface the most recent items, so
        # filling those first makes the fix visible soonest during a long run.
        selected = (
            await session.execute(
                select(TechItem.id, TechItem.source_url, TechItem.image_url)
                .where(selector)
                .order_by(
                    func.coalesce(TechItem.published_at, TechItem.created_at).desc()
                )
            )
        ).all()
        if upgrade_favicons:
            # The LIKE selector is a coarse *superset*: a URL like
            # ".../render?source=/favicon.ico" ends with the literal string but
            # its path is "/render", so is_favicon_url() (and the cover
            # templates) treat it as a real image. Gate the rows through the same
            # path-only check so a genuine cover is never re-crawled and clobbered.
            selected = [r for r in selected if _is_favicon(r.image_url)]
        rows = [(r.id, r.source_url) for r in selected]

        sem = asyncio.Semaphore(concurrency)

        async def _resolve(source_url: str) -> str | None:
            if not refetch:
                return favicon_for_domain(source_url)
            async with sem:
                return await _refetch_image_url(source_url)

        total = len(rows)
        for start in range(0, total, batch_size):
            chunk = rows[start : start + batch_size]
            resolved = await asyncio.gather(*(_resolve(su) for _, su in chunk))
            for (tid, _su), new_url in zip(chunk, resolved):
                if not new_url:
                    continue
                # Upgrade mode only replaces a favicon with a *real* image; a
                # re-resolved favicon is not progress, so skip it.
                if upgrade_favicons and _is_favicon(new_url):
                    continue
                result = await session.execute(
                    update(TechItem)
                    .where(TechItem.id == tid, guard)
                    .values(image_url=new_url)
                )
                filled += result.rowcount or 0
            await session.commit()
            print(
                f"backfill-images: {min(start + batch_size, total)}/{total} "
                f"processed, {filled} filled",
                flush=True,
            )

    print(f"backfill-images: filled {filled} of {len(rows)} {noun} image_url rows")
    return EXIT_OK


def _build_backfill_digests_parser(
    sub: argparse._SubParsersAction,
    common: argparse.ArgumentParser,
) -> None:
    """Wire the ``argos backfill-digests`` subcommand (ARG-183)."""
    bf_p = sub.add_parser(
        "backfill-digests",
        help="Generate longform digest for items where digest IS NULL (LLM, slow)",
        parents=[common],
        description=(
            "Fill tech_items.digest where it is NULL and raw_content is present. "
            "Uses the digest LLM (qwen3:14b by default) — slow, one call per row. "
            "Existing non-NULL digests are never overwritten. Idempotent: re-runs "
            "only touch rows still NULL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bf_p.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Process at most N rows this run (default: all)",
    )
    bf_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many rows would be processed, without calling the LLM",
    )


def _cmd_backfill_digests(args: argparse.Namespace) -> int:
    return asyncio.run(
        _backfill_digests(
            limit=getattr(args, "limit", None),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    )


async def _backfill_digests(limit: int | None = None, dry_run: bool = False) -> int:
    """Fill NULL tech_items.digest rows via the digest LLM, one row at a time.

    Per-row failure isolation: a row whose generation raises is logged and
    skipped, never aborting the run. The UPDATE is guarded by ``digest IS NULL``
    so a value set between SELECT and UPDATE is never clobbered.
    """
    from sqlalchemy import select, update

    from argos.brain.llm_client import get_digest_llm_client
    from argos.brain.nodes.digest import generate_digest
    from argos.models.tech_item import TechItem

    async with AsyncSessionLocal() as session:
        stmt = select(
            TechItem.id, TechItem.raw_content, TechItem.summary
        ).where(
            TechItem.digest.is_(None),
            TechItem.raw_content.isnot(None),
        ).order_by(TechItem.created_at.asc())
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = (await session.execute(stmt)).all()

    print(f"backfill-digests: {len(rows)} candidate row(s).")
    if dry_run or not rows:
        return 0

    client = get_digest_llm_client()
    filled = 0
    skipped = 0
    try:
        for row in rows:
            try:
                digest = await generate_digest(
                    row.raw_content, summary=row.summary,
                    client=client, keep_alive="5m",
                )
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "backfill-digests: generation failed for %s: %r", row.id, exc
                )
                skipped += 1
                continue
            if digest is None:
                skipped += 1
                continue
            async with AsyncSessionLocal() as write_session:
                await write_session.execute(
                    update(TechItem)
                    .where(TechItem.id == row.id, TechItem.digest.is_(None))
                    .values(digest=digest)
                )
                await write_session.commit()
            filled += 1
            print(f"  ✓ {row.id} ({filled}/{len(rows)})")
    finally:
        try:
            await client.unload("large")
        except Exception:  # noqa: BLE001
            pass

    print(f"backfill-digests done: filled={filled}, skipped={skipped}.")
    return 0


def _build_backfill_trust_parser(
    sub: argparse._SubParsersAction,
    common: argparse.ArgumentParser,
) -> None:
    """Wire the ``argos backfill-trust`` subcommand (ARG-211)."""
    bf_p = sub.add_parser(
        "backfill-trust",
        help="Re-run the evidence rubric for items where trust_rubric IS NULL (LLM, slow)",
        parents=[common],
        description=(
            "Fill tech_items.trust_rubric where it is NULL and raw_content is "
            "present, by re-running the triage rubric prompt (small LLM), then "
            "re-synthesize trust_score from rubric + source prior + "
            "corroboration_count. Existing non-NULL trust_rubric rows are "
            "never overwritten. Idempotent: re-runs only touch rows still NULL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bf_p.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Process at most N rows this run (default: all)",
    )
    bf_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many rows would be processed, without calling the LLM",
    )


def _cmd_backfill_trust(args: argparse.Namespace) -> int:
    return asyncio.run(
        _backfill_trust(
            limit=getattr(args, "limit", None),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    )


async def _backfill_trust(limit: int | None = None, dry_run: bool = False) -> int:
    """Fill NULL tech_items.trust_rubric rows via the triage rubric LLM.

    Per-row failure isolation: a row whose rubric extraction fails (infra
    error or unparseable response) is logged and skipped, never aborting the
    run. The UPDATE is guarded by ``trust_rubric IS NULL`` so a value set
    between SELECT and UPDATE (e.g. by a live triage run) is never clobbered.
    """
    from sqlalchemy import select, update

    from argos.brain.llm_client import get_llm_client
    from argos.brain.nodes.triage import extract_rubric_via_llm
    from argos.brain.trust import (
        corroboration_score,
        score_rubric,
        source_prior,
        synthesize_trust,
    )
    from argos.models.tech_item import TechItem

    async with AsyncSessionLocal() as session:
        stmt = select(
            TechItem.id,
            TechItem.raw_content,
            TechItem.source_url,
            TechItem.corroboration_count,
        ).where(
            TechItem.trust_rubric.is_(None),
            TechItem.raw_content.isnot(None),
        ).order_by(TechItem.created_at.asc())
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = (await session.execute(stmt)).all()

    print(f"backfill-trust: {len(rows)} candidate row(s).")
    if dry_run or not rows:
        return 0

    client = get_llm_client()
    trust_cfg = settings.user.trust
    weights = {
        "rubric": trust_cfg.weight_rubric,
        "prior": trust_cfg.weight_prior,
        "corroboration": trust_cfg.weight_corroboration,
    }
    filled = 0
    skipped = 0
    try:
        for row in rows:
            try:
                rubric = await extract_rubric_via_llm(
                    row.raw_content, client, keep_alive="5m"
                )
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "backfill-trust: rubric extraction failed for %s: %r", row.id, exc
                )
                skipped += 1
                continue
            if rubric is None:
                skipped += 1
                continue
            rubric_score = score_rubric(rubric)
            prior_score = source_prior(row.source_url or "", trust_cfg.source_tiers)
            corr_score = corroboration_score(row.corroboration_count or 0)
            trust_score = synthesize_trust(rubric_score, prior_score, corr_score, weights)
            async with AsyncSessionLocal() as write_session:
                await write_session.execute(
                    update(TechItem)
                    .where(TechItem.id == row.id, TechItem.trust_rubric.is_(None))
                    .values(trust_rubric=rubric, trust_score=trust_score)
                )
                await write_session.commit()
            filled += 1
            print(f"  ✓ {row.id} ({filled}/{len(rows)})")
    finally:
        try:
            await client.unload("small")
        except Exception:
            pass

    print(f"backfill-trust done: filled={filled}, skipped={skipped}.")
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
    from argos import scheduler
    from argos.scheduler import SchedulerError, bootout_plist

    failures: list[str] = []
    for label in ("com.argos.run", "com.argos.brief", "com.argos.brief-weekly", "com.argos.web"):
        try:
            bootout_plist(label)
            print(f"Unloaded: {label}")
        except SchedulerError as exc:
            failures.append(f"{label}: {exc}")
            print(f"Failed to unload {label}: {exc}", file=sys.stderr)
    # com.argos.web is RunAtLoad + KeepAlive: a plist left in ~/Library/
    # LaunchAgents is auto-loaded at the next login and resurrects the daemon,
    # so bootout alone doesn't uninstall it. Delete the file too (mirrors the
    # opt-out path in reload_schedule). The scheduled jobs are calendar-driven
    # and follow the existing bootout-only convention.
    (scheduler._DEFAULT_LAUNCH_AGENTS / "com.argos.web.plist").unlink(missing_ok=True)
    return EXIT_GENERIC if failures else EXIT_OK


def _cmd_schedule_status(_args: argparse.Namespace) -> int:
    from argos.scheduler import is_loaded

    for label in ("com.argos.run", "com.argos.brief", "com.argos.brief-weekly", "com.argos.web"):
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
    _build_backfill_images_parser(sub, common)
    _build_backfill_digests_parser(sub, common)
    _build_backfill_trust_parser(sub, common)
    _build_backup_parser(sub)
    _build_restore_parser(sub)
    _build_config_parser(sub)
    _build_doctor_parser(sub)
    _build_init_parser(sub)
    _build_portfolio_parser(sub, common)
    _build_schedule_parser(sub)
    _build_search_parser(sub, common)
    _build_status_parser(sub)
    _build_stats_parser(sub, common)
    _build_web_parser(sub, common)

    args = parser.parse_args(argv)

    if args.command == "run":
        # ARG-114: _run sets up its own shared-Console logging (RichHandler) so
        # that log output does NOT corrupt the Rich Live progress bar. The global
        # basicConfig below is intentionally skipped for "run" — _run owns its
        # own handler configuration.
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        return asyncio.run(_run(args.url, verbose=getattr(args, "verbose", False)))

    # For all commands OTHER than "run", use the plain stream-based handler.
    # The "run" command skips this block entirely (it returns above) and
    # installs a RichHandler inside _run instead.
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "slack":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        from argos.main import main as slack_main

        asyncio.run(slack_main())
        return 0
    if args.command == "web":
        return _cmd_web(args)
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
    if args.command == "status":
        return _cmd_status(args)
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
    if args.command == "stats":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        if args.days <= 0:
            print(f"오류: --days 값은 양의 정수여야 합니다. (입력값: {args.days})", file=sys.stderr)
            return 1
        return asyncio.run(_stats(args.days))
    if args.command == "backfill-images":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        return _cmd_backfill_images(args)
    if args.command == "backfill-digests":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        return _cmd_backfill_digests(args)
    if args.command == "backfill-trust":
        rc = _apply_config_override(args)
        if rc is not None:
            return rc
        return _cmd_backfill_trust(args)
    if args.command == "backup":
        return _cmd_backup(args)
    if args.command == "restore":
        return _cmd_restore(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
