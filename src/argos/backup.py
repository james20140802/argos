"""Postgres backup/restore helpers behind the ``argos backup`` / ``argos restore`` CLI (ARG-192).

Months of LLM triage/embed/genealogist output live in the bind-mounted
``./pgdata`` volume with no backup path — a disk failure, a stray
``docker compose down -v``, or a bad migration would be permanent data loss.
This module gives operators a single-command dump/restore path.

Design notes:

* Shells out to ``docker exec <container> pg_dump`` / ``pg_restore`` rather
  than requiring a local Postgres client install — most operators only have
  the ``pgvector/pgvector:pg16`` image, not host-side ``pg_dump``.
* Targets the container **by name** (``argos-db``, from
  ``docker-compose.yml``'s ``container_name``) instead of
  ``docker compose exec``. The compose *project* name defaults to the
  containing directory's basename, which differs across clones/worktrees
  (e.g. this very worktree checkout) — the container name is stable
  regardless of where the repo lives on disk.
* DB credentials come from :data:`argos.config.settings` (the same
  ``POSTGRES_*`` values used to build ``database_url``), so backup/restore
  never need their own copy of the connection info.
* Dumps use ``pg_dump -Fc`` (custom format): compressed, and restorable with
  ``pg_restore`` against a database that doesn't yet match the dump's schema
  ordering. Restore uses ``docker cp`` to stage the dump file inside the
  container, then ``pg_restore --clean --if-exists`` — piping a custom-format
  archive through ``pg_restore``'s stdin is unreliable for anything beyond
  the most trivial dumps, so we stage a real file instead.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
import subprocess
from pathlib import Path

from argos.config import settings

logger = logging.getLogger(__name__)

# Matches docker-compose.yml `services.db.container_name`.
DEFAULT_CONTAINER_NAME = "argos-db"

_DUMP_GLOB = "argos-*.dump"
_DUMP_SUFFIX = ".dump"


class BackupError(RuntimeError):
    """Raised when a backup or restore operation cannot complete."""


def default_backup_dir() -> Path:
    """Return the XDG data directory for dumps (``~/.local/share/argos/backups``).

    Honors ``XDG_DATA_HOME`` like the rest of Argos's config/env resolution
    honors ``XDG_CONFIG_HOME`` (see ``argos.config_store.default_config_path``).
    """
    xdg = os.environ.get("XDG_DATA_HOME")
    xdg_base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return xdg_base / "argos" / "backups"


def _timestamped_filename(prefix: str = "argos") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}{_DUMP_SUFFIX}"


def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    logger.debug("subprocess: %s", " ".join(cmd))
    return subprocess.run(cmd, **kwargs)  # type: ignore[call-overload]


def docker_available() -> bool:
    """True when a ``docker`` binary is on PATH."""
    return shutil.which("docker") is not None


def container_running(container: str = DEFAULT_CONTAINER_NAME) -> bool:
    """True when ``container`` exists and is currently running."""
    if not docker_available():
        return False
    proc = _run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _require_container(container: str) -> None:
    if not docker_available():
        raise BackupError("docker binary not found on PATH — install Docker Desktop or Colima")
    if not container_running(container):
        raise BackupError(
            f"container '{container}' is not running — start it with `docker compose up -d` "
            "(run `docker ps` to confirm the name)"
        )


def create_backup(
    *,
    container: str = DEFAULT_CONTAINER_NAME,
    output_dir: Path | None = None,
    keep: int | None = None,
) -> Path:
    """Dump the Argos Postgres DB to a timestamped custom-format file.

    Runs ``docker exec <container> pg_dump -Fc -U <user> -d <db>`` and streams
    stdout straight to disk. Writes to a ``.part`` temp file first and
    atomically renames on success so a crashed/killed dump never leaves a
    corrupt file at the final path.

    When ``keep`` is set, prunes older dumps in ``output_dir`` (matching the
    ``argos-*.dump`` naming convention this function writes) down to the most
    recent ``keep`` files, newest last written of course also being kept.

    Raises :class:`BackupError` on any failure. Returns the path to the
    created dump.
    """
    _require_container(container)

    out_dir = output_dir or default_backup_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / _timestamped_filename()
    tmp_dest = dest.with_name(dest.name + ".part")

    secrets = settings.secrets
    cmd = [
        "docker",
        "exec",
        "-e",
        f"PGPASSWORD={secrets.POSTGRES_PASSWORD}",
        container,
        "pg_dump",
        "-U",
        secrets.POSTGRES_USER,
        "-d",
        secrets.POSTGRES_DB,
        "-Fc",
    ]

    try:
        with open(tmp_dest, "wb") as f:
            proc = _run(cmd, stdout=f, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        tmp_dest.unlink(missing_ok=True)
        raise BackupError(f"failed to invoke docker exec: {exc}") from exc

    if proc.returncode != 0:
        tmp_dest.unlink(missing_ok=True)
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise BackupError(f"pg_dump failed (exit {proc.returncode}): {stderr or '(no stderr)'}")

    if tmp_dest.stat().st_size == 0:
        tmp_dest.unlink(missing_ok=True)
        raise BackupError("pg_dump produced an empty file — aborting")

    tmp_dest.replace(dest)
    logger.info("backup written to %s", dest)

    if keep is not None and keep > 0:
        prune_old_backups(out_dir, keep)

    return dest


def prune_old_backups(out_dir: Path, keep: int) -> list[Path]:
    """Delete all but the ``keep`` most recently modified dumps in ``out_dir``.

    Returns the list of deleted paths (empty if nothing needed pruning).
    """
    dumps = sorted(out_dir.glob(_DUMP_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    removed: list[Path] = []
    for old in dumps[keep:]:
        old.unlink(missing_ok=True)
        removed.append(old)
    return removed


def restore_backup(
    dump_path: Path,
    *,
    container: str = DEFAULT_CONTAINER_NAME,
    clean: bool = True,
) -> None:
    """Restore ``dump_path`` into the Argos Postgres DB.

    **Destructive**: with ``clean=True`` (the default) this drops and
    recreates objects that exist in the dump before restoring them
    (``pg_restore --clean --if-exists``), overwriting current data in the
    target database. Callers (the CLI) are expected to confirm with the
    operator before calling this.

    Stages the dump inside the container via ``docker cp`` (piping a
    custom-format archive through ``pg_restore``'s stdin is unreliable), runs
    ``pg_restore``, then removes the staged copy — including on failure.
    """
    if not dump_path.exists():
        raise BackupError(f"dump file not found: {dump_path}")
    _require_container(container)

    remote_path = f"/tmp/{dump_path.name}"
    secrets = settings.secrets

    cp_in = _run(
        ["docker", "cp", str(dump_path), f"{container}:{remote_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if cp_in.returncode != 0:
        raise BackupError(f"docker cp (stage) failed: {cp_in.stderr.strip() or cp_in.stdout.strip()}")

    try:
        restore_cmd = [
            "docker",
            "exec",
            "-e",
            f"PGPASSWORD={secrets.POSTGRES_PASSWORD}",
            container,
            "pg_restore",
            "-U",
            secrets.POSTGRES_USER,
            "-d",
            secrets.POSTGRES_DB,
            "--no-owner",
        ]
        if clean:
            restore_cmd += ["--clean", "--if-exists"]
        restore_cmd.append(remote_path)

        proc = _run(restore_cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            stderr = (proc.stderr or proc.stdout or "").strip()
            raise BackupError(f"pg_restore failed (exit {proc.returncode}): {stderr or '(no output)'}")
        logger.info("restore complete from %s", dump_path)
    finally:
        _run(
            ["docker", "exec", container, "rm", "-f", remote_path],
            capture_output=True,
            text=True,
            check=False,
        )


def list_backups(output_dir: Path | None = None) -> list[Path]:
    """Return dumps in ``output_dir`` (or the default backup dir), newest first."""
    out_dir = output_dir or default_backup_dir()
    if not out_dir.exists():
        return []
    return sorted(out_dir.glob(_DUMP_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
