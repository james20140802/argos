"""Owner-only file/dir helpers for secret-bearing Argos files (ARG-195).

Every file that can hold a secret (``.env``, ``config.toml``, ``web_token``)
is written ``0600`` regardless of the process umask, and directories Argos
creates for them are ``0700``.

Existing directories are left alone unless the caller opts in with
``tighten_existing=True``: ``ARGOS_ENV_FILE`` may point into a shared
directory (e.g. a repo root) that must not be silently chmodded.
"""

from __future__ import annotations

import os
from pathlib import Path

PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700


def ensure_private_dir(path: Path, *, tighten_existing: bool = False) -> None:
    """Create ``path`` (and parents) with the leaf at ``0700``.

    Pre-existing directories keep their mode unless ``tighten_existing``.
    """
    existed = path.is_dir()
    path.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    if not existed or tighten_existing:
        # mkdir(mode=) is masked by umask; chmod sets it exactly.
        os.chmod(path, PRIVATE_DIR_MODE)


def write_private_bytes(path: Path, payload: bytes) -> None:
    """Atomically write ``payload`` to ``path`` with ``0600`` permissions.

    Writes ``path.tmp`` (created ``0600`` via ``os.open``) then ``os.replace``;
    re-applies ``0600`` afterwards so a looser pre-existing file is tightened.
    """
    ensure_private_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.chmod(tmp, PRIVATE_FILE_MODE)  # stale tmp could predate us
        os.replace(tmp, path)
        os.chmod(path, PRIVATE_FILE_MODE)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def file_mode(path: Path) -> int:
    """Return the permission bits of ``path`` (``stat.S_IMODE``)."""
    return path.stat().st_mode & 0o777


def is_group_or_world_accessible(path: Path) -> bool:
    return bool(file_mode(path) & 0o077)
