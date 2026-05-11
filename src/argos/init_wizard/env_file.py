"""Atomic dotenv reader/writer for the init wizard.

The wizard only ever needs a tiny subset of dotenv semantics:

* Read key/value pairs from an existing ``.env`` (so we can re-display
  current values as prompt defaults).
* Merge user-supplied updates with the existing data so we never
  inadvertently drop unrelated keys.
* Write the merged result back **atomically** (``.env.tmp`` → ``os.replace``)
  with ``0600`` permissions so a partial write cannot corrupt secrets.

We deliberately do not handle exotic dotenv features (multiline values,
shell expansion, export prefixes) — the format the wizard writes is
``KEY=VALUE`` per line.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_FILE_MODE = 0o600


def load_env(path: Path) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` dotenv file. Missing file → empty dict.

    Blank lines and ``#`` comments are skipped. Surrounding double or
    single quotes on the value are stripped.
    """
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            logger.debug(".env line %d skipped (no '='): %r", lineno, raw)
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (len(value) >= 2) and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not key:
            continue
        result[key] = value
    return result


def merge_env(existing: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    """Return a new dict with ``updates`` layered on top of ``existing``.

    Values from ``updates`` whose value is ``None`` are skipped (allowing
    callers to express "don't touch this key"). Empty strings *do* overwrite —
    callers must filter them out beforehand if that's not desired.
    """
    merged: dict[str, str] = dict(existing)
    for key, value in updates.items():
        if value is None:  # type: ignore[unreachable]
            continue
        merged[key] = value
    return merged


def _serialise(data: dict[str, str]) -> str:
    """Render the merged dict as ``KEY=VALUE`` text.

    Values that contain whitespace, ``#`` or ``=`` are wrapped in double quotes
    so a future ``load_env`` round-trip is lossless. We do not escape internal
    quotes — secrets like ``xoxb-…`` and DB passwords don't contain them in
    practice and the user can always edit the file by hand.
    """
    out_lines: list[str] = []
    for key, value in data.items():
        needs_quote = any(ch in value for ch in (" ", "\t", "#", "="))
        rendered = f'"{value}"' if needs_quote else value
        out_lines.append(f"{key}={rendered}")
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def atomic_write_env(path: Path, data: dict[str, str]) -> None:
    """Serialise ``data`` to ``path`` atomically with ``0600`` permissions.

    Mirrors :func:`argos.config_store.atomic_write` — writes to ``path.tmp``
    first, then ``os.replace`` so a crash mid-write leaves the original
    file intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = _serialise(data)
    try:
        # Use os.open with explicit mode so the temp file is never world-readable.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, ENV_FILE_MODE)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
        except Exception:
            # fdopen owns the fd on success; on failure we close it ourselves.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.replace(tmp, path)
        # os.replace preserves source perms (0600); ensure the destination is locked down
        # even if the file already existed with looser permissions.
        os.chmod(path, ENV_FILE_MODE)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def file_mode(path: Path) -> int:
    """Return the lower 9 permission bits of ``path`` (for tests / healthcheck)."""
    return stat.S_IMODE(path.stat().st_mode)


__all__ = [
    "ENV_FILE_MODE",
    "atomic_write_env",
    "file_mode",
    "load_env",
    "merge_env",
]
