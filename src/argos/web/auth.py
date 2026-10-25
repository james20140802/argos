"""Local web token + session-cookie signing for the Argos web app (ARG-195).

The web app is reachable over the tailnet, so "can reach the port" must not be
the same as "can change my portfolio". A random token lives in an owner-only
file next to the rest of the Argos config; logging in once with it plants a
long-lived HttpOnly cookie whose value is an HMAC of the token. Rotating the
token therefore invalidates every existing cookie at once.

Only stdlib (``secrets`` / ``hmac`` / ``hashlib``) — no new dependency.
Never log the token or the cookie value.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path

from argos.private_files import (
    PRIVATE_FILE_MODE,
    ensure_private_dir,
    is_group_or_world_accessible,
    write_private_bytes,
)

TOKEN_FILE_ENV = "ARGOS_WEB_TOKEN_FILE"
SESSION_COOKIE = "argos_session"
SESSION_MAX_AGE = 400 * 24 * 60 * 60  # Chrome's cookie lifetime cap
_SESSION_CONTEXT = b"argos-web-session-v1"
_TOKEN_BYTES = 32


def default_token_path() -> Path:
    """``$ARGOS_WEB_TOKEN_FILE`` or ``${XDG_CONFIG_HOME:-~/.config}/argos/web_token``."""
    override = os.environ.get(TOKEN_FILE_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "argos" / "web_token"


def generate_token() -> str:
    return secrets.token_urlsafe(_TOKEN_BYTES)


def read_token(path: Path) -> str | None:
    """Return the stored token, or ``None`` when the file is missing/empty."""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return value or None


def write_token(path: Path, token: str) -> None:
    write_private_bytes(path, (token + "\n").encode("utf-8"))


def ensure_token(path: Path) -> tuple[str, bool, bool]:
    """Make sure a token exists at ``path`` with owner-only permissions.

    Returns ``(token, created, tightened)`` — ``created`` when a new token was
    generated, ``tightened`` when an existing file/dir had looser permissions
    and was corrected to ``0600`` / ``0700``.
    """
    tightened = False
    parent = path.parent
    if parent.is_dir() and is_group_or_world_accessible(parent):
        tightened = True
    ensure_private_dir(parent, tighten_existing=True)
    token = read_token(path)
    if token is None:
        token = generate_token()
        write_token(path, token)
        return token, True, tightened
    if is_group_or_world_accessible(path):
        os.chmod(path, PRIVATE_FILE_MODE)
        tightened = True
    return token, False, tightened


def rotate_token(path: Path) -> str:
    """Replace the token with a fresh one (invalidates every session cookie)."""
    ensure_private_dir(path.parent, tighten_existing=True)
    token = generate_token()
    write_token(path, token)
    return token


def session_value(token: str) -> str:
    """Cookie value bound to ``token``: HMAC-SHA256(token, context)."""
    return hmac.new(token.encode("utf-8"), _SESSION_CONTEXT, hashlib.sha256).hexdigest()


def verify_session(cookie: str | None, token: str | None) -> bool:
    if not cookie or not token:
        return False
    return hmac.compare_digest(cookie.encode("utf-8"), session_value(token).encode("utf-8"))


def verify_token(candidate: str | None, token: str | None) -> bool:
    if not candidate or not token:
        return False
    return hmac.compare_digest(candidate.strip().encode("utf-8"), token.encode("utf-8"))
