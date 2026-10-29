"""Web Push plumbing for the morning briefing (ARG-321).

Standard Web Push (RFC 8030/8291/8292) via ``pywebpush``: the Mac encrypts each
message for one browser subscription and hands it to that browser's push
service (Apple / Google / Mozilla), which delivers it to the device. No server
of ours is involved beyond the Mac, and the payload carries only the
notification text (event count + one event title) and the URL to open.

The VAPID key pair lives in an owner-only PEM next to the web token
(``~/.config/argos/web_push_vapid.pem``), generated on first use. Rotating it
(deleting the file) invalidates every existing subscription — devices must
turn notifications on again.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from argos.private_files import ensure_private_dir, write_private_bytes

_log = logging.getLogger("argos.web.push")

KEY_FILE_ENV = "ARGOS_WEB_PUSH_KEY_FILE"
# VAPID ``sub`` claim fallback. Each subscription carries the https origin it
# was made from (see ``subject_origin``) and that is what gets sent; py_vapid
# only accepts ``mailto:`` or a bare ``https://host[:port]`` here.
FALLBACK_SUBJECT = "https://localhost"
_SUBJECT_RE = re.compile(r"^https://[\w.-]+(:\d{1,5})?$", re.IGNORECASE)
# How long a push service may hold an undelivered message (phone asleep or
# offline). Half a day: a morning briefing that lands after dinner is noise.
PUSH_TTL_SECONDS = 12 * 60 * 60
# Push services answer these for a subscription that will never work again.
EXPIRED_STATUSES = frozenset({404, 410})


def default_key_path() -> Path:
    """``$ARGOS_WEB_PUSH_KEY_FILE`` or ``${XDG_CONFIG_HOME:-~/.config}/argos/web_push_vapid.pem``."""
    override = os.environ.get(KEY_FILE_ENV)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "argos" / "web_push_vapid.pem"


def load_or_create_vapid(path: Optional[Path] = None):
    """Return the ``py_vapid.Vapid02`` key at ``path``, creating it (0600) if missing."""
    from py_vapid import Vapid02

    path = path or default_key_path()
    try:
        return Vapid02.from_pem(path.read_bytes())
    except FileNotFoundError:
        pass
    ensure_private_dir(path.parent)
    vapid = Vapid02()
    vapid.generate_keys()
    write_private_bytes(path, vapid.private_pem())
    _log.info("web push: generated a new VAPID key at %s", path)
    return vapid


def public_key_b64(vapid) -> str:
    """The ``applicationServerKey`` browsers subscribe with (uncompressed P-256, base64url)."""
    from cryptography.hazmat.primitives import serialization
    from py_vapid import b64urlencode

    raw = vapid.public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return b64urlencode(raw)


def device_label(user_agent: Optional[str]) -> str:
    """A short human label for a subscribing browser, e.g. ``iPhone`` / ``Mac · Chrome``."""
    ua = user_agent or ""
    if "iPhone" in ua:
        return "iPhone"
    if "iPad" in ua:
        return "iPad"
    if "Android" in ua:
        device = "Android"
    elif "Macintosh" in ua or "Mac OS X" in ua:
        device = "Mac"
    elif "Windows" in ua:
        device = "Windows"
    elif "Linux" in ua:
        device = "Linux"
    else:
        device = "기기"
    if "Edg/" in ua:
        browser = "Edge"
    elif "Firefox/" in ua:
        browser = "Firefox"
    elif "Chrome/" in ua:
        browser = "Chrome"
    elif "Safari/" in ua:
        browser = "Safari"
    else:
        return device
    return f"{device} · {browser}"


def subject_origin(scheme: str, host: Optional[str]) -> str:
    """``https://<host>`` for a secure subscribing page, else the fallback.

    ``host`` is what the browser reached (``X-Forwarded-Host`` behind
    ``tailscale serve``). Anything that doesn't look like a bare https origin
    falls back, so a bogus header can't produce a claim push services reject.
    """
    candidate = f"{scheme.lower()}://{(host or '').strip().lower()}"
    return candidate if _SUBJECT_RE.match(candidate) else FALLBACK_SUBJECT


@dataclass(frozen=True)
class PushTarget:
    endpoint: str
    p256dh: str
    auth: str
    subject: str = FALLBACK_SUBJECT


@dataclass(frozen=True)
class SendOutcome:
    """``status`` is ``"ok"``, ``"expired"`` (drop the subscription) or ``"failed"``."""

    status: str
    detail: str = ""


def _send_sync(target: PushTarget, payload: str, vapid) -> SendOutcome:
    from pywebpush import WebPushException, webpush

    try:
        webpush(
            subscription_info={
                "endpoint": target.endpoint,
                "keys": {"p256dh": target.p256dh, "auth": target.auth},
            },
            data=payload,
            vapid_private_key=vapid,
            # Fresh dict per call: pywebpush writes ``aud``/``exp`` into it.
            vapid_claims={"sub": target.subject},
            ttl=PUSH_TTL_SECONDS,
            timeout=15,
        )
    except WebPushException as exc:
        status = getattr(exc.response, "status_code", None)
        if status in EXPIRED_STATUSES:
            return SendOutcome("expired", f"HTTP {status}")
        return SendOutcome("failed", f"HTTP {status}" if status else str(exc)[:200])
    except Exception as exc:  # network errors, bad keys stored by a client, …
        return SendOutcome("failed", f"{type(exc).__name__}: {str(exc)[:200]}")
    return SendOutcome("ok")


async def send_push(target: PushTarget, message: dict, vapid) -> SendOutcome:
    """Encrypt ``message`` (JSON) for ``target`` and hand it to its push service."""
    payload = json.dumps(message, ensure_ascii=False)
    return await asyncio.to_thread(_send_sync, target, payload, vapid)
