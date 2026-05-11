"""Read/write helpers behind the ``argos config`` CLI.

This module owns dotted-path navigation against the pydantic
:class:`argos.config.UserConfig` model. Reads use stdlib ``tomllib``; writes
use ``tomli-w`` with an atomic ``write→os.replace`` pattern so a partial write
cannot corrupt the user's ``config.toml``.

Secret values (POSTGRES_PASSWORD, SLACK_*_TOKEN, anything matching ``*token*``
``*password*`` ``*secret*`` in the dotted path) are *never* written by this
module — they must live in environment variables or be edited into the file
directly. The defense-in-depth check is "explicit allowlist *and* pattern
match" so future additions to :class:`UserConfig` stay safe by default.
"""

from __future__ import annotations

import os
import typing
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import tomli_w

try:
    import tomllib
except ImportError:  # pragma: no cover - Python <3.11
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel

from argos.config import UserConfig

# Dotted keys that must never be settable via the CLI even if they ever land in
# UserConfig. The pattern check below covers anything matching
# ``*token*`` / ``*password*`` / ``*secret*`` — this set lists explicit
# canonical names so the rejection message is deterministic.
_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "postgres_password",
        "slack_bot_token",
        "slack_app_token",
        "slack.bot_token",
        "slack.app_token",
    }
)

# Token-prefix substrings used to mask values in `argos config list`. Even if a
# user puts a token in a non-secret field like ``slack.channel_id``, list output
# should not echo it back in plaintext.
_TOKEN_PREFIXES: tuple[str, ...] = ("xoxb-", "xapp-", "xoxa-", "xoxp-", "xoxs-")

DEFAULT_LIST_DELIMITER = ","


def default_config_path() -> Path:
    """Return the canonical user config path (``~/.config/argos/config.toml``)."""
    return Path.home() / ".config" / "argos" / "config.toml"


def is_secret(dotted_key: str) -> bool:
    """Return True if ``dotted_key`` names a secret that must not be CLI-settable."""
    lower = dotted_key.lower()
    if lower in _SECRET_KEYS:
        return True
    for needle in ("token", "password", "secret"):
        if needle in lower:
            return True
    return False


def _mask_token_value(value: Any) -> Any:
    """Mask string values that look like API tokens. List/scalar aware."""
    if isinstance(value, str):
        for prefix in _TOKEN_PREFIXES:
            if value.startswith(prefix):
                return f"{prefix}***"
        return value
    if isinstance(value, list):
        return [_mask_token_value(v) for v in value]
    return value


def load_raw(path: Path) -> dict[str, Any]:
    """Load the on-disk TOML as a plain dict. Missing file → empty dict."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Serialize ``data`` to TOML and write atomically via ``os.replace``.

    Parent directories are created if missing. If the rename fails, the
    on-disk file is untouched and the temp file is cleaned up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            tomli_w.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup; don't shadow original error.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Dotted-path navigation
# ---------------------------------------------------------------------------


class UnknownKeyError(KeyError):
    """Raised when a dotted key does not exist on the pydantic schema."""


class SecretKeyError(ValueError):
    """Raised when a dotted key resolves to (or pattern-matches) a secret."""


def _resolve_field(dotted_key: str) -> tuple[list[str], Any]:
    """Walk ``UserConfig`` along ``dotted_key`` and return (path_parts, annotation).

    Raises :class:`UnknownKeyError` if any segment doesn't exist.
    """
    parts = dotted_key.split(".")
    if not all(parts):
        raise UnknownKeyError(dotted_key)

    current: type[BaseModel] = UserConfig
    annotation: Any = UserConfig
    for i, part in enumerate(parts):
        if not isinstance(current, type) or not issubclass(current, BaseModel):
            # We've descended past a BaseModel into a primitive — there's no
            # further attribute to resolve.
            raise UnknownKeyError(dotted_key)
        fields = current.model_fields
        if part not in fields:
            raise UnknownKeyError(dotted_key)
        annotation = fields[part].annotation
        # Descend into nested BaseModel if there are more segments to consume.
        if i < len(parts) - 1:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                current = annotation
            else:
                raise UnknownKeyError(dotted_key)
    return parts, annotation


def _coerce(raw: str, annotation: Any) -> Any:
    """Coerce a CLI string ``raw`` into the field's declared annotation.

    Supports: ``str``, ``int``, ``float``, ``bool``, ``list[str]``,
    ``Literal[...]``. Pydantic ultimately re-validates the whole model, so we
    only need to produce *plausible* native values here.
    """
    origin = get_origin(annotation)
    if annotation is str:
        return raw
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    if annotation is bool:
        lower = raw.strip().lower()
        if lower in {"true", "1", "yes", "on"}:
            return True
        if lower in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"cannot coerce {raw!r} to bool")
    if origin is list or annotation is list:
        # Strip whitespace around each element; empty string → empty list.
        if not raw.strip():
            return []
        return [s.strip() for s in raw.split(DEFAULT_LIST_DELIMITER) if s.strip()]
    if origin is Literal or origin is typing.Literal:
        choices = get_args(annotation)
        if raw not in choices:
            raise ValueError(f"value {raw!r} is not one of {list(choices)!r}")
        return raw
    # Fallback: hand the raw string to pydantic and let it coerce.
    return raw


def _walk_dict(data: dict[str, Any], parts: list[str], create: bool = False) -> Any:
    """Walk ``data`` along ``parts``. With ``create``, return parent dict for last segment."""
    cursor: Any = data
    for p in parts[:-1]:
        if not isinstance(cursor, dict) or p not in cursor:
            if create:
                cursor[p] = {}
            else:
                raise KeyError(p)
        cursor = cursor[p]
        if not isinstance(cursor, dict):
            raise KeyError(p)
    return cursor


def get_value(path: Path, dotted_key: str) -> Any:
    """Resolve ``dotted_key`` against the merged (file + defaults) UserConfig."""
    if is_secret(dotted_key):
        raise SecretKeyError(dotted_key)
    parts, _annotation = _resolve_field(dotted_key)
    # Use the fully-validated UserConfig so defaults apply when the file is partial.
    cfg = UserConfig.load(path=path)
    cursor: Any = cfg
    for p in parts:
        cursor = getattr(cursor, p)
    return cursor


def set_value(path: Path, dotted_key: str, raw_value: str) -> UserConfig:
    """Validate + persist ``raw_value`` at ``dotted_key``. Returns the new config."""
    # 1. Secret rejection FIRST (covers keys that don't exist on UserConfig but
    #    look like tokens, e.g. ``slack.bot_token``).
    if is_secret(dotted_key):
        raise SecretKeyError(dotted_key)

    # 2. Key existence check against the pydantic schema.
    parts, annotation = _resolve_field(dotted_key)

    # 3. Coerce the raw string into the field's annotation.
    coerced = _coerce(raw_value, annotation)

    # 4. Merge into the existing TOML data and re-validate the whole model.
    data = load_raw(path)
    parent = _walk_dict(data, parts, create=True)
    parent[parts[-1]] = coerced

    new_cfg = UserConfig.model_validate(data)  # raises ValidationError → CLI exit 3

    # 5. Atomic write only after validation succeeds.
    atomic_write(path, data)
    return new_cfg


def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(obj, BaseModel):
        out: list[tuple[str, Any]] = []
        for name in type(obj).model_fields:
            value = getattr(obj, name)
            new_prefix = f"{prefix}.{name}" if prefix else name
            out.extend(_flatten(value, new_prefix))
        return out
    return [(prefix, obj)]


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def list_entries(path: Path) -> list[tuple[str, str]]:
    """Return ``(dotted_key, masked_string_value)`` rows for ``argos config list``."""
    cfg = UserConfig.load(path=path)
    rows: list[tuple[str, str]] = []
    for key, value in _flatten(cfg):
        if is_secret(key):
            rows.append((key, "***"))
            continue
        masked = _mask_token_value(value)
        rows.append((key, _format_value(masked)))
    return rows


__all__ = [
    "DEFAULT_LIST_DELIMITER",
    "SecretKeyError",
    "UnknownKeyError",
    "atomic_write",
    "default_config_path",
    "get_value",
    "is_secret",
    "list_entries",
    "load_raw",
    "set_value",
]


# ---------------------------------------------------------------------------
# Sanity check: any new UserConfig field whose name matches a secret pattern
# must also appear in _SECRET_KEYS. This is run at import time so a misconfigured
# schema fails fast in tests (and prod) rather than silently leaking values.
# ---------------------------------------------------------------------------


def _audit_schema() -> None:
    leaked: list[str] = []
    for top_name, top_field in UserConfig.model_fields.items():
        annotation = top_field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            for sub in annotation.model_fields:
                dotted = f"{top_name}.{sub}"
                if is_secret(dotted) and dotted.lower() not in _SECRET_KEYS:
                    leaked.append(dotted)
        else:
            if is_secret(top_name) and top_name.lower() not in _SECRET_KEYS:
                leaked.append(top_name)
    if leaked:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Secret-like UserConfig fields not in _SECRET_KEYS: {leaked!r}"
        )


_audit_schema()
