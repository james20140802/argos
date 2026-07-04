"""Read/write service backing the 설정 screen (ARG-186).

A thin web layer on top of :mod:`argos.config_store` — the same dotted-key
read/write/mask primitives that back the ``argos config`` CLI. The web page
only ever exposes a small allowlist of *user-preference* fields for editing
(:data:`EDITABLE_FIELDS`); everything else (sources, secrets, advanced knobs)
is shown read-only with ``config_store``'s existing masking.

This module deliberately touches no database — it wraps ``config_store``,
which is pure ``tomllib``/pydantic — so importing it from ``argos.web.app`` at
module scope keeps the ``build_web_app`` import graph free of ``argos.database``
(guarded by ``test_build_web_app_does_not_import_argos_database``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import ValidationError

from argos import config_store

FieldKind = Literal["text", "list", "int", "bool"]


@dataclass(frozen=True)
class FieldSpec:
    """Static description of one editable config field."""

    key: str
    label: str
    kind: FieldKind


# Single source of truth for which config keys the web page may edit. Every key
# here is non-secret (``config_store.is_secret`` is False) and coercible by
# ``config_store.set_value`` (scalar / ``list[str]`` — not the nested-model
# lists ``rss.feeds`` / ``spa.sources``, which stay read-only in v1).
EDITABLE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("interests.topics", "관심 토픽 (쉼표 구분)", "list"),
    FieldSpec("interests.exclusions", "제외 토픽 (쉼표 구분)", "list"),
    FieldSpec("briefing.time", "일일 브리핑 시각 (HH:MM)", "text"),
    FieldSpec("briefing.weekdays", "브리핑 요일 (쉼표 구분)", "list"),
    FieldSpec("briefing.limit_per_category", "카테고리별 항목 수", "int"),
    FieldSpec("briefing.lookback_days", "조회 기간 (일)", "int"),
    FieldSpec("briefing.weekly_enabled", "주간 브리핑 사용", "bool"),
    FieldSpec("briefing.weekly_weekday", "주간 브리핑 요일", "text"),
    FieldSpec("run.time", "수집 실행 시각 (HH:MM)", "text"),
    FieldSpec("run.daily_limit", "일일 수집 한도", "int"),
    FieldSpec("slack.summary_language", "요약 언어", "text"),
)

_EDITABLE_KEYS: frozenset[str] = frozenset(f.key for f in EDITABLE_FIELDS)


@dataclass(frozen=True)
class SettingField:
    key: str
    label: str
    kind: FieldKind
    value: str
    error: Optional[str] = None


@dataclass(frozen=True)
class SettingsView:
    editable: list[SettingField]
    readonly: list[tuple[str, str]]
    saved: bool = False


def _format_value(value: object) -> str:
    """Render a live config value as the string form the form input expects."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def load_settings_view(
    path: Optional[Path] = None,
    *,
    submitted: Optional[dict[str, str]] = None,
    errors: Optional[dict[str, str]] = None,
    saved: bool = False,
) -> SettingsView:
    """Build the view model for GET /settings (and error re-renders of POST).

    ``submitted``/``errors`` are supplied only when re-rendering after a failed
    save: the field keeps the value the user typed (not the on-disk value) and
    shows the inline validation error.
    """
    resolved = path or config_store.default_config_path()
    submitted = submitted or {}
    errors = errors or {}

    editable: list[SettingField] = []
    for spec in EDITABLE_FIELDS:
        if spec.key in submitted:
            value = submitted[spec.key]
        else:
            value = _format_value(config_store.get_value(resolved, spec.key))
        editable.append(
            SettingField(
                key=spec.key,
                label=spec.label,
                kind=spec.kind,
                value=value,
                error=errors.get(spec.key),
            )
        )

    # Everything the page does not edit, shown read-only with config_store's
    # secret/token masking. Editable keys are dropped to avoid duplication.
    readonly = [
        (key, value)
        for key, value in config_store.list_entries(resolved)
        if key not in _EDITABLE_KEYS
    ]

    return SettingsView(editable=editable, readonly=readonly, saved=saved)


def apply_settings(
    updates: dict[str, str], path: Optional[Path] = None
) -> dict[str, str]:
    """Persist submitted edits via ``config_store.set_value``.

    Only keys in :data:`EDITABLE_FIELDS` are written — any other key (a secret
    or advanced field slipped into the form) is ignored, so the web page can
    never write outside its allowlist. Returns ``{key: message}`` for fields
    that failed validation (empty dict on full success).

    Note: ``set_value`` writes the file per key, so if a later key fails after
    an earlier one succeeded, the earlier change is already persisted (partial
    apply). Each ``set_value`` re-validates the whole model before writing, so
    the *failing* key itself leaves the file unchanged. Acceptable for v1.
    """
    resolved = path or config_store.default_config_path()
    errors: dict[str, str] = {}

    for spec in EDITABLE_FIELDS:
        if spec.key not in updates:
            continue
        raw = updates[spec.key]
        try:
            current = _format_value(config_store.get_value(resolved, spec.key))
            if raw == current:
                continue
            config_store.set_value(resolved, spec.key, raw)
        except config_store.SecretKeyError:
            # Should never happen for allowlist keys, but be defensive.
            errors[spec.key] = "시크릿 값은 웹에서 편집할 수 없습니다."
        except config_store.UnknownKeyError:
            errors[spec.key] = "알 수 없는 설정 키입니다."
        except (ValidationError, ValueError) as exc:
            errors[spec.key] = str(exc).strip().splitlines()[0]
        except OSError:
            errors[spec.key] = "설정 파일을 쓰지 못했습니다."

    return errors
