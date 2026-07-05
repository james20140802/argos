"""Unit tests for the settings service (ARG-186).

Exercises the thin config_store wrapper directly against a ``tmp_path``
config.toml — no database, no web layer. Mirrors the ``tmp_path`` round-trip
style of ``tests/test_config_cli.py``.
"""
from __future__ import annotations

import tomllib

from argos import config_store
from argos.web.services.settings import (
    EDITABLE_FIELDS,
    apply_settings,
    load_settings_view,
)

_ALLOWLIST = {f.key for f in EDITABLE_FIELDS}


def _load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_load_view_exposes_editable_fields_with_current_values(tmp_path):
    cfg = tmp_path / "config.toml"
    config_store.set_value(cfg, "briefing.time", "09:30")
    config_store.set_value(cfg, "interests.topics", "llm,agents")

    view = load_settings_view(cfg)

    fields = {f.key: f for f in view.editable}
    assert {f.key for f in view.editable} == _ALLOWLIST
    assert fields["briefing.time"].value == "09:30"
    assert fields["interests.topics"].value == "llm,agents"
    # bool renders as the string form the checkbox template expects.
    assert fields["briefing.weekly_enabled"].value in {"true", "false"}


def test_load_view_masks_token_value_in_readonly(tmp_path):
    cfg = tmp_path / "config.toml"
    # A token accidentally stored in a non-secret field must not echo in plaintext.
    config_store.set_value(cfg, "slack.channel_id", "xoxb-super-secret")

    view = load_settings_view(cfg)
    readonly = dict(view.readonly)

    assert readonly["slack.channel_id"] == "xoxb-***"
    # Editable keys are never duplicated into the read-only dump.
    assert not (_ALLOWLIST & set(readonly))


def test_apply_valid_updates_persists(tmp_path):
    cfg = tmp_path / "config.toml"

    errors = apply_settings(
        {"briefing.time": "08:15", "run.daily_limit": "42"}, cfg
    )

    assert errors == {}
    data = _load_toml(cfg)
    assert data["briefing"]["time"] == "08:15"
    assert data["run"]["daily_limit"] == 42


def test_apply_list_field_splits_csv(tmp_path):
    cfg = tmp_path / "config.toml"

    errors = apply_settings({"interests.topics": "llm, agents , rag"}, cfg)

    assert errors == {}
    data = _load_toml(cfg)
    assert data["interests"]["topics"] == ["llm", "agents", "rag"]


def test_apply_invalid_value_reports_error_and_leaves_file_unchanged(tmp_path):
    cfg = tmp_path / "config.toml"
    config_store.set_value(cfg, "briefing.limit_per_category", "10")
    before = cfg.read_bytes()

    errors = apply_settings({"briefing.limit_per_category": "0"}, cfg)  # ge=1

    assert "briefing.limit_per_category" in errors
    # set_value validates the whole model before writing → the failing key
    # leaves the file byte-for-byte unchanged.
    assert cfg.read_bytes() == before


def test_apply_ignores_non_allowlist_and_secret_keys(tmp_path):
    cfg = tmp_path / "config.toml"

    errors = apply_settings(
        {"slack.bot_token": "xoxb-evil", "ollama.host": "http://evil"}, cfg
    )

    # Neither key is in the allowlist → both silently ignored, nothing written.
    assert errors == {}
    if cfg.exists():
        data = _load_toml(cfg)
        assert "bot_token" not in data.get("slack", {})
        assert "ollama" not in data


def test_load_view_formats_sources_as_bare_urls(tmp_path):
    cfg = tmp_path / "config.toml"
    # Defaults ship several RSS feeds and one SPA source. The read-only display
    # must show only the URLs — not the raw pydantic repr (``url='…' …``).
    view = load_settings_view(cfg)
    readonly = dict(view.readonly)

    feeds = readonly["rss.feeds"]
    assert "url=" not in feeds
    assert "category=" not in feeds
    assert "https://openai.com/blog/rss.xml" in feeds
    # One URL per line so multiple feeds stay legible.
    assert feeds.count("\n") >= 1

    sources = readonly["spa.sources"]
    assert "listing_url=" not in sources
    assert "https://www.anthropic.com/news" in sources


def test_load_view_normalizes_unpadded_time_for_native_input(tmp_path):
    cfg = tmp_path / "config.toml"
    # `scheduler._parse_hhmm` accepts "6:00", but a native <input type="time">
    # only populates from a zero-padded HH:MM — normalize so it isn't blank.
    config_store.set_value(cfg, "briefing.time", "6:00")

    view = load_settings_view(cfg)
    fields = {f.key: f for f in view.editable}

    assert fields["briefing.time"].value == "06:00"


def test_apply_empty_time_is_rejected_not_persisted(tmp_path):
    cfg = tmp_path / "config.toml"
    config_store.set_value(cfg, "briefing.time", "07:00")

    # A cleared native time input posts "" — persisting it would break the next
    # `argos schedule install`, so it must be rejected.
    errors = apply_settings({"briefing.time": ""}, cfg)

    assert "briefing.time" in errors
    assert _load_toml(cfg)["briefing"]["time"] == "07:00"  # unchanged


def test_apply_invalid_time_is_rejected(tmp_path):
    cfg = tmp_path / "config.toml"
    config_store.set_value(cfg, "run.time", "06:00")

    errors = apply_settings({"run.time": "25:99"}, cfg)

    assert "run.time" in errors
    assert _load_toml(cfg)["run"]["time"] == "06:00"


def test_apply_canonicalizes_unpadded_time(tmp_path):
    cfg = tmp_path / "config.toml"

    errors = apply_settings({"briefing.time": "6:05"}, cfg)

    assert errors == {}
    assert _load_toml(cfg)["briefing"]["time"] == "06:05"


def test_apply_repairs_schema_invalid_on_disk_value_not_skipped(tmp_path):
    cfg = tmp_path / "config.toml"
    # A schema-invalid value on disk (ge=1 violated). UserConfig.load silently
    # falls back to *all* defaults, so get_value/the form would show 10. Saving
    # that 10 must actually write (repair the file) — the no-op shortcut must
    # compare against the raw on-disk "0", not the fallback 10, or the bad value
    # would survive and break the next unrelated edit.
    cfg.write_text("[briefing]\nlimit_per_category = 0\n", encoding="utf-8")

    errors = apply_settings({"briefing.limit_per_category": "10"}, cfg)

    assert errors == {}
    assert _load_toml(cfg)["briefing"]["limit_per_category"] == 10  # repaired


def test_apply_unchanged_value_is_a_noop(tmp_path):
    cfg = tmp_path / "config.toml"
    config_store.set_value(cfg, "briefing.time", "07:00")
    before = cfg.read_bytes()

    errors = apply_settings({"briefing.time": "07:00"}, cfg)

    assert errors == {}
    assert cfg.read_bytes() == before  # identical value → no rewrite
