from __future__ import annotations

import stat

import pytest

from argos.init_wizard.env_file import (
    ENV_FILE_MODE,
    atomic_write_env,
    file_mode,
    load_env,
    merge_env,
)


def test_load_env_returns_empty_dict_for_missing_file(tmp_path):
    assert load_env(tmp_path / ".env") == {}


def test_load_env_parses_basic_key_value(tmp_path):
    f = tmp_path / ".env"
    f.write_text("FOO=bar\nBAZ=qux\n")
    assert load_env(f) == {"FOO": "bar", "BAZ": "qux"}


def test_load_env_strips_comments_and_blank_lines(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# a comment\n\nFOO=bar\n# inline=ignored\n")
    assert load_env(f) == {"FOO": "bar"}


def test_load_env_strips_surrounding_quotes(tmp_path):
    f = tmp_path / ".env"
    f.write_text('FOO="quoted value"\nBAR=\'single\'\n')
    assert load_env(f) == {"FOO": "quoted value", "BAR": "single"}


def test_load_env_skips_malformed_lines(tmp_path):
    f = tmp_path / ".env"
    f.write_text("FOO=bar\nno_equals_here\n=missing_key\nBAZ=qux\n")
    assert load_env(f) == {"FOO": "bar", "BAZ": "qux"}


def test_merge_env_updates_layer_on_top():
    existing = {"A": "1", "B": "2"}
    updates = {"B": "3", "C": "4"}
    assert merge_env(existing, updates) == {"A": "1", "B": "3", "C": "4"}
    # Originals untouched
    assert existing == {"A": "1", "B": "2"}


def test_atomic_write_env_round_trips(tmp_path):
    path = tmp_path / ".env"
    data = {"FOO": "bar", "BAZ": "qux"}
    atomic_write_env(path, data)
    assert load_env(path) == data


def test_atomic_write_env_quotes_values_with_special_chars(tmp_path):
    path = tmp_path / ".env"
    atomic_write_env(path, {"X": "value with spaces", "Y": "no#hash=ok"})
    raw = path.read_text()
    assert '"value with spaces"' in raw
    assert '"no#hash=ok"' in raw
    # round-trip
    assert load_env(path) == {"X": "value with spaces", "Y": "no#hash=ok"}


def test_atomic_write_env_chmod_600(tmp_path):
    path = tmp_path / ".env"
    atomic_write_env(path, {"FOO": "bar"})
    assert file_mode(path) == ENV_FILE_MODE
    # Sanity: file is not world-readable
    mode = path.stat().st_mode
    assert not (mode & stat.S_IROTH)
    assert not (mode & stat.S_IRGRP)


def test_atomic_write_env_tightens_perms_on_existing_loose_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text("FOO=stale\n")
    path.chmod(0o644)
    atomic_write_env(path, {"FOO": "fresh"})
    assert file_mode(path) == ENV_FILE_MODE


def test_atomic_write_env_leaves_original_intact_on_failure(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    atomic_write_env(path, {"FOO": "original"})

    import argos.init_wizard.env_file as mod

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_env(path, {"FOO": "overwrite"})
    # Original untouched
    assert load_env(path) == {"FOO": "original"}
    # Temp file cleaned up
    assert not (tmp_path / ".env.tmp").exists()
