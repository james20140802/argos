"""ARG-195: secret-bearing files are 0600 and argos-owned dirs are 0700."""

from __future__ import annotations

import os
import stat

import pytest

from argos import config_store
from argos.init_wizard.env_file import atomic_write_env
from argos.private_files import ensure_private_dir, write_private_bytes


def _mode(p) -> int:
    return stat.S_IMODE(os.stat(p).st_mode)


@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


def test_config_toml_written_0600_under_umask_022(tmp_path, umask_022):
    path = tmp_path / "argos" / "config.toml"
    config_store.atomic_write(path, {"slack": {"channel_id": "C1"}})
    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700


def test_config_toml_overwrite_tightens_existing_644(tmp_path, umask_022):
    path = tmp_path / "config.toml"
    path.write_text("x = 1\n")
    os.chmod(path, 0o644)
    config_store.atomic_write(path, {"x": 2})
    assert _mode(path) == 0o600
    assert "x = 2" in path.read_text()


def test_new_private_dir_is_0700(tmp_path, umask_022):
    d = tmp_path / "new" / "argos"
    ensure_private_dir(d)
    assert _mode(d) == 0o700


def test_existing_foreign_dir_is_not_chmodded(tmp_path, umask_022):
    """An existing directory (e.g. repo root via ARGOS_ENV_FILE) keeps its mode."""
    d = tmp_path / "shared"
    d.mkdir()
    os.chmod(d, 0o755)
    ensure_private_dir(d)
    assert _mode(d) == 0o755


def test_existing_dir_tightened_when_requested(tmp_path, umask_022):
    d = tmp_path / "argos"
    d.mkdir()
    os.chmod(d, 0o755)
    ensure_private_dir(d, tighten_existing=True)
    assert _mode(d) == 0o700


def test_write_private_bytes_0600_and_atomic(tmp_path, umask_022):
    p = tmp_path / "argos" / "web_token"
    write_private_bytes(p, b"abc\n")
    assert p.read_bytes() == b"abc\n"
    assert _mode(p) == 0o600
    assert _mode(p.parent) == 0o700
    assert not list((tmp_path / "argos").glob("*.tmp"))


def test_write_private_bytes_tightens_existing_file(tmp_path, umask_022):
    p = tmp_path / "web_token"
    p.write_bytes(b"old")
    os.chmod(p, 0o644)
    write_private_bytes(p, b"new")
    assert _mode(p) == 0o600
    assert p.read_bytes() == b"new"


def test_env_file_still_0600_and_new_dir_0700(tmp_path, umask_022):
    p = tmp_path / "cfg" / ".env"
    atomic_write_env(p, {"A": "1"})
    assert _mode(p) == 0o600
    assert _mode(p.parent) == 0o700
