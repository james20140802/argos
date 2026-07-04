"""Tests for `argos.backup` (ARG-192).

All docker/pg_dump/pg_restore interactions go through `argos.backup._run`
(a thin `subprocess.run` wrapper) which is monkeypatched here — no real
docker/subprocess/DB calls happen in this file.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from argos import backup


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _docker_on_path(monkeypatch):
    """Default: `docker` binary resolves, so tests opt into "missing docker" explicitly."""
    monkeypatch.setattr(backup.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)


# ---------------------------------------------------------------------------
# default_backup_dir
# ---------------------------------------------------------------------------


def test_default_backup_dir_uses_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert backup.default_backup_dir() == tmp_path / "argos" / "backups"


def test_default_backup_dir_falls_back_to_local_share(monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert backup.default_backup_dir() == Path.home() / ".local" / "share" / "argos" / "backups"


# ---------------------------------------------------------------------------
# container_running / docker_available
# ---------------------------------------------------------------------------


def test_docker_available_false_when_binary_missing(monkeypatch):
    monkeypatch.setattr(backup.shutil, "which", lambda name: None)
    assert backup.docker_available() is False


def test_container_running_true_on_running_container(monkeypatch):
    monkeypatch.setattr(backup, "_run", lambda cmd, **kw: _completed(0, stdout="true\n"))
    assert backup.container_running("argos-db") is True


def test_container_running_false_on_stopped_container(monkeypatch):
    monkeypatch.setattr(backup, "_run", lambda cmd, **kw: _completed(0, stdout="false\n"))
    assert backup.container_running("argos-db") is False


def test_container_running_false_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(backup, "_run", lambda cmd, **kw: _completed(1, stderr="no such container"))
    assert backup.container_running("argos-db") is False


def test_container_running_false_when_docker_missing(monkeypatch):
    monkeypatch.setattr(backup.shutil, "which", lambda name: None)
    assert backup.container_running("argos-db") is False


# ---------------------------------------------------------------------------
# create_backup
# ---------------------------------------------------------------------------


def test_create_backup_requires_docker(monkeypatch, tmp_path):
    monkeypatch.setattr(backup.shutil, "which", lambda name: None)
    with pytest.raises(backup.BackupError, match="docker binary"):
        backup.create_backup(output_dir=tmp_path)


def test_create_backup_requires_running_container(monkeypatch, tmp_path):
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: False)
    with pytest.raises(backup.BackupError, match="is not running"):
        backup.create_backup(output_dir=tmp_path)


def test_create_backup_writes_dump_and_returns_path(monkeypatch, tmp_path):
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    def fake_run(cmd, **kwargs):
        # Simulate pg_dump writing bytes to the stdout file handle argument.
        stdout_fh = kwargs["stdout"]
        stdout_fh.write(b"fake-dump-bytes")
        return _completed(0)

    monkeypatch.setattr(backup, "_run", fake_run)

    dest = backup.create_backup(output_dir=tmp_path)

    assert dest.parent == tmp_path
    assert dest.name.startswith("argos-") and dest.name.endswith(".dump")
    assert dest.read_bytes() == b"fake-dump-bytes"
    # No leftover .part temp file.
    assert list(tmp_path.glob("*.part")) == []


def test_create_backup_wraps_unwritable_output_dir_in_backup_error(monkeypatch, tmp_path):
    """An --output-dir that cannot be created must surface as BackupError,
    not a raw OSError traceback (the CLI only catches BackupError)."""
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)
    monkeypatch.setattr(backup, "_run", lambda cmd, **kw: _completed(0))

    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file, not a directory")

    with pytest.raises(backup.BackupError, match="cannot create backup directory"):
        backup.create_backup(output_dir=blocker / "backups")


def test_create_backup_rapid_double_run_yields_distinct_dumps(monkeypatch, tmp_path):
    """Two back-to-back backups must never derive the same dump path
    (filenames carry microseconds)."""
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    def fake_run(cmd, **kwargs):
        kwargs["stdout"].write(b"bytes")
        return _completed(0)

    monkeypatch.setattr(backup, "_run", fake_run)

    first = backup.create_backup(output_dir=tmp_path)
    second = backup.create_backup(output_dir=tmp_path)

    assert first != second
    assert first.exists() and second.exists()


def test_create_backup_refuses_to_clobber_in_flight_temp_file(monkeypatch, tmp_path):
    """A pre-existing .part (a concurrent run's in-flight dump) must raise
    BackupError and be left untouched — never unlinked or overwritten."""
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)
    monkeypatch.setattr(backup, "_run", lambda cmd, **kw: _completed(0))
    monkeypatch.setattr(backup, "_timestamped_filename", lambda prefix="argos": "argos-fixed.dump")

    in_flight = tmp_path / "argos-fixed.dump.part"
    in_flight.write_bytes(b"other run's partial dump")

    with pytest.raises(backup.BackupError, match="another backup appears to be in flight"):
        backup.create_backup(output_dir=tmp_path)

    assert in_flight.read_bytes() == b"other run's partial dump"


def test_create_backup_keeps_password_out_of_argv(monkeypatch, tmp_path):
    """The DB password must never enter argv (ps exposure + _run debug logging).

    Regression pin for CodeQL py/clear-text-logging-sensitive-data: docker
    exec gets a bare `-e PGPASSWORD` and the value travels only via the
    subprocess env.
    """
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env")
        kwargs["stdout"].write(b"fake-dump-bytes")
        return _completed(0)

    monkeypatch.setattr(backup, "_run", fake_run)

    backup.create_backup(output_dir=tmp_path)

    assert "PGPASSWORD" in seen["cmd"]
    assert not any(arg.startswith("PGPASSWORD=") for arg in seen["cmd"])
    assert seen["env"]["PGPASSWORD"] == backup.settings.secrets.POSTGRES_PASSWORD


def test_create_backup_raises_and_cleans_up_on_pg_dump_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    def fake_run(cmd, **kwargs):
        kwargs["stdout"].write(b"partial")
        return _completed(1, stderr=b"pg_dump: error: connection failed")

    monkeypatch.setattr(backup, "_run", fake_run)

    with pytest.raises(backup.BackupError, match="pg_dump failed"):
        backup.create_backup(output_dir=tmp_path)

    # Temp and final files are both cleaned up — no partial dump left behind.
    assert list(tmp_path.glob("*")) == []


def test_create_backup_raises_on_empty_dump(monkeypatch, tmp_path):
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)
    monkeypatch.setattr(backup, "_run", lambda cmd, **kwargs: _completed(0))

    with pytest.raises(backup.BackupError, match="empty file"):
        backup.create_backup(output_dir=tmp_path)
    assert list(tmp_path.glob("*")) == []


def test_create_backup_prunes_old_dumps_when_keep_set(monkeypatch, tmp_path):
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    def fake_run(cmd, **kwargs):
        kwargs["stdout"].write(b"data")
        return _completed(0)

    monkeypatch.setattr(backup, "_run", fake_run)

    # Pre-existing dumps to be pruned.
    for i in range(3):
        p = tmp_path / f"argos-2020010{i}-000000.dump"
        p.write_bytes(b"old")
        # Ensure distinct mtimes so ordering is deterministic.
        import os as _os

        _os.utime(p, (i, i))

    backup.create_backup(output_dir=tmp_path, keep=2)

    remaining = sorted(tmp_path.glob("argos-*.dump"))
    # 3 pre-existing + 1 new = 4, keep=2 -> 2 remain (the 2 newest by mtime).
    assert len(remaining) == 2


# ---------------------------------------------------------------------------
# prune_old_backups
# ---------------------------------------------------------------------------


def test_prune_old_backups_keeps_n_newest(tmp_path):
    import os

    paths = []
    for i in range(5):
        p = tmp_path / f"argos-{i}.dump"
        p.write_bytes(b"x")
        os.utime(p, (i, i))
        paths.append(p)

    removed = backup.prune_old_backups(tmp_path, keep=2)

    remaining = set(tmp_path.glob("argos-*.dump"))
    assert remaining == {paths[3], paths[4]}
    assert set(removed) == {paths[0], paths[1], paths[2]}


# ---------------------------------------------------------------------------
# restore_backup
# ---------------------------------------------------------------------------


def test_restore_backup_missing_file_raises(tmp_path):
    with pytest.raises(backup.BackupError, match="not found"):
        backup.restore_backup(tmp_path / "nope.dump")


def test_restore_backup_requires_running_container(monkeypatch, tmp_path):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: False)
    with pytest.raises(backup.BackupError, match="is not running"):
        backup.restore_backup(dump)


def test_restore_backup_happy_path_cps_execs_and_cleans_up(monkeypatch, tmp_path):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed(0)

    monkeypatch.setattr(backup, "_run", fake_run)

    backup.restore_backup(dump, container="argos-db")

    assert calls[0][:2] == ["docker", "cp"]
    assert calls[0][2] == str(dump)
    assert calls[0][3] == "argos-db:/tmp/argos-x.dump"

    assert calls[1][:3] == ["docker", "exec", "-e"]
    assert "pg_restore" in calls[1]
    assert "--clean" in calls[1]
    assert "--if-exists" in calls[1]
    # Atomic restore: the whole thing runs in one transaction so a mid-restore
    # failure rolls back instead of leaving the DB partially dropped/restored.
    assert "--single-transaction" in calls[1]
    assert calls[1][-1] == "/tmp/argos-x.dump"

    # cleanup call removes the staged file inside the container.
    assert calls[2] == ["docker", "exec", "argos-db", "rm", "-f", "/tmp/argos-x.dump"]


def test_restore_backup_keeps_password_out_of_argv(monkeypatch, tmp_path):
    """Same secret-hygiene pin as the backup variant, for pg_restore."""
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    calls: list[tuple[list[str], object]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("env")))
        return _completed(0)

    monkeypatch.setattr(backup, "_run", fake_run)

    backup.restore_backup(dump)

    restore_cmd, restore_env = calls[1]
    assert "PGPASSWORD" in restore_cmd
    assert not any(arg.startswith("PGPASSWORD=") for arg in restore_cmd)
    assert restore_env["PGPASSWORD"] == backup.settings.secrets.POSTGRES_PASSWORD


def test_restore_backup_no_clean_omits_clean_flags(monkeypatch, tmp_path):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    calls: list[list[str]] = []
    monkeypatch.setattr(backup, "_run", lambda cmd, **kw: calls.append(cmd) or _completed(0))

    backup.restore_backup(dump, clean=False)

    assert "--clean" not in calls[1]
    assert "--if-exists" not in calls[1]
    # Atomicity is independent of --clean: a partial add-on restore is just as
    # unwanted, so --single-transaction stays on either way.
    assert "--single-transaction" in calls[1]


def test_create_backup_dump_file_is_private(monkeypatch, tmp_path):
    """A dump holds the whole database — it must be created 0600, not left
    world/group-readable by the process umask (0644 under the common 022)."""
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    def fake_run(cmd, **kwargs):
        kwargs["stdout"].write(b"secret-db-bytes")
        return _completed(0)

    monkeypatch.setattr(backup, "_run", fake_run)

    # Force a permissive umask so a bare "xb" open would yield 0644; the opener
    # must clamp to 0600 regardless.
    old_umask = os.umask(0o022)
    try:
        dest = backup.create_backup(output_dir=tmp_path)
    finally:
        os.umask(old_umask)

    mode = stat.S_IMODE(dest.stat().st_mode)
    assert mode == 0o600, f"dump created with mode {oct(mode)}, expected 0o600"


def test_restore_backup_cp_failure_raises_before_exec(monkeypatch, tmp_path):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _completed(1, stderr="docker cp: no such container")

    monkeypatch.setattr(backup, "_run", fake_run)

    with pytest.raises(backup.BackupError, match="docker cp"):
        backup.restore_backup(dump)

    # Only the failed `docker cp` ran — pg_restore/cleanup never attempted.
    assert len(calls) == 1


def test_restore_backup_pg_restore_failure_still_cleans_up(monkeypatch, tmp_path):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")
    monkeypatch.setattr(backup, "container_running", lambda container=backup.DEFAULT_CONTAINER_NAME: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "cp"]:
            return _completed(0)
        if "pg_restore" in cmd:
            return _completed(1, stderr="pg_restore: error: out of memory")
        return _completed(0)

    monkeypatch.setattr(backup, "_run", fake_run)

    with pytest.raises(backup.BackupError, match="pg_restore failed"):
        backup.restore_backup(dump)

    # Cleanup (`rm -f`) still ran despite the pg_restore failure.
    assert calls[-1][:3] == ["docker", "exec", "argos-db"]
    assert calls[-1][3:] == ["rm", "-f", "/tmp/argos-x.dump"]


# ---------------------------------------------------------------------------
# list_backups
# ---------------------------------------------------------------------------


def test_list_backups_empty_when_dir_missing(tmp_path):
    assert backup.list_backups(tmp_path / "nope") == []


def test_list_backups_newest_first(tmp_path):
    import os

    a = tmp_path / "argos-a.dump"
    b = tmp_path / "argos-b.dump"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    os.utime(a, (1, 1))
    os.utime(b, (2, 2))

    assert backup.list_backups(tmp_path) == [b, a]
