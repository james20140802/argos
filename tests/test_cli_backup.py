"""Tests for `argos backup` / `argos restore` CLI subcommands (ARG-192).

Mocks `argos.cli.backup` (the `argos.backup` module imported into cli.py) so
no real docker/subprocess/DB calls happen here — see `tests/test_backup.py`
for coverage of the underlying module itself.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from argos import backup
from argos.cli import main


# ---------------------------------------------------------------------------
# argos backup
# ---------------------------------------------------------------------------


def test_backup_command_success_prints_path_and_returns_zero(tmp_path, capsys):
    dest = tmp_path / "argos-20260704-000000.dump"
    with patch("argos.cli.backup.create_backup", return_value=dest) as create_mock:
        rc = main(["backup"])

    assert rc == 0
    create_mock.assert_called_once_with(container=backup.DEFAULT_CONTAINER_NAME, output_dir=None, keep=None)
    assert str(dest) in capsys.readouterr().out


def test_backup_command_passes_through_options(tmp_path):
    dest = tmp_path / "argos-x.dump"
    with patch("argos.cli.backup.create_backup", return_value=dest) as create_mock:
        rc = main(
            [
                "backup",
                "--container",
                "custom-db",
                "--output-dir",
                str(tmp_path),
                "--keep",
                "5",
            ]
        )

    assert rc == 0
    create_mock.assert_called_once_with(container="custom-db", output_dir=Path(tmp_path), keep=5)


def test_backup_command_reports_error_and_returns_nonzero(capsys):
    with patch("argos.cli.backup.create_backup", side_effect=backup.BackupError("container not running")):
        rc = main(["backup"])

    assert rc != 0
    assert "container not running" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# argos restore
# ---------------------------------------------------------------------------


def test_restore_command_missing_dump_file_returns_error(tmp_path, capsys):
    missing = tmp_path / "nope.dump"
    with patch("argos.cli.backup.restore_backup") as restore_mock:
        rc = main(["restore", str(missing), "--yes"])

    assert rc != 0
    restore_mock.assert_not_called()
    assert "찾을 수 없습니다" in capsys.readouterr().err


def test_restore_command_with_yes_skips_prompt_and_restores(tmp_path):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")

    with patch("argos.cli.backup.restore_backup") as restore_mock:
        rc = main(["restore", str(dump), "--yes"])

    assert rc == 0
    restore_mock.assert_called_once_with(dump, container=backup.DEFAULT_CONTAINER_NAME, clean=True)


def test_restore_command_no_clean_flag_passes_clean_false(tmp_path):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")

    with patch("argos.cli.backup.restore_backup") as restore_mock:
        rc = main(["restore", str(dump), "--yes", "--no-clean"])

    assert rc == 0
    restore_mock.assert_called_once_with(dump, container=backup.DEFAULT_CONTAINER_NAME, clean=False)


def test_restore_command_without_yes_prompts_and_cancels_on_no(tmp_path):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")

    with (
        patch("argos.cli.backup.restore_backup") as restore_mock,
        patch("builtins.input", return_value="n"),
    ):
        rc = main(["restore", str(dump)])

    assert rc != 0
    restore_mock.assert_not_called()


def test_restore_command_without_yes_prompts_and_proceeds_on_yes(tmp_path):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")

    with (
        patch("argos.cli.backup.restore_backup") as restore_mock,
        patch("builtins.input", return_value="y"),
    ):
        rc = main(["restore", str(dump)])

    assert rc == 0
    restore_mock.assert_called_once_with(dump, container=backup.DEFAULT_CONTAINER_NAME, clean=True)


def test_restore_command_reports_error_and_returns_nonzero(tmp_path, capsys):
    dump = tmp_path / "argos-x.dump"
    dump.write_bytes(b"data")

    with patch("argos.cli.backup.restore_backup", side_effect=backup.BackupError("pg_restore failed")):
        rc = main(["restore", str(dump), "--yes"])

    assert rc != 0
    assert "pg_restore failed" in capsys.readouterr().err
