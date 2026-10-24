"""``argos alias-names`` — 미리보기 후 확인을 받고서만 쓴다 (ARG-312). DB 없이."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos import cli
from argos.brain import alias_store
from argos.brain.entity_aliases import AliasLink, AliasPlan, AliasRule, HeldPair


def _plan(changed: bool = True) -> AliasPlan:
    added = (
        (
            AliasLink(
                uuid.uuid4(), uuid.uuid4(), "openai codex", "codex", AliasRule.VENDOR_PREFIX
            ),
        )
        if changed
        else ()
    )
    return AliasPlan(
        added=added,
        removed=(),
        unchanged=2,
        held=(HeldPair("claude code", "code", "base_lowercase"),),
        groups=1 if changed else 0,
    )


@pytest.fixture
def wired(monkeypatch):
    session = MagicMock()
    session.commit = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: ctx)
    apply = AsyncMock(return_value=1)
    monkeypatch.setattr(alias_store, "plan_aliases", AsyncMock(return_value=_plan()))
    monkeypatch.setattr(alias_store, "alias_table_exists", AsyncMock(return_value=True))
    monkeypatch.setattr(alias_store, "apply_aliases", apply)
    return session, apply


def test_dry_run_prints_preview_and_writes_nothing(wired, capsys, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("dry-run must not ask"))
    assert cli.main(["alias-names", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "별칭 묶음 1개 (별칭 +1 / -0, 유지 2)" in out
    assert "openai codex → codex" in out
    assert "↛ claude code / code" in out and "base_lowercase" in out
    apply.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_declined_confirmation_writes_nothing(wired, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert cli.main(["alias-names"]) != 0
    apply.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_confirmed_run_applies_and_commits(wired, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert cli.main(["alias-names"]) == 0
    apply.assert_awaited_once()
    session.commit.assert_awaited_once()


def test_yes_flag_skips_the_prompt(wired, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("--yes must not ask"))
    assert cli.main(["alias-names", "--yes"]) == 0
    apply.assert_awaited_once()


def test_no_changes_reports_and_exits_without_writing(wired, capsys, monkeypatch):
    session, apply = wired
    monkeypatch.setattr(alias_store, "plan_aliases", AsyncMock(return_value=_plan(changed=False)))
    assert cli.main(["alias-names", "--yes"]) == 0
    assert "변경 없음" in capsys.readouterr().out
    apply.assert_not_awaited()


def test_missing_table_blocks_write_but_dry_run_still_previews(wired, capsys, monkeypatch):
    session, apply = wired
    monkeypatch.setattr(alias_store, "alias_table_exists", AsyncMock(return_value=False))
    assert cli.main(["alias-names", "--dry-run"]) == 0
    assert "alembic upgrade head" in capsys.readouterr().out
    assert cli.main(["alias-names", "--yes"]) != 0
    apply.assert_not_awaited()
    session.commit.assert_not_awaited()
