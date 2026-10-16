"""``argos backfill-entities`` — 미리보기 후 확인을 받고서만 쓴다 (ARG-304). DB 없이."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from argos import cli
from argos.brain import entity_refill
from argos.brain.entity_refill import DocChange, RefillPlan


@pytest.fixture
def wired(monkeypatch):
    plan = RefillPlan(
        total=10,
        changes=(
            DocChange(uuid.uuid4(), "Some title", ("frontier",), ("we", "openai")),
        ),
    )
    session = MagicMock()
    session.commit = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: ctx)
    apply = AsyncMock(return_value=1)
    monkeypatch.setattr(entity_refill, "plan_refill", AsyncMock(return_value=(plan, {})))
    monkeypatch.setattr(entity_refill, "apply_refill", apply)
    return session, apply


def test_backfill_entities_dry_run_writes_nothing(wired, capsys, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("dry-run must not ask"))
    assert cli.main(["backfill-entities", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "전체 10건 중 1건 변경 (이름 링크 +1 / -2)" in out
    assert "Some title" in out and "frontier" in out and "openai" in out
    apply.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_backfill_entities_declined_writes_nothing(wired, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert cli.main(["backfill-entities"]) != 0
    apply.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_backfill_entities_confirmed_applies(wired, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert cli.main(["backfill-entities"]) == 0
    apply.assert_awaited_once()
    session.commit.assert_awaited_once()


def test_backfill_entities_yes_applies_without_asking(wired, monkeypatch):
    session, apply = wired
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("--yes must not ask"))
    assert cli.main(["backfill-entities", "--yes"]) == 0
    apply.assert_awaited_once()
    session.commit.assert_awaited_once()


def test_backfill_entities_no_changes_does_not_ask(wired, monkeypatch, capsys):
    session, apply = wired
    monkeypatch.setattr(
        entity_refill, "plan_refill", AsyncMock(return_value=(RefillPlan(3, ()), {}))
    )
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("must not ask"))
    assert cli.main(["backfill-entities"]) == 0
    assert "변경 없음" in capsys.readouterr().out
    apply.assert_not_awaited()
