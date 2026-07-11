"""Tests for `argos backfill-trust` (ARG-211 Task 3).

Mock-level tests — no live DB required. Patches `argos.cli.AsyncSessionLocal`
and the small-model LLM client exactly like `test_cli_backfill_images.py`.

Deliberately NOT a live-DB integration test: ARG-206 (trust_rubric column)
landed recently, so the shared dev DB still has ~1600+ real rows with
trust_rubric IS NULL and no --limit guard would make that safe to touch with
a canned fake rubric — it would silently overwrite real trust data for the
whole legacy backlog. `test_cli_backfill_digests.py` (the template for this
command) has the same DB-safety constraint and is parser-only for the same
reason; this file goes one step further and also exercises the fill/dry-run
logic, but entirely against a mocked session.
"""
from __future__ import annotations

import argparse
import uuid
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argos.cli import _build_backfill_trust_parser, main


def _parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    common = argparse.ArgumentParser(add_help=False)
    _build_backfill_trust_parser(sub, common)
    return p


def test_backfill_trust_parses_defaults():
    args = _parser().parse_args(["backfill-trust"])
    assert args.command == "backfill-trust"
    assert args.limit is None
    assert args.dry_run is False


def test_backfill_trust_parses_flags():
    args = _parser().parse_args(["backfill-trust", "--limit", "5", "--dry-run"])
    assert args.limit == 5
    assert args.dry_run is True


# Mirrors the SELECT column shape in `_backfill_trust`:
# select(TechItem.id, TechItem.raw_content, TechItem.source_url, TechItem.corroboration_count)
_Row = namedtuple("_Row", ["id", "raw_content", "source_url", "corroboration_count"])


def _make_session_ctx():
    """A MagicMock async-context-manager whose __aenter__ returns a fresh
    AsyncMock session — mirrors `test_cli_backfill_images.py::_make_session_ctx`.
    """
    session = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return session, ctx


_RUBRIC_JSON = (
    '{"is_valid": true, "reason": "ok", "summary": "s", "category": "Alpha", '
    '"is_primary_source": true, "has_evidence_links": true, '
    '"has_concrete_numbers": false, "claim_evidence_balance": "balanced", '
    '"marketing_intensity": "low"}'
)


def test_backfill_trust_dry_run_reports_candidates_and_never_calls_llm(capsys):
    """--dry-run prints the candidate count, never calls the LLM, and issues
    only the SELECT (no write session, no UPDATE)."""
    session, session_ctx = _make_session_ctx()
    row = _Row(
        id=uuid.uuid4(),
        raw_content="raw content",
        source_url="https://example.com/a",
        corroboration_count=0,
    )
    session.execute = AsyncMock(return_value=MagicMock(**{"all.return_value": [row]}))

    fake_llm = AsyncMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.brain.llm_client.get_llm_client", return_value=fake_llm),
    ):
        rc = main(["backfill-trust", "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "1 candidate" in out
    fake_llm.query.assert_not_called()
    # Only the SELECT executed — dry-run must never touch the write path.
    assert session.execute.await_count == 1


def test_backfill_trust_fills_rubric_and_resynthesizes_score():
    """A real (non-dry-run) pass extracts the rubric via the fake LLM client
    and issues a trust_rubric IS NULL-guarded UPDATE with the deterministically
    re-synthesized trust_score."""
    from argos.brain.trust import (
        corroboration_score,
        score_rubric,
        source_prior,
        synthesize_trust,
    )
    from argos.config import settings

    row_id = uuid.uuid4()
    source_url = "https://example.com/fill-me"
    row = _Row(
        id=row_id,
        raw_content="raw content long enough to matter",
        source_url=source_url,
        corroboration_count=2,
    )

    session, session_ctx = _make_session_ctx()
    session.execute = AsyncMock(
        side_effect=[
            MagicMock(**{"all.return_value": [row]}),  # SELECT candidates
            MagicMock(rowcount=1),  # UPDATE for the one filled row
        ]
    )
    session.commit = AsyncMock()

    fake_llm = AsyncMock()
    fake_llm.query = AsyncMock(return_value=_RUBRIC_JSON)
    fake_llm.unload = AsyncMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.brain.llm_client.get_llm_client", return_value=fake_llm),
    ):
        rc = main(["backfill-trust"])

    assert rc == 0
    # The rubric LLM was called exactly once, against the "small" role
    # (the same model triage uses), never touching the "large" model.
    fake_llm.query.assert_awaited_once()
    assert fake_llm.query.call_args.args[0] == "small"
    fake_llm.unload.assert_awaited_once_with("small")

    # Exactly two execute() calls: the candidate SELECT, then the one UPDATE.
    assert session.execute.await_count == 2
    update_stmt = session.execute.await_args_list[1].args[0]
    params = update_stmt.compile().params

    # The non-overwrite guard (`trust_rubric IS NULL`) compiles to a literal
    # IS NULL in the SQL text with NO bound param, so it never shows up in
    # .compile().params — assert it off the rendered SQL. Without this, a
    # regression that dropped the guard from the WHERE clause (reintroducing
    # the clobbering race) would still pass every param assertion below.
    assert "trust_rubric IS NULL" in str(update_stmt)
    # And the UPDATE must target the intended row (id_1 is the bound row id).
    assert params["id_1"] == row_id

    expected_rubric = {
        "is_primary_source": True,
        "has_evidence_links": True,
        "has_concrete_numbers": False,
        "claim_evidence_balance": "balanced",
        "marketing_intensity": "low",
    }
    assert params["trust_rubric"] == expected_rubric

    trust_cfg = settings.user.trust
    expected_score = synthesize_trust(
        score_rubric(expected_rubric),
        source_prior(source_url, trust_cfg.source_tiers),
        corroboration_score(2),
        {
            "rubric": trust_cfg.weight_rubric,
            "prior": trust_cfg.weight_prior,
            "corroboration": trust_cfg.weight_corroboration,
        },
    )
    assert params["trust_score"] == pytest.approx(expected_score)


def test_backfill_trust_skips_row_on_llm_infra_failure(capsys):
    """A row whose rubric extraction raises is logged/skipped, not fatal —
    the run finishes with filled=0, skipped=1 rather than crashing."""
    row = _Row(
        id=uuid.uuid4(),
        raw_content="raw content",
        source_url="https://example.com/b",
        corroboration_count=0,
    )
    session, session_ctx = _make_session_ctx()
    session.execute = AsyncMock(return_value=MagicMock(**{"all.return_value": [row]}))
    session.commit = AsyncMock()

    fake_llm = AsyncMock()
    fake_llm.query = AsyncMock(side_effect=RuntimeError("ollama unreachable"))
    fake_llm.unload = AsyncMock()

    with (
        patch("argos.cli.AsyncSessionLocal", return_value=session_ctx),
        patch("argos.brain.llm_client.get_llm_client", return_value=fake_llm),
    ):
        rc = main(["backfill-trust"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "filled=0, skipped=1" in out
    # No UPDATE was issued for the failed row — only the candidate SELECT.
    assert session.execute.await_count == 1
    fake_llm.unload.assert_awaited_once_with("small")
