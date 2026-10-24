"""`argos taxonomy assign/why/eval` (ARG-313) — 목 세션·가짜 계획, DB/Ollama 불필요.

자동화 셸의 FORCE_COLOR 영향을 피하려면 `env -u FORCE_COLOR`로 실행한다.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from argos import cli, taxonomy_cli
from argos.brain import taxonomy as tx
from argos.brain import taxonomy_store
from argos.brain import tech_classify_store as store
from argos.brain.tech_classify import AssignmentWrite, AssignPlan, ProfileWrite, TargetInput
from argos.models.technology_profile import TechVerdict

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass


class FakeFactory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *exc):
        return False


def _plan():
    return AssignPlan(
        run_id=uuid.uuid4(),
        profiles=(ProfileWrite("e1", "claude code", "Claude Code", TechVerdict.TECHNOLOGY,
                               description="코드를 짜는 에이전트"),),
        assignments=(
            AssignmentWrite("e1", "claude code", "Claude Code", "b1", "에이전트 > 코딩", None, {}),
            AssignmentWrite("e2", "scion", "Scion", None, None, "InsufficientEvidence", {}),
        ),
        screened=(("AI", "Common"),),
        llm_calls=3,
    )


@pytest.fixture
def env(monkeypatch):
    state = {"tree": tx.SPIKE_SEED, "written": [], "tables": True, "rows": ["r"]}
    session = FakeSession()
    state["session"] = session
    monkeypatch.setattr(taxonomy_cli, "AsyncSessionLocal", FakeFactory(session))

    async def load_confirmed(_s):
        return state["tree"]

    async def tables_exist(_s):
        return state["tables"]

    async def load_name_rows(_s):
        return state["rows"]

    def select_targets(rows, cfg, *, all_names=False, limit=None):
        state["select"] = (all_names, limit)
        return [(SimpleNamespace(key="claude code"), True)] if rows else []

    async def collect(_s, selected, rows):
        return [TargetInput("e1", "claude code", "Claude Code", 3, (), None)]

    async def current(_s, tree):
        return {}

    async def make_plan(tree, inputs, cur):
        state["made"] = state.get("made", 0) + 1
        return _plan()

    async def write_plan(_s, plan):
        state["written"].append(plan)
        return 3

    monkeypatch.setattr(taxonomy_store, "load_confirmed", load_confirmed)
    monkeypatch.setattr(store, "tables_exist", tables_exist)
    monkeypatch.setattr(store, "load_name_rows", load_name_rows)
    monkeypatch.setattr(store, "select_targets", select_targets)
    monkeypatch.setattr(store, "current_assignments", current)
    monkeypatch.setattr(store, "write_plan", write_plan)
    monkeypatch.setattr(taxonomy_cli, "_collect_inputs", collect)
    monkeypatch.setattr(taxonomy_cli, "_make_plan", make_plan)
    return state


def run(*argv):
    return cli.main(["taxonomy", *argv])


def test_assign_without_confirmed_tree_explains_and_exits_ok(env, capsys):
    env["tree"] = None
    assert run("assign", "--yes") == 0
    assert "먼저 `argos taxonomy draft`" in capsys.readouterr().out
    assert env["written"] == []


def test_assign_dry_run_shows_preview_and_never_writes(env, capsys):
    assert run("assign", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "■ 에이전트 > 코딩 (1)" in out
    assert "Claude Code — 코드를 짜는 에이전트" in out
    assert "Scion (InsufficientEvidence)" in out
    assert "AI (Common)" in out
    assert env["written"] == []


def test_assign_asks_and_writes_the_same_plan_without_recomputing(env, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _p: "n")
    assert run("assign") == 1
    assert env["written"] == []

    monkeypatch.setattr("builtins.input", lambda _p: "y")
    assert run("assign") == 0
    assert len(env["written"]) == 1 and env["made"] == 2  # 실행마다 한 번만 계산
    assert env["session"].commits == 1


def test_assign_yes_for_scheduler_and_all_flag(env, capsys):
    assert run("assign", "--yes", "--all") == 0
    assert env["select"] == (True, None)  # 소급은 회당 상한 없음
    assert run("assign", "--yes") == 0
    assert env["select"] == (False, 40)


def test_assign_with_nothing_new(env, capsys):
    env["rows"] = []
    assert run("assign", "--yes") == 0
    assert "새로 배정할 기술 없음" in capsys.readouterr().out


def test_assign_without_tables_tells_to_migrate(env, capsys):
    env["tables"] = False
    assert run("assign") == 1
    assert "alembic upgrade head" in capsys.readouterr().out


def test_eval_assignments_file_reproduces_spike_d(capsys):
    rc = run(
        "eval",
        "--truth", str(ROOT / "evals" / "taxonomy_truth_arg313.json"),
        "--assignments", str(ROOT / "tests" / "brain" / "fixtures" / "taxonomy_spike" / "assignD.json"),
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "F1 0.81" in out and "정밀도 0.82" in out and "재현율 0.80" in out


def test_eval_confirmed_reads_db_labels(env, capsys, monkeypatch):
    async def labels(_s, gold):
        return {"Codex": "b1", "Claude Code": "b1", "Cursor": None}

    monkeypatch.setattr(store, "assigned_labels", labels)
    assert run("eval", "--truth", str(ROOT / "evals" / "taxonomy_truth_arg313.json")) == 0
    out = capsys.readouterr().out
    assert "확정 트리" in out
    assert "미분류(단독 처리) 1개: Cursor" in out
    assert "DB/결과에 없는 이름(단독 처리) 30개" in out


def test_why_prints_branch_and_evidence(env, capsys, monkeypatch):
    profile = SimpleNamespace(verdict=TechVerdict.TECHNOLOGY, not_tech_reason=None, kind="coding agent",
                              description="코드를 짜는 에이전트", described_doc_count=12, model="gemma4:26b")
    assignment = SimpleNamespace(
        source=SimpleNamespace(value="Auto"), unclassified_reason=None,
        evidence={
            "neighbors": [{"name": "codex", "count": 4, "branch": "에이전트 > 코딩",
                           "examples": ["Claude Code, Codex, and Cursor"]}],
            "isa": ["terminal coding agent"], "external_tags": ["gh:cli"],
            "sentence_count": 9, "llm_reason": "listed with coding agents",
        },
    )

    async def explain(_s, name, tree):
        return store.Explanation("Claude Code", "claude code", profile, assignment, "에이전트 > 코딩")

    monkeypatch.setattr(store, "explain", explain)
    assert run("why", "claude code") == 0
    out = capsys.readouterr().out
    for s in ("갈래: 에이전트 > 코딩", "codex (4회", "Claude Code, Codex, and Cursor",
              "is a terminal coding agent", "gh:cli", "문장 수: 9", "LLM 이유"):
        assert s in out
