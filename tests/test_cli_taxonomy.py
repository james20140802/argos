"""`argos taxonomy` 명령 (ARG-311 T5) — 목 세션·목 LLM, DB/Ollama 불필요.

자동화 셸의 FORCE_COLOR 영향을 피하려면 `env -u FORCE_COLOR`로 실행한다.
"""
from __future__ import annotations

import pytest

from argos import cli, taxonomy_cli
from argos.brain import taxonomy as tx
from argos.brain import taxonomy_store

DESC = "설명은 열 글자를 넘는 한 문장."


class FakeSession:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1


class FakeFactory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def db(monkeypatch):
    """확정본 = None, apply_plan 호출을 기록하는 가짜 DB."""
    state = {"confirmed": None, "applied": []}
    session = FakeSession()
    monkeypatch.setattr(taxonomy_cli, "AsyncSessionLocal", FakeFactory(session))

    async def fake_load(_session):
        return state["confirmed"]

    async def fake_apply(_session, plan):
        state["applied"].append(plan)

    monkeypatch.setattr(taxonomy_store, "load_confirmed", fake_load)
    monkeypatch.setattr(taxonomy_store, "apply_plan", fake_apply)
    state["session"] = session
    return state


@pytest.fixture
def f(tmp_path):
    return str(tmp_path / "draft.json")


def run(*argv):
    return cli.main(["taxonomy", *argv])


def test_draft_seed_prints_five_tops_and_writes_file_only(db, f, capsys, monkeypatch):
    # 초안 단계에서 DB 세션을 열면 실패하게 한다
    def boom():
        raise AssertionError("draft --seed must not open a DB session")

    monkeypatch.setattr(taxonomy_cli, "AsyncSessionLocal", boom)
    assert run("draft", "--seed", "--file", f) == 0
    out = capsys.readouterr().out
    for name in (
        "AI 모델 및 엔진",
        "AI 에이전트 및 어시스턴트",
        "에이전트 인프라 및 도구",
        "데이터 및 개발 인프라",
        "평가",
    ):
        assert name in out
    assert "DB에는 아무것도 쓰지 않았습니다" in out
    assert tx.load_file(f) == tx.SPIKE_SEED
    assert db["applied"] == []


def test_draft_refuses_overwrite_without_force(db, f, capsys):
    assert run("draft", "--seed", "--file", f) == 0
    assert run("draft", "--seed", "--file", f) == 1
    assert run("draft", "--seed", "--force", "--file", f) == 0


def test_show_when_nothing_confirmed(db, capsys):
    assert run("show") == 0
    assert "확정된 갈래 없음" in capsys.readouterr().out


def test_edit_commands_print_changed_tree_and_persist(db, f, capsys):
    run("draft", "--seed", "--file", f)
    capsys.readouterr()

    assert run("rename", "벤치마크 (Benchmarks)", "벤치마크", "--file", f) == 0
    assert "벤치마크" in capsys.readouterr().out

    assert (
        run(
            "split", "버전 관리·협업 (Version Control)",
            "--part", f"깃 호스팅={DESC}", "--part", f"리뷰 도구={DESC}", "--file", f,
        )
        == 0
    )
    assert "깃 호스팅" in capsys.readouterr().out

    assert run("merge", "깃 호스팅", "리뷰 도구", "--file", f) == 0
    tree = tx.load_file(f)
    names = [c.name for _d, c, _p in tx.iter_branches(tree)]
    assert "리뷰 도구" in names and "깃 호스팅" not in names

    assert run("delete", "리뷰 도구", "--file", f) == 0
    names = [c.name for _d, c, _p in tx.iter_branches(tx.load_file(f))]
    assert "리뷰 도구" not in names


def test_rename_to_word_list_is_rejected_and_file_unchanged(db, f, capsys):
    run("draft", "--seed", "--file", f)
    before = open(f, encoding="utf-8").read()
    capsys.readouterr()
    rc = run("rename", "평가", "claude, went, seconds", "--file", f)
    assert rc == 3
    assert "단어 나열" in capsys.readouterr().out
    assert open(f, encoding="utf-8").read() == before


def test_edit_without_draft_file_explains(db, f, capsys):
    assert run("rename", "평가", "검증", "--file", f) == 3
    assert "초안 파일이 없습니다" in capsys.readouterr().out


def test_apply_dry_run_never_writes(db, f, capsys):
    run("draft", "--seed", "--file", f)
    capsys.readouterr()
    assert run("apply", "--dry-run", "--file", f) == 0
    out = capsys.readouterr().out
    assert "추가: 'AI 모델 및 엔진'" in out
    assert db["applied"] == [] and db["session"].commits == 0


def test_apply_declined_never_writes(db, f, monkeypatch):
    run("draft", "--seed", "--file", f)
    monkeypatch.setattr("builtins.input", lambda _p="": "n")
    assert run("apply", "--file", f) == 1
    assert db["applied"] == [] and db["session"].commits == 0


def test_apply_confirmed_writes_once_and_commits(db, f, monkeypatch, capsys):
    run("draft", "--seed", "--file", f)
    monkeypatch.setattr("builtins.input", lambda _p="": "y")
    assert run("apply", "--file", f) == 0
    assert len(db["applied"]) == 1 and db["session"].commits == 1
    assert "확정했습니다" in capsys.readouterr().out


def test_apply_yes_skips_prompt(db, f, monkeypatch):
    run("draft", "--seed", "--file", f)

    def no_input(_p=""):
        raise AssertionError("--yes must not prompt")

    monkeypatch.setattr("builtins.input", no_input)
    assert run("apply", "--yes", "--file", f) == 0
    assert len(db["applied"]) == 1


def test_apply_no_changes(db, f, capsys):
    run("draft", "--seed", "--file", f)
    db["confirmed"] = tuple(  # 파일과 같은 내용의 확정본(id 부여)
        tx.Branch(
            b.name, b.description, id=f"t{i}",
            children=tuple(tx.Branch(c.name, c.description, id=f"t{i}c{j}") for j, c in enumerate(b.children)),
        )
        for i, b in enumerate(tx.SPIKE_SEED)
    )
    # 파일에는 id가 없으므로 같은 내용이어도 id 기준으로는 전부 새 노드다 -> 변경 있음.
    capsys.readouterr()
    assert run("apply", "--dry-run", "--file", f) == 0
    assert "삭제" in capsys.readouterr().out
    # export 후에는 변경 없음
    ef = f + ".export.json"
    assert run("export", "--file", ef) == 0
    assert run("apply", "--dry-run", "--file", ef) == 0
    assert "변경 없음" in capsys.readouterr().out


def test_export_without_confirmed(db, f, capsys):
    assert run("export", "--file", f) == 1
    assert "확정된 갈래 없음" in capsys.readouterr().out


def test_draft_llm_path_with_keep_top_uses_confirmed_tops(db, f, monkeypatch, capsys):
    seen = {}

    async def fake_build(session, llm, *, limit=None, keep_top=None):
        seen["keep_top"] = keep_top
        seen["limit"] = limit
        return tx.SPIKE_SEED

    import argos.brain.taxonomy_draft as td
    import argos.brain.llm_client as lc

    monkeypatch.setattr(td, "build_draft", fake_build)
    monkeypatch.setattr(lc, "get_taxonomy_llm_client", lambda: object())

    assert run("draft", "--keep-top", "--limit", "10", "--file", f) == 0
    assert seen["keep_top"] == tx.SPIKE_SEED  # 확정본이 없으면 시드
    assert seen["limit"] == 10

    db["confirmed"] = tx.SPIKE_SEED[:4] + (tx.SPIKE_SEED[4],)
    run("draft", "--keep-top", "--force", "--file", f)
    assert seen["keep_top"] is db["confirmed"]
