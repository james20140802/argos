"""자동 처리가 확정 갈래를 바꾸지 않음을 지키는 검사 (ARG-311 T6) — DB 불필요.

AC "확정된 위쪽 갈래는 자동 처리로 바뀌지 않는다"를 구조로 보장한다:
``role_branches``에 쓰는 경로는 ``taxonomy_store.apply_plan`` 하나이고, 그것을
부르는 곳은 사람이 친 ``argos taxonomy apply``(taxonomy_cli)뿐이다. 이 테스트는
소스를 AST로 읽어 그 경계가 새지 않는지 본다.

``RoleBranch``를 참조하는 모듈 허용 목록을 일부러 좁게 둔다. 나중에 읽기 전용
소비자(ARG-313 자동 배정 등)가 생기면 이 목록에 **의식적으로** 추가해야 하고,
그때 "읽기만 하는가"를 리뷰에서 확인한다.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "argos"

APPLY_ALLOWED = {"brain/taxonomy_store.py", "taxonomy_cli.py"}
ROLE_BRANCH_ALLOWED = {
    "models/role_branch.py",
    "models/__init__.py",
    "brain/taxonomy_store.py",
}


def _modules() -> dict[str, ast.AST]:
    out = {}
    for path in SRC.rglob("*.py"):
        out[path.relative_to(SRC).as_posix()] = ast.parse(path.read_text(encoding="utf-8"))
    return out


def _mentions(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.alias) and node.name.split(".")[-1] == name:
            return True
    return False


def test_only_the_apply_command_can_call_apply_plan():
    offenders = sorted(
        rel for rel, tree in _modules().items() if _mentions(tree, "apply_plan")
    )
    assert set(offenders) <= APPLY_ALLOWED, (
        f"apply_plan referenced outside the apply command: {offenders}"
    )
    assert "taxonomy_cli.py" in offenders  # 경계가 실제로 존재함을 확인


def test_role_branch_table_is_referenced_only_by_model_and_store():
    offenders = sorted(
        rel for rel, tree in _modules().items() if _mentions(tree, "RoleBranch")
    )
    assert set(offenders) <= ROLE_BRANCH_ALLOWED, (
        f"RoleBranch referenced outside the allowlist: {offenders}. "
        "읽기 전용 소비자라면 허용 목록에 추가하고 리뷰에서 쓰기가 없음을 확인하세요."
    )


def test_draft_module_is_read_only_wrt_confirmed_branches():
    tree = _modules()["brain/taxonomy_draft.py"]
    for forbidden in ("apply_plan", "RoleBranch", "taxonomy_store"):
        assert not _mentions(tree, forbidden)


def test_cli_entrypoint_does_not_reach_apply_directly():
    tree = _modules()["cli.py"]
    assert not _mentions(tree, "apply_plan")


def test_no_scheduled_job_runs_taxonomy():
    from argos import scheduler

    source = (SRC / "scheduler.py").read_text(encoding="utf-8")
    assert "taxonomy" not in source.lower()
    assert hasattr(scheduler, "__file__")
