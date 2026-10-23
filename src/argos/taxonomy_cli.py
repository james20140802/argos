"""``argos taxonomy`` — 역할 갈래 초안·편집·확정 명령 (ARG-311).

흐름: ``draft``(파일) -> ``rename/merge/split/delete``(파일 위 편집) -> ``apply``
(현재 확정본과 차이 미리보기 -> y/N -> DB). 확정 후 다시 고칠 때도 같은 길이다:
``export``(DB -> 파일) -> 편집 -> ``apply``.

DB에 **쓰는** 명령은 ``apply``뿐이고 ``apply``는 확인 없이는 아무것도 쓰지 않는다.
이 모듈이 ``taxonomy_store.apply_plan``을 부르는 유일한 곳이어야 한다
(``tests/brain/test_taxonomy_isolation.py``).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from argos.brain import taxonomy as tx
from argos.brain import taxonomy_store
from argos.brain.taxonomy import Tree
from argos.database import AsyncSessionLocal

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_VALIDATION = 3

DEFAULT_FILE = Path.home() / ".config" / "argos" / "taxonomy-draft.json"


def _positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not an integer")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be ≥ 1, got {n}")
    return n


def _key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"'{value}' 형식은 이름=값 이어야 합니다")
    key, _, val = value.partition("=")
    return key.strip(), val.strip()


def register(sub: argparse._SubParsersAction, common: argparse.ArgumentParser) -> None:
    """``argos taxonomy ...`` 파서를 등록한다."""
    p = sub.add_parser(
        "taxonomy",
        help="Draft, edit and confirm the role-branch tree (draft -> edit -> apply)",
        parents=[common],
        description=(
            "역할 갈래 트리를 초안으로 받아 고치고 한 번 확정한다. draft/편집 명령은 파일만 "
            "바꾸고, DB에 쓰는 것은 확인을 거치는 apply뿐이다."
        ),
    )
    actions = p.add_subparsers(dest="taxonomy_action", required=True)

    def file_opt(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--file",
            type=Path,
            default=DEFAULT_FILE,
            help=f"초안 파일 (기본: {DEFAULT_FILE})",
        )

    d = actions.add_parser("draft", help="갈래 초안을 만들어 보여 주고 파일로 저장 (DB 미기록)")
    d.add_argument("--seed", action="store_true", help="내장 시드로 시작 (LLM 없음)")
    d.add_argument(
        "--keep-top",
        action="store_true",
        help="현재 확정본(없으면 시드)의 위쪽 갈래를 고정하고 아래 갈래만 LLM에게 제안받기",
    )
    d.add_argument("--limit", type=_positive_int, default=None, metavar="N", help="입력 엔티티 수")
    d.add_argument("--force", action="store_true", help="기존 초안 파일 덮어쓰기")
    file_opt(d)

    actions.add_parser("show", help="확정된 갈래 트리를 보여 준다")

    e = actions.add_parser("export", help="확정본을 초안 파일로 내보낸다 (다시 고칠 때)")
    e.add_argument("--force", action="store_true", help="기존 초안 파일 덮어쓰기")
    file_opt(e)

    r = actions.add_parser("rename", help="갈래 이름 바꾸기 (초안 파일)")
    r.add_argument("name")
    r.add_argument("new_name")
    file_opt(r)

    m = actions.add_parser("merge", help="갈래를 다른 갈래에 합치기 (초안 파일)")
    m.add_argument("absorbed", help="흡수되어 사라질 갈래")
    m.add_argument("into", help="남는 갈래")
    file_opt(m)

    sp = actions.add_parser("split", help="갈래 나누기 (초안 파일)")
    sp.add_argument("name")
    sp.add_argument(
        "--part",
        action="append",
        type=_key_value,
        default=[],
        metavar="새이름=설명",
        help="새 갈래 (2개 이상, 반복)",
    )
    sp.add_argument(
        "--move",
        action="append",
        type=_key_value,
        default=[],
        metavar="아래갈래=새위갈래",
        help="위 갈래를 나눌 때 아래 갈래를 어느 새 갈래로 보낼지 (반복)",
    )
    file_opt(sp)

    de = actions.add_parser("delete", help="갈래 지우기 (초안 파일)")
    de.add_argument("name")
    de.add_argument("--force", action="store_true", help="아래 갈래가 있는 위 갈래도 지우기")
    file_opt(de)

    a = actions.add_parser("apply", help="초안 파일을 확정본으로 저장 (차이 미리보기 -> y/N)")
    a.add_argument("--dry-run", action="store_true", help="미리보기만 보이고 종료")
    a.add_argument("--yes", action="store_true", help="확인 질문 생략")
    file_opt(a)


def run(args: argparse.Namespace) -> int:
    action = args.taxonomy_action
    try:
        if action == "draft":
            return asyncio.run(_draft(args))
        if action == "show":
            return asyncio.run(_show())
        if action == "export":
            return asyncio.run(_export(args))
        if action == "apply":
            return asyncio.run(_apply(args))
        return _edit(args)
    except tx.TaxonomyError as exc:
        print("❌ 갈래 트리가 규칙을 어깁니다:")
        for line in exc.errors:
            print(f"  - {line}")
        return EXIT_VALIDATION


# --------------------------------------------------------------------- #
def _print_tree(tree: Tree) -> None:
    print(tx.render(tree))
    for w in tx.warnings(tree):
        print(f"⚠️  {w}")


def _load_draft(path: Path) -> Tree:
    if not path.exists():
        raise tx.TaxonomyError(
            [f"초안 파일이 없습니다: {path} — `argos taxonomy draft --seed` 또는 `export`로 먼저 만드세요"]
        )
    return tx.load_file(path)


async def _draft(args: argparse.Namespace) -> int:
    path: Path = args.file
    if path.exists() and not args.force:
        print(f"❌ 초안 파일이 이미 있습니다: {path} (덮어쓰려면 --force)")
        return EXIT_GENERIC

    if args.seed:
        tree = tx.SPIKE_SEED
    else:
        from argos.brain.llm_client import get_taxonomy_llm_client
        from argos.brain.taxonomy_draft import TaxonomyDraftError, build_draft

        async with AsyncSessionLocal() as session:
            keep: Tree | None = None
            if args.keep_top:
                keep = await taxonomy_store.load_confirmed(session) or tx.SPIKE_SEED
            try:
                tree = await build_draft(
                    session, get_taxonomy_llm_client(), limit=args.limit, keep_top=keep
                )
            except TaxonomyDraftError as exc:
                print("❌ 초안을 만들지 못했습니다:")
                for line in exc.errors:
                    print(f"  - {line}")
                return EXIT_GENERIC

    _print_tree(tree)
    tx.save_file(tree, path)
    print(f"\n초안을 저장했습니다: {path}  (아직 DB에는 아무것도 쓰지 않았습니다)")
    return EXIT_OK


async def _show() -> int:
    async with AsyncSessionLocal() as session:
        tree = await taxonomy_store.load_confirmed(session)
    if tree is None:
        print("확정된 갈래 없음")
        return EXIT_OK
    _print_tree(tree)
    return EXIT_OK


async def _export(args: argparse.Namespace) -> int:
    path: Path = args.file
    if path.exists() and not args.force:
        print(f"❌ 초안 파일이 이미 있습니다: {path} (덮어쓰려면 --force)")
        return EXIT_GENERIC
    async with AsyncSessionLocal() as session:
        tree = await taxonomy_store.load_confirmed(session)
    if tree is None:
        print("확정된 갈래 없음 — 내보낼 것이 없습니다")
        return EXIT_GENERIC
    tx.save_file(tree, path)
    print(f"확정본을 초안 파일로 내보냈습니다: {path}")
    return EXIT_OK


def _edit(args: argparse.Namespace) -> int:
    """초안 파일 위의 편집. 검증을 통과해야 파일이 바뀐다."""
    path: Path = args.file
    tree = _load_draft(path)
    action = args.taxonomy_action
    if action == "rename":
        new = tx.rename(tree, args.name, args.new_name)
    elif action == "merge":
        new = tx.merge(tree, args.absorbed, args.into)
    elif action == "split":
        new = tx.split(tree, args.name, list(args.part), dict(args.move) or None)
    elif action == "delete":
        new = tx.delete(tree, args.name, force=args.force)
    else:  # pragma: no cover — argparse가 막는다
        raise tx.TaxonomyError([f"알 수 없는 명령: {action}"])
    tx.save_file(new, path)
    _print_tree(new)
    return EXIT_OK


async def _apply(args: argparse.Namespace) -> int:
    new = _load_draft(args.file)
    async with AsyncSessionLocal() as session:
        current = await taxonomy_store.load_confirmed(session)
        plan = taxonomy_store.plan_apply(current, new)

        if current is None:
            print("현재 확정된 갈래 없음 — 처음 확정합니다.")
        if plan.is_empty:
            print("변경 없음")
            return EXIT_OK
        if current is not None and not any(b.id for _d, b, _p in tx.iter_branches(new)):
            print(
                "⚠️  이 파일에는 갈래 id가 없습니다 (export로 받은 파일이 아님) — "
                "기존 갈래가 전부 삭제되고 새로 추가됩니다. 고쳐 쓰려면 `argos taxonomy export`를 쓰세요."
            )
        print("확정본에 적용될 변경:")
        for line in plan.summary_lines():
            print(line)
        for w in tx.warnings(new):
            print(f"⚠️  {w}")
        if args.dry_run:
            return EXIT_OK

        if not args.yes:
            reply = input("계속하시겠습니까? [y/N] ").strip().lower()
            if reply not in ("y", "yes"):
                print("취소되었습니다.")
                return EXIT_GENERIC

        await taxonomy_store.apply_plan(session, plan)
        await session.commit()
    print("✅ 갈래를 확정했습니다.")
    return EXIT_OK
