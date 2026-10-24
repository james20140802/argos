"""``argos taxonomy`` — 역할 갈래 초안·편집·확정 명령 (ARG-311).

흐름: ``draft``(파일) -> ``rename/merge/split/delete``(파일 위 편집) -> ``apply``
(현재 확정본과 차이 미리보기 -> y/N -> DB). 확정 후 다시 고칠 때도 같은 길이다:
``export``(DB -> 파일) -> 편집 -> ``apply``.

갈래 확정본(``role_branches``)에 **쓰는** 명령은 ``apply``뿐이고 ``apply``는 확인
없이는 아무것도 쓰지 않는다. 이 모듈이 ``taxonomy_store.apply_plan``을 부르는 유일한
곳이어야 한다(``tests/brain/test_taxonomy_isolation.py``).

ARG-313: ``assign``은 확정된 갈래를 **읽기만** 하고 기술 -> 아래 갈래 배정
(``technology_profiles``·``role_assignments``)을 쓴다. 계획(LLM 포함) -> 미리보기 ->
y/N -> 같은 계획 기록. 하루 한 번 launchd(``com.argos.classify``)가 ``--yes``로 돈다.
``why``는 배정 근거를, ``eval``은 정답 묶음 대비 쌍 일치도를 보여 준다(둘 다 읽기 전용).
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
DEFAULT_TRUTH = Path("evals") / "taxonomy_truth_arg313.json"
DEFAULT_SPIKE_TREE = Path("evals") / "taxonomy_spike_treeA.json"


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

    asg = actions.add_parser(
        "assign",
        help="새 기술을 확정된 아래 갈래에 자동 배정 (미리보기 -> y/N -> 기록)",
    )
    asg.add_argument("--all", action="store_true", help="기존 기술 전체를 다시 배정 (소급)")
    asg.add_argument(
        "--limit", type=_positive_int, default=None, metavar="N",
        help="이번에 볼 이름 수 (기본: taxonomy.max_per_run, --all이면 제한 없음)",
    )
    asg.add_argument(
        "--name", action="append", default=[], metavar="NAME",
        help="이 이름(들)만 다시 배정 (반복 가능; 별칭은 대표로). 예: 평가 세트만 배정",
    )
    asg.add_argument(
        "--names-from", type=Path, default=None, metavar="TRUTH_JSON",
        help="정답 파일(groups)의 이름들만 다시 배정",
    )
    asg.add_argument("--dry-run", action="store_true", help="미리보기만 보이고 종료 (DB 미기록)")
    asg.add_argument("--yes", action="store_true", help="확인 질문 생략 (스케줄 실행용)")
    asg.add_argument(
        "--sample", type=_positive_int, default=None, metavar="N",
        help="미리보기에서 묶음마다 N개만 보이기",
    )

    why = actions.add_parser("why", help="기술 하나의 설명·갈래·배정 근거")
    why.add_argument("name")

    ev = actions.add_parser("eval", help="정답 묶음 대비 배정 정확도(쌍 일치도) — 읽기 전용")
    ev.add_argument(
        "--truth", type=Path, default=DEFAULT_TRUTH, help=f"정답 파일 (기본: {DEFAULT_TRUTH})"
    )
    ev.add_argument(
        "--tree", choices=("confirmed", "spike"), default="confirmed",
        help="confirmed: DB의 배정 결과로 채점(AC 판정). spike: 시험판 고정 트리로 "
        "정답 이름만 새로 배정해 채점(참고값, LLM 사용, DB 미기록)",
    )
    ev.add_argument(
        "--spike-tree", type=Path, default=DEFAULT_SPIKE_TREE, help="--tree spike의 트리 파일"
    )
    ev.add_argument(
        "--assignments", type=Path, default=None,
        help="{이름: 갈래} JSON을 채점 (DB 없음, 시험판 결과 재현용)",
    )


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
        if action == "assign":
            return asyncio.run(_assign(args))
        if action == "why":
            return asyncio.run(_why(args))
        if action == "eval":
            return asyncio.run(_eval(args))
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


# --------------------------------------------------------------------- #
# ARG-313: 자동 배정 · 근거 · 평가
# --------------------------------------------------------------------- #
_NO_TABLES = "❌ 배정 테이블이 없습니다 — `uv run alembic upgrade head`를 먼저 실행하세요."


def _size_counts(current: dict[str, str | None], plan, inputs) -> dict[str, int]:
    """이번 계획을 반영한 뒤의 아래 갈래별 기술 수."""
    after = dict(current)
    for a in plan.assignments:
        after[a.key] = a.leaf_path
    key_of = {t.entity_id: t.key for t in inputs}
    for eid, _display in plan.unassign:
        after.pop(key_of.get(eid, ""), None)
    counts: dict[str, int] = {}
    for path in after.values():
        if path:
            counts[path] = counts.get(path, 0) + 1
    return counts


async def _make_plan(tree: Tree, inputs, current):
    """LLM 배정 계획. 다른 모델을 먼저 내리고 끝에 배정 모델도 내린다(한 번에 한 LLM)."""
    from argos.brain.llm_client import get_taxonomy_llm_client
    from argos.brain.taxonomy_draft import _unload_resident_models
    from argos.brain.tech_classify import build_plan, leaves_of
    from argos.config import settings

    cfg = settings.user.taxonomy
    llm = get_taxonomy_llm_client()
    await _unload_resident_models()
    try:
        return await build_plan(
            inputs, leaves_of(tree), llm, cfg, current=current, model_name=cfg.model
        )
    finally:
        try:
            await llm.unload("large")
        except Exception:  # noqa: BLE001 — 언로드 실패가 결과를 가리면 안 된다
            pass


async def _collect_inputs(session, selected, rows):
    from argos.brain import tech_classify_store as store
    from argos.config import settings

    cfg = settings.user.taxonomy
    external = None
    if cfg.external_enabled:
        from argos.crawler.tech_external import ExternalEvidence

        external = ExternalEvidence(
            max_requests=cfg.external_max_requests,
            ttl_seconds=cfg.external_cache_ttl_days * 86400,
        )
    try:
        return await store.build_inputs(session, selected, cfg, all_rows=rows, external=external)
    finally:
        if external is not None:
            external.save()
            await external.aclose()


async def _assign(args: argparse.Namespace) -> int:
    from argos.brain import tech_classify_store as store
    from argos.brain.tech_classify import size_suggestions
    from argos.config import settings

    cfg = settings.user.taxonomy
    async with AsyncSessionLocal() as session:
        if not await store.tables_exist(session):
            print(_NO_TABLES)
            return EXIT_GENERIC
        tree = await taxonomy_store.load_confirmed(session)
        if tree is None:
            print(
                "확정된 갈래 없음 — 먼저 `argos taxonomy draft` → `argos taxonomy apply`로 "
                "갈래를 확정하세요. (배정할 곳이 없어 아무것도 하지 않았습니다)"
            )
            return EXIT_OK
        rows = await store.load_name_rows(session)
        names = list(args.name)
        if args.names_from is not None:
            from argos.brain.taxonomy_eval import load_truth

            names += list(load_truth(args.names_from))
        if names:
            wanted = set()
            for name in names:
                found = await store.resolve_canonical(session, name)
                if found is None:
                    print(f"⚠️  이름을 찾을 수 없습니다: {name!r}")
                else:
                    wanted.add(found[0])
            subset = [r for r in rows if r.entity_id in wanted]
            selected = store.select_targets(subset, cfg, all_names=True, limit=args.limit)
        else:
            limit = args.limit or (None if args.all else cfg.max_per_run)
            selected = store.select_targets(rows, cfg, all_names=args.all, limit=limit)
        if not selected:
            print("새로 배정할 기술 없음")
            return EXIT_OK
        print(f"대상 이름 {len(selected)}개 — 설명·배정 계산 중 (LLM, 한 번에 하나)…")
        inputs = await _collect_inputs(session, selected, rows)
        current = await store.current_assignments(session, tree)
        plan = await _make_plan(tree, inputs, current)

        for line in plan.preview_lines(sample=args.sample):
            print(line)
        for line in size_suggestions(_size_counts(current, plan, inputs), cfg):
            print(f"💡 {line}")
        print(f"(LLM 호출 {plan.llm_calls}회)")
        if plan.is_empty:
            print("변경 없음")
            return EXIT_OK
        if args.dry_run:
            print("--dry-run: DB에는 아무것도 쓰지 않았습니다.")
            return EXIT_OK
        if not args.yes:
            reply = input("이대로 기록할까요? [y/N] ").strip().lower()
            if reply not in ("y", "yes"):
                print("취소되었습니다.")
                return EXIT_GENERIC
        n = await store.write_plan(session, plan)
        await session.commit()
    print(f"✅ 배정을 기록했습니다 ({n}행, run {plan.run_id}).")
    return EXIT_OK


async def _why(args: argparse.Namespace) -> int:
    from argos.brain import tech_classify_store as store

    async with AsyncSessionLocal() as session:
        if not await store.tables_exist(session):
            print(_NO_TABLES)
            return EXIT_GENERIC
        tree = await taxonomy_store.load_confirmed(session)
        exp = await store.explain(session, args.name, tree)
    if exp is None:
        print(f"이름을 찾을 수 없습니다: {args.name!r}")
        return EXIT_GENERIC
    print(f"■ {exp.name}")
    p, a = exp.profile, exp.assignment
    if p is None:
        print("  아직 설명·배정 전입니다 (다음 `argos taxonomy assign`에서 다룹니다).")
        return EXIT_OK
    verdict = getattr(p.verdict, "value", p.verdict)
    print(f"  판정: {verdict}" + (f" ({p.not_tech_reason})" if p.not_tech_reason else ""))
    if p.kind:
        print(f"  종류: {p.kind}")
    if p.description:
        print(f"  설명: {p.description}")
    model = f" · 모델 {p.model}" if p.model else ""
    print(f"  설명에 쓴 문서 수: {p.described_doc_count}{model}")
    if a is None:
        print("  갈래: (배정 없음)")
        return EXIT_OK
    source = getattr(a.source, "value", a.source)
    if exp.leaf_path:
        print(f"  갈래: {exp.leaf_path}  [{source}]")
    else:
        print(f"  갈래: 미분류 ({a.unclassified_reason})  [{source}]")
    ev = a.evidence or {}
    print("  근거:")
    for nb in ev.get("neighbors") or []:
        where = nb.get("branch") or "미분류"
        print(
            f"    - 함께 나열/비교된 기술: {nb.get('name')} "
            f"({nb.get('count')}회, 그 기술의 갈래: {where})"
        )
        for ex in nb.get("examples") or []:
            print(f"        “{ex}”")
    for x in ev.get("isa") or []:
        print(f'    - 기사 표현: "{exp.name} is a {x}"')
    if ev.get("external_tags"):
        print(f"    - 외부 목록 분류: {', '.join(ev['external_tags'])}")
    print(f"    - 설명에 쓴 문장 수: {ev.get('sentence_count', 0)}")
    if ev.get("llm_reason"):
        print(f"    - LLM 이유: {ev['llm_reason']}")
    return EXIT_OK


async def _eval(args: argparse.Namespace) -> int:
    import json

    from argos.brain.taxonomy_eval import load_truth, pair_f1

    if not args.truth.exists():
        print(f"❌ 정답 파일이 없습니다: {args.truth} (레포 루트에서 실행하거나 --truth로 지정)")
        return EXIT_GENERIC
    gold = load_truth(args.truth)

    if args.assignments is not None:
        predicted = json.loads(args.assignments.read_text(encoding="utf-8"))
        label = f"파일 {args.assignments}"
    elif args.tree == "confirmed":
        from argos.brain import tech_classify_store as store

        async with AsyncSessionLocal() as session:
            if not await store.tables_exist(session):
                print(_NO_TABLES)
                return EXIT_GENERIC
            predicted = await store.assigned_labels(session, gold)
            await session.rollback()
        label = "확정 트리 · DB 배정 결과"
    else:
        spike = await _eval_spike(args, gold)
        if spike is None:
            return EXIT_GENERIC
        predicted = spike
        label = f"시험판 고정 트리({args.spike_tree}) · 정답 이름만 새로 배정(DB 미기록)"

    score = pair_f1(predicted, gold)
    print(f"정답: {args.truth} ({len(gold)}개) — 기준: {label}")
    print(score.line())
    if score.unclassified:
        print(f"미분류(단독 처리) {len(score.unclassified)}개: {', '.join(score.unclassified)}")
    if score.missing:
        print(f"DB/결과에 없는 이름(단독 처리) {len(score.missing)}개: {', '.join(score.missing)}")
    groups: dict[str, list[str]] = {}
    for name in gold:
        lab = predicted.get(name)
        if lab:
            groups.setdefault(str(lab), []).append(name)
    for lab, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  [{lab}] {', '.join(names)}")
    return EXIT_OK


async def _eval_spike(
    args: argparse.Namespace, gold: dict[str, str]
) -> dict[str, str | None] | None:
    """시험판 고정 트리로 정답 이름만 배정해 본다. LLM을 쓰고 DB에는 쓰지 않는다."""
    from argos.brain import tech_classify_store as store

    tree = tx.load_file(args.spike_tree)
    async with AsyncSessionLocal() as session:
        if not await store.tables_exist(session):
            print(_NO_TABLES)
            return None
        rows = await store.load_name_rows(session)
        by_id = {r.entity_id: r for r in rows}
        wanted: dict[object, list[str]] = {}
        for name in gold:
            found = await store.resolve_canonical(session, name)
            if found is not None and found[0] in by_id:
                wanted.setdefault(found[0], []).append(name)
        selected = [(by_id[eid], by_id[eid].profile is None) for eid in wanted]
        print(f"정답 {len(gold)}개 중 DB에 있는 이름 {len(selected)}개를 시험판 트리로 배정 (LLM)…")
        inputs = await _collect_inputs(session, selected, rows)
        plan = await _make_plan(tree, inputs, {})
        await session.rollback()
    leaf_of = {a.entity_id: a.leaf_path for a in plan.assignments}
    out: dict[str, str | None] = {}
    for eid, names in wanted.items():
        for name in names:
            out[name] = leaf_of.get(eid)
    return out
