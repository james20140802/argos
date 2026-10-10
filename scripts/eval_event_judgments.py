"""사건 묶음 판정 128개로 경계 품질을 잰다 — ARG-294.

로컬 DB를 **읽기만** 한다(세션은 끝에 rollback). 판정은 Claude가 내린
잠정 정답(미검증)이다. 낮(backfill 미리보기)·밤(재군집) 두 경로를 같은 자로
잰다.

    uv run python scripts/eval_event_judgments.py [--mode day|night|both]
        [--join-threshold F] [--same-source-penalty F] [--sweep 0,0.02,...]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import text

from argos.brain import event_backfill
from argos.brain.event_eval import (
    UNVERIFIED,
    evaluate,
    format_report,
    load_judgments,
    platform_pairs_together,
)
from argos.brain.recluster_core import detect_communities
from argos.brain.recluster_input import fetch_period_input
from argos.config import settings
from argos.database import AsyncSessionLocal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGMENTS = REPO_ROOT / "evals" / "event_judgments_arg283.json"

SWEEP_HEADER = (
    "penalty| 모드  | 틀린 비율(애매 제외) | 애매 남음 | 엄격 유지 | 느슨 유지 "
    "| #21 | #25 | 2건+ 그룹 | 플랫폼 쌍  (Claude 판정 기준(미검증))"
)

_SPAN_SQL = text(
    """
    SELECT min(COALESCE(published_at, created_at)) AS start,
           max(COALESCE(published_at, created_at)) AS finish,
           count(*) AS total
    FROM tech_items
    """
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("day", "night", "both"), default="both")
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--join-threshold", type=float, default=None)
    parser.add_argument("--same-source-penalty", type=float, default=None)
    parser.add_argument(
        "--sweep",
        default=None,
        help="쉼표로 구분한 same_source_penalty 목록 — 각 값으로 낮·밤을 돌려 한 줄 요약 표를 찍는다",
    )
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict[str, float]:
    # 넘긴 인자만 덮어쓴다 — 이후 보정 노브도 이 딕셔너리에 키만 더하면 된다.
    overrides: dict[str, float] = {}
    if args.join_threshold is not None:
        overrides["join_threshold"] = args.join_threshold
    if args.same_source_penalty is not None:
        overrides["same_source_penalty"] = args.same_source_penalty
    return overrides


def _display_path(path: Path) -> str:
    # 출력(README에 붙여넣는다)에 작업 디렉터리 절대 경로가 남지 않게 레포 기준 상대 경로로.
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _platform_line(count: int) -> str:
    return f"  플랫폼 도메인(arxiv.org·github.com) 같은 그룹 쌍: {count}"


def _night_partition(period, cfg) -> dict:
    communities = detect_communities(period.documents, period.neighbor_pairs, config=cfg)
    return {
        member: index
        for index, community in enumerate(communities)
        for member in community.members
    }


async def _day_partition(session, docs, cfg) -> dict:
    plan = await event_backfill.plan_backfill(session, docs, config=cfg)
    return {a.doc.tech_item_id: a.event_id for a in plan.assignments}


def _sweep_row(penalty: float, mode: str, report, platform_pairs: int) -> str:
    keep = dict(report.must_keep)
    mark = lambda n: "유지" if keep.get(n) else "깨짐"  # noqa: E731
    ratio = report.wrong_ratio
    return (
        f"{penalty:<7g}| {mode:<5}| "
        f"{report.bad_merged}/{report.ok_merged + report.bad_merged} "
        f"({'—' if ratio is None else f'{ratio:.1%}'}) | "
        f"{report.maybe_merged}/{report.maybe_total} | "
        f"{report.ok_strict}/{report.ok_total} | {report.ok_merged}/{report.ok_total} | "
        f"{mark(21)} | {mark(25)} | {report.multi_doc_groups} | {platform_pairs}"
    )


async def main() -> None:
    args = _parse_args()
    cfg = settings.user.event_detection.model_copy(update=_overrides(args))
    judgments = load_judgments(args.judgments)
    sweep = (
        [float(value) for value in args.sweep.split(",") if value.strip()]
        if args.sweep
        else None
    )

    async with AsyncSessionLocal() as session:
        span = (await session.execute(_SPAN_SQL)).one()
        print(
            f"판정 파일: {_display_path(args.judgments)} "
            f"({len(judgments.clusters)}개 묶음) — {UNVERIFIED}"
        )
        print(
            f"설정: join_threshold={cfg.join_threshold} γ={cfg.effective_leiden_resolution} "
            f"window_days={cfg.window_days} candidate_k={cfg.candidate_k} "
            f"weights(cos/entity/time/kw)={cfg.weight_cosine}/{cfg.weight_entity}/"
            f"{cfg.weight_time}/{cfg.weight_keyword} "
            f"same_source_penalty={cfg.same_source_penalty}"
        )
        print(f"코퍼스 문서 수: {span.total}")

        docs = None
        day_sources: dict = {}
        if args.mode in ("day", "both"):
            assigned = (
                await session.execute(text("SELECT count(*) FROM event_documents"))
            ).scalar_one()
            if assigned:
                print(
                    f"경고: event_documents {assigned}건 — 이미 배정된 문서는 낮 분할에서 빠진다"
                )
            docs = await event_backfill.fetch_unassigned_documents(session)
            day_sources = {d.tech_item_id: d.features.source for d in docs}

        period = None
        night_sources: dict = {}
        if args.mode in ("night", "both"):
            # 이웃 조회에 cfg의 창·상한을 그대로 넘긴다. 이웃 후보는 보정과 무관해
            # 스윕에서도 한 번만 조회한다.
            period = await fetch_period_input(
                session,
                start=span.start,
                end=span.finish,
                window_days=cfg.window_days,
                limit=cfg.candidate_k,
            )
            night_sources = {d.tech_item_id: d.features.source for d in period.documents}

        if sweep is not None:
            print()
            print(SWEEP_HEADER)
            for penalty in sweep:
                run_cfg = cfg.model_copy(update={"same_source_penalty": penalty})
                if docs is not None:
                    partition = await _day_partition(session, docs, run_cfg)
                    print(
                        _sweep_row(
                            penalty,
                            "day",
                            evaluate(judgments, partition),
                            platform_pairs_together(partition, day_sources),
                        ),
                        flush=True,
                    )
                if period is not None:
                    partition = _night_partition(period, run_cfg)
                    print(
                        _sweep_row(
                            penalty,
                            "night",
                            evaluate(judgments, partition),
                            platform_pairs_together(partition, night_sources),
                        ),
                        flush=True,
                    )
        else:
            if docs is not None:
                partition = await _day_partition(session, docs, cfg)
                print()
                print(format_report(evaluate(judgments, partition), title="day · backfill 미리보기"))
                print(_platform_line(platform_pairs_together(partition, day_sources)))
            if period is not None:
                partition = _night_partition(period, cfg)
                print()
                print(format_report(evaluate(judgments, partition), title="night · 전 기간 재군집"))
                print(_platform_line(platform_pairs_together(partition, night_sources)))

        await session.rollback()


if __name__ == "__main__":
    asyncio.run(main())
