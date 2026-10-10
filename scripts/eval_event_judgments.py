"""사건 묶음 판정 128개로 경계 품질을 잰다 — ARG-294.

로컬 DB를 **읽기만** 한다(세션은 끝에 rollback). 판정은 Claude가 내린
잠정 정답(미검증)이다. 낮(backfill 미리보기)·밤(재군집) 두 경로를 같은 자로
잰다.

    uv run python scripts/eval_event_judgments.py [--mode day|night|both]
        [--join-threshold F]
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
)
from argos.brain.recluster_core import detect_communities
from argos.brain.recluster_input import fetch_period_input
from argos.config import settings
from argos.database import AsyncSessionLocal

DEFAULT_JUDGMENTS = Path(__file__).resolve().parents[1] / "evals" / "event_judgments_arg283.json"

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
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict[str, float]:
    # 넘긴 인자만 덮어쓴다 — 이후 보정 노브도 이 딕셔너리에 키만 더하면 된다.
    overrides: dict[str, float] = {}
    if args.join_threshold is not None:
        overrides["join_threshold"] = args.join_threshold
    return overrides


async def main() -> None:
    args = _parse_args()
    cfg = settings.user.event_detection.model_copy(update=_overrides(args))
    judgments = load_judgments(args.judgments)

    async with AsyncSessionLocal() as session:
        span = (await session.execute(_SPAN_SQL)).one()
        print(f"판정 파일: {args.judgments} ({len(judgments.clusters)}개 묶음) — {UNVERIFIED}")
        print(
            f"설정: join_threshold={cfg.join_threshold} γ={cfg.effective_leiden_resolution} "
            f"window_days={cfg.window_days} candidate_k={cfg.candidate_k} "
            f"weights(cos/entity/time/kw)={cfg.weight_cosine}/{cfg.weight_entity}/"
            f"{cfg.weight_time}/{cfg.weight_keyword}"
        )
        print(f"코퍼스 문서 수: {span.total}")

        if args.mode in ("day", "both"):
            assigned = (
                await session.execute(text("SELECT count(*) FROM event_documents"))
            ).scalar_one()
            if assigned:
                print(
                    f"경고: event_documents {assigned}건 — 이미 배정된 문서는 낮 분할에서 빠진다"
                )
            docs = await event_backfill.fetch_unassigned_documents(session)
            plan = await event_backfill.plan_backfill(session, docs, config=cfg)
            partition = {a.doc.tech_item_id: a.event_id for a in plan.assignments}
            print()
            print(format_report(evaluate(judgments, partition), title="day · backfill 미리보기"))

        if args.mode in ("night", "both"):
            # 이웃 조회에 cfg의 창·상한을 그대로 넘긴다.
            period = await fetch_period_input(
                session,
                start=span.start,
                end=span.finish,
                window_days=cfg.window_days,
                limit=cfg.candidate_k,
            )
            communities = detect_communities(
                period.documents, period.neighbor_pairs, config=cfg
            )
            partition = {
                member: index
                for index, community in enumerate(communities)
                for member in community.members
            }
            print()
            print(format_report(evaluate(judgments, partition), title="night · 전 기간 재군집"))

        await session.rollback()


if __name__ == "__main__":
    asyncio.run(main())
