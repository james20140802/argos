"""사건 묶음 판정 128개로 경계 품질을 잰다 — ARG-294.

로컬 DB를 **읽기만** 한다(세션은 끝에 rollback). 판정은 Claude가 내린
잠정 정답(미검증)이다. 낮(backfill 미리보기)·밤(재군집) 두 경로를 같은 자로
잰다.

    uv run python scripts/eval_event_judgments.py [--mode day|night|both]
        [--join-threshold F] [--same-source-penalty F] [--same-source-gap-days F]
        [--sweep 0,0.02,0.03:0.5,...] [--sweep-mismatch 0,0.05:3,...] [--fill-names]

`--fill-names`는 DB의 이름 링크 대신 크롤 날짜 배치로 그 자리에서 추출한 이름으로
잰다(ARG-301). DB에는 쓰지 않는다.

`--pair-check`는 밤 분할에 애매한 짝 LLM 확인(ARG-307)을 반영해 다시 재고, 확인 전후로
판정 밖에서 갈리거나 묶인 짝을 찍는다. **실제 Ollama를 부르므로** 다른 모델이 VRAM을
쓰는 동안에는 돌리지 않는다. 기본 실행(옵션 없음)은 LLM을 부르지 않는다.

판정은 고르기 몫(홀수 번 묶음)과 채점 전용 몫(짝수 번 묶음)으로 나눠 따로 찍는다
(ARG-299). 값은 고르기 몫만 보고 고르고, 채점 전용 몫은 그 값을 확인하는 데만 쓴다.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import uuid
from collections import Counter
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import text

from argos.brain import event_backfill, event_candidates, recluster_input
from argos.brain.entity_frequency import common_names
from argos.brain.entity_refill import extract_by_crawl_date, load_refill_rows
from argos.brain.entity_store import canonical_names
from argos.brain.event_eval import (
    UNVERIFIED,
    evaluate,
    format_report,
    format_split_line,
    load_judgments,
    platform_pairs_together,
    split_for_tuning,
)
from argos.brain.recluster_core import detect_communities
from argos.brain.recluster_input import fetch_period_input
from argos.config import settings
from argos.database import AsyncSessionLocal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGMENTS = REPO_ROOT / "evals" / "event_judgments_arg283.json"

# 이름을 읽는 모듈들 — `from ... import names_for_documents`로 이름을 붙들고 있어
# 각 모듈 속성을 바꿔야 한다.
_NAME_READERS = (event_backfill, recluster_input, event_candidates)


@contextlib.contextmanager
def _memory_names(names: dict[uuid.UUID, frozenset[str]]):
    async def lookup(session, ids):
        return {i: names[i] for i in ids if i in names}

    originals = [(m, m.names_for_documents) for m in _NAME_READERS]
    for module in _NAME_READERS:
        module.names_for_documents = lookup
    try:
        yield
    finally:
        for module, original in originals:
            module.names_for_documents = original


def _sweep_header(must_keep: tuple[int, ...]) -> str:
    # 반드시 유지 열은 판정 파일의 must_keep에서 만든다 — 재매핑한 파일의 번호가
    # 바뀌어도 표가 엉뚱한 묶음을 가리키지 않게.
    keep_columns = "".join(f"| #{n} " for n in must_keep)
    return (
        "penalty:gap | 모드  | 틀린 비율(애매 제외) | 애매 남음 | 엄격 유지 | 느슨 유지 "
        f"{keep_columns}| 2건+ 그룹 | 플랫폼 쌍 | 고르기 몫 틀린/엄격 "
        f"| 채점 전용 몫 틀린/엄격  ({UNVERIFIED})"
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
    parser.add_argument("--same-source-gap-days", type=float, default=None)
    parser.add_argument(
        "--sweep",
        default=None,
        help=(
            "쉼표로 구분한 same_source_penalty[:same_source_gap_days] 목록 — 각 값으로 "
            "낮·밤을 돌려 한 줄 요약 표를 찍는다. gap을 빼면 설정값(또는 "
            "--same-source-gap-days)을 쓴다"
        ),
    )
    parser.add_argument("--entity-mismatch-penalty", type=float, default=None)
    parser.add_argument("--entity-mismatch-min-names", type=int, default=None)
    parser.add_argument(
        "--sweep-mismatch",
        default=None,
        help=(
            "쉼표로 구분한 entity_mismatch_penalty[:entity_mismatch_min_names] 목록 — "
            "--sweep과 같은 표를 찍는다(라벨 m{penalty}:{min}). --sweep과 함께 쓸 수 없다"
        ),
    )
    parser.add_argument("--entity-common-doc-ratio", type=float, default=None)
    parser.add_argument("--entity-common-min-documents", type=int, default=None)
    parser.add_argument(
        "--fill-names",
        action="store_true",
        help="DB 이름 링크 대신 크롤 날짜 배치로 메모리에서 추출한 이름으로 잰다(DB 쓰기 없음)",
    )
    parser.add_argument(
        "--pair-check",
        action="store_true",
        help=(
            "밤 분할을 애매한 짝 LLM 확인(ARG-307)을 반영해 한 번 더 재고, 확인 전후로 "
            "판정 밖 짝이 어떻게 달라졌는지 찍는다. 실제 Ollama를 부른다(캐시 재사용). "
            "DB는 읽기만 한다."
        ),
    )
    parser.add_argument("--pair-check-upper", type=float, default=None)
    parser.add_argument(
        "--pair-check-cross-source",
        action="store_true",
        help="같은 사이트 짝만이 아니라 다른 사이트 짝도 묻는다(#21 위험 비교용)",
    )
    parser.add_argument("--pair-check-model", default=None)
    parser.add_argument(
        "--pair-check-max-calls",
        type=int,
        default=100000,
        help="이번 측정의 LLM 호출 상한(기본: 사실상 무제한)",
    )
    parser.add_argument(
        "--pair-check-cache",
        type=Path,
        default=None,
        help="판단 캐시 JSONL 경로(기본: 운영과 같은 XDG state 파일)",
    )
    args = parser.parse_args()
    pair_check_options = (
        args.pair_check_upper is not None
        or args.pair_check_cross_source
        or args.pair_check_model is not None
        or args.pair_check_cache is not None
    )
    if pair_check_options and not args.pair_check:
        parser.error("--pair-check-* 옵션은 --pair-check와 함께만 쓸 수 있다")
    if args.pair_check and (args.sweep is not None or args.sweep_mismatch is not None):
        parser.error("--pair-check는 스윕과 함께 쓸 수 없다")
    if args.pair_check and args.mode == "day":
        parser.error("--pair-check는 밤 분할에만 적용된다 (--mode night|both)")
    # 흔한 이름 필터 옵션은 메모리 채움에서만 읽는다. DB 이름 모드는 운영 설정
    # (names_for_documents)을 그대로 쓰므로, 조용히 무시하느니 거부한다.
    if not args.fill_names and (
        args.entity_common_doc_ratio is not None
        or args.entity_common_min_documents is not None
    ):
        parser.error("--entity-common-* 옵션은 --fill-names와 함께만 쓸 수 있다")
    if args.sweep is not None and args.same_source_penalty is not None:
        parser.error("--sweep과 --same-source-penalty는 함께 쓸 수 없다 (스윕이 값을 직접 정한다)")
    if args.sweep is not None and args.sweep_mismatch is not None:
        parser.error("--sweep과 --sweep-mismatch는 함께 쓸 수 없다")
    if args.sweep_mismatch is not None and args.entity_mismatch_penalty is not None:
        parser.error("--sweep-mismatch와 --entity-mismatch-penalty는 함께 쓸 수 없다")
    return args


def _mismatch_points(spec: str) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for item in (v.strip() for v in spec.split(",")):
        if not item:
            continue
        penalty, _, min_names = item.partition(":")
        point: dict[str, float] = {"entity_mismatch_penalty": float(penalty)}
        if min_names:
            point["entity_mismatch_min_names"] = int(min_names)
        points.append(point)
    return points


def _sweep_points(spec: str) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for item in (v.strip() for v in spec.split(",")):
        if not item:
            continue
        penalty, _, gap = item.partition(":")
        point = {"same_source_penalty": float(penalty)}
        if gap:
            point["same_source_gap_days"] = float(gap)
        points.append(point)
    return points


def _overrides(args: argparse.Namespace) -> dict[str, float]:
    # 넘긴 인자만 덮어쓴다 — 이후 보정 노브도 이 딕셔너리에 키만 더하면 된다.
    overrides: dict[str, float] = {}
    if args.join_threshold is not None:
        overrides["join_threshold"] = args.join_threshold
    if args.same_source_penalty is not None:
        overrides["same_source_penalty"] = args.same_source_penalty
    if args.same_source_gap_days is not None:
        overrides["same_source_gap_days"] = args.same_source_gap_days
    if args.entity_mismatch_penalty is not None:
        overrides["entity_mismatch_penalty"] = args.entity_mismatch_penalty
    if args.entity_mismatch_min_names is not None:
        overrides["entity_mismatch_min_names"] = args.entity_mismatch_min_names
    if args.entity_common_doc_ratio is not None:
        overrides["entity_common_doc_ratio"] = args.entity_common_doc_ratio
    if args.entity_common_min_documents is not None:
        overrides["entity_common_min_documents"] = args.entity_common_min_documents
    if args.pair_check_upper is not None:
        overrides["pair_check_upper"] = args.pair_check_upper
    if args.pair_check_cross_source:
        overrides["pair_check_same_source_only"] = False
    if args.pair_check_model is not None:
        overrides["pair_check_model"] = args.pair_check_model
    return overrides


def _with_overrides(cfg, overrides: dict[str, float]):
    # model_copy(update=...)는 검증을 건너뛴다 — penalty 2·nan 같은 값이 그대로
    # 들어가 운영에서 불가능한 설정을 재게 된다. 다시 검증해서 만든다.
    try:
        return type(cfg).model_validate({**cfg.model_dump(), **overrides})
    except ValidationError as exc:
        raise SystemExit(f"잘못된 보정 값 {overrides}: {exc}") from exc


def _display_path(path: Path) -> str:
    # 출력(README에 붙여넣는다)에 작업 디렉터리 절대 경로가 남지 않게 레포 기준 상대 경로로.
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _platform_line(count: int) -> str:
    return f"  플랫폼 도메인(arxiv.org·github.com) 같은 그룹 쌍: {count}"


def _night_partition(period, cfg, vetoed=frozenset()) -> dict:
    communities = detect_communities(
        period.documents, period.neighbor_pairs, config=cfg, vetoed_pairs=vetoed
    )
    return {
        member: index
        for index, community in enumerate(communities)
        for member in community.members
    }


async def _day_partition(session, docs, cfg) -> dict:
    plan = await event_backfill.plan_backfill(session, docs, config=cfg)
    return {a.doc.tech_item_id: a.event_id for a in plan.assignments}


def _wrong_cell(report) -> str:
    ratio = report.wrong_ratio
    return (
        f"{report.bad_merged}/{report.ok_merged + report.bad_merged} "
        f"({'—' if ratio is None else f'{ratio:.1%}'})"
    )


def _split_cell(report) -> str:
    return f"{_wrong_cell(report)} {report.ok_strict}/{report.ok_total}"


def _sweep_row(label: str, mode: str, judgments, partition, platform_pairs: int) -> str:
    report = evaluate(judgments, partition)
    select, holdout = split_for_tuning(judgments)
    keep_cells = "".join(
        f"{'유지' if kept else '깨짐'} | " for _, kept in report.must_keep
    )
    return (
        f"{label:<12}| {mode:<5}| {_wrong_cell(report)} | "
        f"{report.maybe_merged}/{report.maybe_total} | "
        f"{report.ok_strict}/{report.ok_total} | {report.ok_merged}/{report.ok_total} | "
        f"{keep_cells}{report.multi_doc_groups} | {platform_pairs} | "
        f"{_split_cell(evaluate(select, partition))} | "
        f"{_split_cell(evaluate(holdout, partition))}"
    )


def _print_report(judgments, partition, *, title: str, platform_pairs: int) -> None:
    select, holdout = split_for_tuning(judgments)
    print()
    print(format_report(evaluate(judgments, partition), title=title))
    print(_platform_line(platform_pairs))
    print(format_split_line(evaluate(select, partition), label="고르기 몫(홀수 번 묶음)"))
    print(
        format_split_line(evaluate(holdout, partition), label="채점 전용 몫(짝수 번 묶음)")
    )


def _judged_label_of_pair(judgments, left, right) -> str | None:
    """두 문서가 같은 판정 묶음 안에 있으면 그 묶음의 라벨, 아니면 None(판정 밖)."""
    for cluster in judgments.clusters:
        if left in cluster.doc_ids and right in cluster.doc_ids:
            return cluster.label
    return None


async def _report_pair_check(
    session, period, cfg, args, judgments, before, night_sources
) -> None:
    """애매한 짝 LLM 확인을 반영한 밤 분할을 재고, 전후 판정 밖 변화를 찍는다 (ARG-307)."""
    from argos.brain.event_eval import diff_partitions
    from argos.brain.pair_check import (
        PairVerdictCache,
        check_ambiguous_pairs,
        default_cache_path,
        fetch_pair_texts,
    )
    from argos.brain.recluster_core import score_pairs

    cache_path = args.pair_check_cache or default_cache_path()
    print()
    print(
        f"LLM 확인: model={cfg.pair_check_model} 구간=[{cfg.join_threshold}, "
        f"{cfg.pair_check_upper}) 같은 사이트만={cfg.pair_check_same_source_only} "
        f"캐시={_display_path(cache_path)}"
    )
    report = await check_ambiguous_pairs(
        session,
        period.documents,
        score_pairs(period.documents, period.neighbor_pairs, config=cfg),
        config=cfg,
        cache=PairVerdictCache.load(cache_path),
        max_calls=args.pair_check_max_calls,
    )
    different = [v for v in report.verdicts if v.verdict == "DIFFERENT"]
    skipped = f"건너뜀 {report.skipped}" + (
        f"({report.skipped_reason})" if report.skipped_reason else ""
    )
    print(
        f"  구간 안 짝 {report.band_pairs} · 물음 {report.asked} · 재사용 "
        f"{report.cache_hits} · {skipped} · 소요 {report.elapsed_s:.1f}초 · "
        f"다른 발표로 거부 {len(different)}"
    )
    tally: Counter[str] = Counter(
        _judged_label_of_pair(judgments, v.left_id, v.right_id) or "판정 밖"
        for v in different
    )
    print(
        "  거부된 짝의 판정: "
        f"틀린 묶음 안 {tally['bad']}(맞게 가름) · 맞는 묶음 안 {tally['ok']}(잘못 가름) · "
        f"애매 안 {tally['maybe']} · 판정 밖 {tally['판정 밖']}"
    )

    after = _night_partition(period, cfg, report.vetoed)
    _print_report(
        judgments,
        after,
        title="night · 전 기간 재군집 + LLM 확인",
        platform_pairs=platform_pairs_together(after, night_sources),
    )

    judged = frozenset(
        doc_id for cluster in judgments.clusters for doc_id in cluster.doc_ids
    )
    diff = diff_partitions(before, after, judged_ids=judged)
    shown = diff.newly_split[:20]
    texts = await fetch_pair_texts(session, {i for pair in shown for i in pair})
    print()
    print(
        f"판정 밖 새로 갈린 짝: {diff.newly_split_total}건 (처음 {len(shown)}건 표시) "
        "— 표본으로 열어 맞음/틀림을 센다"
    )
    for left, right in shown:
        print(f"  - {texts.get(left, ('?',))[0]} | {texts.get(right, ('?',))[0]}")
    print(f"판정 밖 새로 묶인 짝: {diff.newly_joined_total}건 (LLM 확인은 묶지 않으므로 0이어야 한다)")
    for left, right in diff.newly_joined[:20]:
        print(f"  - {left} | {right}")


async def main() -> None:
    args = _parse_args()
    cfg = _with_overrides(settings.user.event_detection, _overrides(args))
    judgments = load_judgments(args.judgments)
    sweep = (
        [
            (
                f"{point['same_source_penalty']:g}:"
                f"{point.get('same_source_gap_days', cfg.same_source_gap_days):g}",
                _with_overrides(cfg, point),
            )
            for point in _sweep_points(args.sweep)
        ]
        if args.sweep
        else None
    )
    if args.sweep_mismatch:
        sweep = [
            (
                f"m{point['entity_mismatch_penalty']:g}:"
                f"{point.get('entity_mismatch_min_names', cfg.entity_mismatch_min_names)}",
                _with_overrides(cfg, point),
            )
            for point in _mismatch_points(args.sweep_mismatch)
        ]

    async with AsyncSessionLocal() as session, contextlib.AsyncExitStack() as stack:
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
            f"same_source_penalty={cfg.same_source_penalty} "
            f"same_source_gap_days={cfg.same_source_gap_days} "
            f"entity_mismatch_penalty={cfg.entity_mismatch_penalty} "
            f"entity_mismatch_min_names={cfg.entity_mismatch_min_names}"
        )
        print(f"코퍼스 문서 수: {span.total}")
        if not span.total:
            # 빈 코퍼스면 기간이 None이라 재군집 조회가 알 수 없는 오류로 죽는다.
            print("평가할 문서가 없다 — tech_items가 비어 있다.")
            await session.rollback()
            return

        names_note = "이름: DB 저장분"
        if args.fill_names:
            extracted = extract_by_crawl_date(await load_refill_rows(session))
            raw_memory = {
                item_id: frozenset(canonical_names(found))
                for item_id, found in extracted.items()
            }
            # 운영의 names_for_documents와 같은 필터를 메모리 사전 전체의 빈도로
            # 건다 — 평가가 운영과 다른 이름을 보면 숫자가 의미가 없다 (ARG-302).
            frequency: Counter[str] = Counter(
                key for names in raw_memory.values() for key in names
            )
            named_documents = sum(1 for names in raw_memory.values() if names)
            dropped_keys = common_names(
                document_frequency=frequency,
                named_documents=named_documents,
                max_ratio=cfg.entity_common_doc_ratio,
                min_documents=cfg.entity_common_min_documents,
            )
            memory = {
                item_id: names - dropped_keys for item_id, names in raw_memory.items()
            }
            dropped = sorted(dropped_keys)
            named = [v for v in memory.values() if v]
            average = sum(len(v) for v in named) / len(named) if named else 0.0
            names_note = (
                f"이름: 메모리 채움(문서 {len(memory)}건 중 {len(named)}건에 이름, "
                f"문서당 평균 {average:.1f}개), 흔한 이름 {len(dropped)}종 제거"
                f"(비율>{cfg.entity_common_doc_ratio:g}; 예: {', '.join(dropped[:8]) or '없음'})"
                " — DB 쓰기 없음"
            )
            stack.enter_context(_memory_names(memory))
        print(names_note)

        docs = None
        day_sources: dict = {}
        if args.mode in ("day", "both"):
            assigned = (
                await session.execute(text("SELECT count(*) FROM event_documents"))
            ).scalar_one()
            if assigned:
                # 낮 분할은 "아직 배정 안 된 문서"의 backfill 미리보기다. 실제 backfill
                # 뒤에는 판정 문서 대부분이 빠져 홀로 있는 것으로 세이므로, 틀린 비율이
                # 거짓으로 좋아 보인다. 경고만 하고 숫자를 내면 그 숫자가 인용된다.
                print(
                    f"낮 모드 건너뜀: event_documents {assigned}건 — 이미 배정된 문서가 "
                    "낮 분할에서 빠져 숫자가 왜곡된다. 사건이 없는 코퍼스에서만 잰다."
                )
            else:
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
            print(_sweep_header(judgments.must_keep))
            for label, run_cfg in sweep:
                if docs is not None:
                    partition = await _day_partition(session, docs, run_cfg)
                    print(
                        _sweep_row(
                            label,
                            "day",
                            judgments,
                            partition,
                            platform_pairs_together(partition, day_sources),
                        ),
                        flush=True,
                    )
                if period is not None:
                    partition = _night_partition(period, run_cfg)
                    print(
                        _sweep_row(
                            label,
                            "night",
                            judgments,
                            partition,
                            platform_pairs_together(partition, night_sources),
                        ),
                        flush=True,
                    )
        else:
            if docs is not None:
                partition = await _day_partition(session, docs, cfg)
                _print_report(
                    judgments,
                    partition,
                    title="day · backfill 미리보기",
                    platform_pairs=platform_pairs_together(partition, day_sources),
                )
            if period is not None:
                partition = _night_partition(period, cfg)
                _print_report(
                    judgments,
                    partition,
                    title="night · 전 기간 재군집",
                    platform_pairs=platform_pairs_together(partition, night_sources),
                )
                if args.pair_check:
                    await _report_pair_check(
                        session, period, cfg, args, judgments, partition, night_sources
                    )

        await session.rollback()


if __name__ == "__main__":
    asyncio.run(main())
