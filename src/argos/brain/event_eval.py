"""사건 묶음 판정으로 경계 품질을 잰다 — ARG-294. DB도 LLM도 쓰지 않는다.

입력은 판정 파일(묶음 → 문서 id + 맞음/애매/틀림)과 분할(문서 id → 그룹 키)
뿐이다. 분할이 낮 배정(backfill 미리보기)에서 왔는지 밤 재군집에서 왔는지는
모른다 — 그래야 두 경로를 같은 자로 잴 수 있다.

**판정은 Claude가 내린 잠정 정답이다(미검증).** 사람이 확인하지 않았으므로
모든 출력에 그 사실을 붙인다(ARG-284 사용자 확정 답변 2).

**지표 정의:**
- 묶음이 "아직 묶임" = 그 묶음 문서 중 2건 이상이 같은 그룹에 있다. 틀린
  묶음은 서로 다른 소식의 모음이라, 둘만 남아도 여전히 오병합이다.
- 맞는 묶음 "엄격 유지" = 구성원 전부가 한 그룹. "느슨 유지" = 아직 묶임.
- 틀린 비율 = 아직 묶인 틀림 / (아직 묶인 맞음 + 아직 묶인 틀림). 애매는
  비율에서 빼고 개수만 따로 본다(사용자 확정 답변 3).
- 분할에 없는 문서는 각자 홀로 있는 것으로 친다 — 빠진 문서끼리 묶였다고
  보면 없는 근거로 묶음을 지어내는 셈이다.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Literal, Mapping

Label = Literal["ok", "maybe", "bad"]
UNVERIFIED = "Claude 판정 기준(미검증)"


@dataclass(frozen=True)
class JudgedCluster:
    n: int
    label: Label
    doc_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True)
class JudgmentSet:
    clusters: tuple[JudgedCluster, ...]
    must_keep: tuple[int, ...]


@dataclass(frozen=True)
class EvalReport:
    ok_total: int
    maybe_total: int
    bad_total: int
    ok_merged: int
    maybe_merged: int
    bad_merged: int
    ok_strict: int
    missing_docs: int
    multi_doc_groups: int
    must_keep: tuple[tuple[int, bool], ...]

    @property
    def wrong_ratio(self) -> float | None:
        denominator = self.ok_merged + self.bad_merged
        return self.bad_merged / denominator if denominator else None

    @property
    def wrong_ratio_maybe_as_wrong(self) -> float | None:
        denominator = self.ok_merged + self.maybe_merged + self.bad_merged
        if not denominator:
            return None
        return (self.bad_merged + self.maybe_merged) / denominator


def load_judgments(path: str | Path) -> JudgmentSet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    clusters = tuple(
        JudgedCluster(
            n=int(item["n"]),
            label=item["label"],
            doc_ids=tuple(uuid.UUID(value) for value in item["doc_ids"]),
        )
        for item in raw["clusters"]
    )
    return JudgmentSet(clusters=clusters, must_keep=tuple(int(n) for n in raw.get("must_keep", ())))


def _groups_of(cluster: JudgedCluster, partition: Mapping[uuid.UUID, Hashable]) -> Counter:
    # 분할에 없는 문서는 자기 id를 그룹 키로 — 홀로 있는 것으로 센다.
    return Counter(
        ("in", partition[doc_id]) if doc_id in partition else ("missing", doc_id)
        for doc_id in cluster.doc_ids
    )


def evaluate(judgments: JudgmentSet, partition: Mapping[uuid.UUID, Hashable]) -> EvalReport:
    totals: Counter = Counter()
    merged: Counter = Counter()
    ok_strict = 0
    strict_by_n: dict[int, bool] = {}
    missing = 0
    for cluster in judgments.clusters:
        totals[cluster.label] += 1
        missing += sum(1 for doc_id in cluster.doc_ids if doc_id not in partition)
        groups = _groups_of(cluster, partition)
        if max(groups.values()) >= 2:
            merged[cluster.label] += 1
        strict = len(groups) == 1
        strict_by_n[cluster.n] = strict
        if cluster.label == "ok" and strict:
            ok_strict += 1
    sizes = Counter(partition.values())
    return EvalReport(
        ok_total=totals["ok"],
        maybe_total=totals["maybe"],
        bad_total=totals["bad"],
        ok_merged=merged["ok"],
        maybe_merged=merged["maybe"],
        bad_merged=merged["bad"],
        ok_strict=ok_strict,
        missing_docs=missing,
        multi_doc_groups=sum(1 for size in sizes.values() if size >= 2),
        must_keep=tuple((n, strict_by_n.get(n, False)) for n in judgments.must_keep),
    )


def _ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return f"{numerator}/{denominator} (—)"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def format_report(report: EvalReport, *, title: str) -> str:
    keep = ", ".join(f"#{n} {'유지' if kept else '깨짐'}" for n, kept in report.must_keep)
    lines = [
        f"[{title}] — {UNVERIFIED}",
        f"  틀린 비율(애매 제외): {_ratio(report.bad_merged, report.ok_merged + report.bad_merged)}",
        f"  애매 묶음 남음: {report.maybe_merged}/{report.maybe_total}",
        "  보조 — 애매=틀림 비율: "
        + _ratio(
            report.bad_merged + report.maybe_merged,
            report.ok_merged + report.maybe_merged + report.bad_merged,
        ),
        f"  맞는 묶음 엄격 유지: {_ratio(report.ok_strict, report.ok_total)}",
        f"  보조 — 맞는 묶음 느슨 유지(2건 이상 함께): {_ratio(report.ok_merged, report.ok_total)}",
        f"  반드시 유지: {keep or '없음'}",
        f"  전체 2건 이상 그룹 수: {report.multi_doc_groups}",
        f"  분할에 없는 판정 문서: {report.missing_docs}",
    ]
    return "\n".join(lines)
