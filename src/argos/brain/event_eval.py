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

**고르기 몫과 채점 전용 몫 (ARG-299).** 보정 값을 판정에 맞춰 고르면 같은 판정으로
채점한 숫자는 낙관적으로 나온다. 그래서 묶음 번호 홀수는 값을 고르는 데 쓰고,
짝수는 채점에만 남긴다(`split_for_tuning`). 나누는 규칙이 번호뿐이라 결정적이다.
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

# 같은 도메인이 곧 같은 소식원이 아닌 플랫폼 도메인(깃허브의 수많은 저장소,
# arXiv의 수많은 논문). 같은 출처 보정(ARG-284)은 이들을 면제하지 않으므로
# 서로 다른 작성자의 글도 감점된다 — 같은 그룹에 든 쌍을 따로 세어 그 부작용을
# 눈으로 확인한다.
PLATFORM_DOMAINS = frozenset({"arxiv.org", "github.com"})


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


_VALID_LABELS = frozenset({"ok", "maybe", "bad"})


def load_judgments(path: str | Path) -> JudgmentSet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    clusters: list[JudgedCluster] = []
    for item in raw["clusters"]:
        n = int(item["n"])
        label = item["label"]
        if label not in _VALID_LABELS:
            raise ValueError(
                f"판정 파일의 클러스터 {n}: 라벨 {label!r}은 ok/maybe/bad 중 하나여야 한다"
            )
        doc_ids = tuple(uuid.UUID(value) for value in item["doc_ids"])
        if len(doc_ids) < 2:
            raise ValueError(
                f"판정 파일의 클러스터 {n}: doc_ids가 {len(doc_ids)}개 — 최소 2개가 필요하다"
            )
        clusters.append(JudgedCluster(n=n, label=label, doc_ids=doc_ids))
    must_keep = tuple(int(n) for n in raw.get("must_keep", ()))
    # 없는 번호를 반드시 유지로 적어 두면 evaluate가 매번 "깨짐"으로 찍는다 —
    # 재매핑으로 번호가 바뀐 파일에서 원인을 찾기 어려우니 읽을 때 막는다.
    unknown = sorted(set(must_keep) - {cluster.n for cluster in clusters})
    if unknown:
        raise ValueError(f"판정 파일의 must_keep {unknown}: 그런 번호의 클러스터가 없다")
    return JudgmentSet(clusters=tuple(clusters), must_keep=must_keep)


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


def split_for_tuning(judgments: JudgmentSet) -> tuple[JudgmentSet, JudgmentSet]:
    """(고르기 몫 = 홀수 번 묶음, 채점 전용 몫 = 짝수 번 묶음).

    반드시 유지(`must_keep`)는 양쪽 다 비운다 — 그 판정은 몫과 상관없이 전체
    판정 리포트에서 매번 확인한다. 몫에 남기면 반대쪽 몫에서 "없는 번호 = 깨짐"
    으로 찍힌다.
    """
    select = tuple(c for c in judgments.clusters if c.n % 2 == 1)
    holdout = tuple(c for c in judgments.clusters if c.n % 2 == 0)
    return JudgmentSet(clusters=select, must_keep=()), JudgmentSet(
        clusters=holdout, must_keep=()
    )


def format_split_line(report: EvalReport, *, label: str) -> str:
    return (
        f"  {label}: 틀린 비율(애매 제외) "
        f"{_ratio(report.bad_merged, report.ok_merged + report.bad_merged)}, "
        f"맞는 묶음 엄격 유지 {_ratio(report.ok_strict, report.ok_total)}"
    )


def platform_pairs_together(
    partition: Mapping[uuid.UUID, Hashable],
    sources: Mapping[uuid.UUID, str | None],
    *,
    platforms: frozenset[str] = PLATFORM_DOMAINS,
) -> int:
    """같은 그룹에 든 같은 플랫폼 도메인 문서 쌍의 수.

    그룹별·플랫폼 도메인별 개수 m마다 m*(m-1)/2를 더한다. 출처를 모르는
    문서(None·sources에 없음)는 세지 않는다.
    """
    per_group: Counter = Counter(
        (group, sources[doc_id])
        for doc_id, group in partition.items()
        if sources.get(doc_id) in platforms
    )
    return sum(m * (m - 1) // 2 for m in per_group.values())


@dataclass(frozen=True)
class PartitionDiff:
    """두 분할 사이에서 짝 단위로 달라진 것 중 **판정 밖** 짝 (ARG-307).

    `newly_split`은 전에는 같은 그룹이었는데 이제 갈린 짝, `newly_joined`는 그 반대다.
    목록은 `limit`건까지 id 오름차순으로 자르고, 개수는 전체를 센다.
    """

    newly_split: tuple[tuple[uuid.UUID, uuid.UUID], ...]
    newly_split_total: int
    newly_joined: tuple[tuple[uuid.UUID, uuid.UUID], ...]
    newly_joined_total: int


def _cross_pairs(
    reference: Mapping[uuid.UUID, Hashable],
    other: Mapping[uuid.UUID, Hashable],
    shared: frozenset[uuid.UUID],
):
    """`reference`에서 같은 그룹인데 `other`에서는 다른 그룹인 짝 (정렬된 id 순)."""
    groups: dict[Hashable, list[uuid.UUID]] = {}
    for doc_id in sorted(shared):
        groups.setdefault(reference[doc_id], []).append(doc_id)
    for members in groups.values():
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                if other[left] != other[right]:
                    yield (left, right)


def diff_partitions(
    before: Mapping[uuid.UUID, Hashable],
    after: Mapping[uuid.UUID, Hashable],
    *,
    judged_ids: frozenset[uuid.UUID],
    limit: int = 50,
) -> PartitionDiff:
    """전후 분할에서 달라진 짝 중 판정 파일 밖의 것. 두 문서가 모두 판정 문서인 짝은 뺀다.

    판정 안 짝은 틀린 비율·엄격 유지가 이미 재고 있다. 이 목록은 그 숫자가 못 보는
    "판정 밖에서 무엇이 바뀌었나"를 사람이 표본으로 열어 보게 하려는 것이다.
    두 분할에 모두 있는 문서만 비교한다.
    """
    shared = frozenset(before) & frozenset(after)

    def outside(pairs):
        return sorted(
            pair for pair in pairs
            if not (pair[0] in judged_ids and pair[1] in judged_ids)
        )

    split = outside(_cross_pairs(before, after, shared))
    joined = outside(_cross_pairs(after, before, shared))
    return PartitionDiff(
        newly_split=tuple(split[:limit]),
        newly_split_total=len(split),
        newly_joined=tuple(joined[:limit]),
        newly_joined_total=len(joined),
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
