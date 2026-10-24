"""역할 갈래 배정 평가 — 쌍 일치도 (ARG-313).

정답 묶음(``evals/taxonomy_truth_arg313.json``)과 배정 결과를 기술 **쌍** 단위로
비교한다. 두 기술이 정답에서 같은 묶음인데 배정도 같은 갈래면 맞음(tp), 배정만
같으면 fp, 정답만 같으면 fn. 시험판(2026-10-22) ``cluster.py``의 ``pair_f1``과 같은
산식이라 시험판 숫자(D 방식 0.81)와 바로 비교할 수 있다.

미분류(``None``)와 배정 결과에 없는 이름은 **단독**으로 친다 — 미분류끼리 같은
갈래로 묶인 것으로 세면 "근거 부족으로 남긴" 결정이 점수를 부풀린다. 빠진 이름은
``missing``으로 따로 보고한다. 순수 함수만 둔다.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class PairScore:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    missing: tuple[str, ...] = ()
    unclassified: tuple[str, ...] = ()

    def line(self) -> str:
        return (
            f"쌍 일치도 F1 {self.f1:.2f} (정밀도 {self.precision:.2f} · 재현율 {self.recall:.2f}; "
            f"tp {self.tp} fp {self.fp} fn {self.fn})"
        )


def load_truth(path: Path) -> dict[str, str]:
    """정답 파일 -> {기술 이름: 정답 묶음 이름}. ``{"groups": {묶음: [이름...]}}`` 형식."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = data.get("groups", data) if isinstance(data, dict) else None
    if not isinstance(groups, dict):
        raise ValueError(f"정답 파일 형식이 아닙니다: {path}")
    out: dict[str, str] = {}
    for group, names in groups.items():
        if not isinstance(names, list):
            continue
        for name in names:
            if name in out:
                raise ValueError(f"정답에서 이름이 두 묶음에 있습니다: {name!r}")
            out[str(name)] = str(group)
    return out


def pair_f1(predicted: Mapping[str, str | None], gold: Mapping[str, str]) -> PairScore:
    """정답에 있는 이름들의 모든 쌍으로 정밀도·재현율·F1."""
    names = list(gold)
    missing = tuple(n for n in names if n not in predicted)
    unclassified = tuple(n for n in names if n in predicted and predicted[n] is None)
    tp = fp = fn = 0
    for a, b in itertools.combinations(names, 2):
        la, lb = predicted.get(a), predicted.get(b)
        same_p = la is not None and lb is not None and la == lb
        same_g = gold[a] == gold[b]
        tp += same_p and same_g
        fp += same_p and not same_g
        fn += same_g and not same_p
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return PairScore(
        precision=p,
        recall=r,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        missing=missing,
        unclassified=unclassified,
    )
