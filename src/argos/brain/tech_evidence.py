"""LLM 없는 문장 패턴 증거 — ARG-313 (시험판 ``patterns.py`` 이식).

같은 글에 함께 나왔다는 사실만으로는 같은 역할이라고 할 수 없다(OpenRouter 기사에
Claude Code가 나온다고 둘이 같은 갈래가 아니다). 그래서 **문장 안의 모양**만 센다.

- 나열: 이름들이 쉼표·and·or·빗금만 사이에 두고 이어진 목록("Claude Code, Codex,
  and Cursor"). 한 목록 안의 모든 쌍을 1씩 센다.
- 대안: 두 이름 사이(60자 이내)에 "alternative to / vs / compared to / instead of
  / like" 같은 표현.
- is-a 구: "X is a/an/the ..." 또는 "X, a/an ...".

시험판은 이웃한 두 이름만 나열 쌍으로 셌다. 여기서는 목록 전체의 쌍을 센다
("A, B, C"에서 A–C도 같은 목록이다). 순수 함수만 둔다 — 어떤 문서·어떤 이름을
넘길지는 호출자(``tech_classify_store``) 몫이다.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from argos.services.technology_names import surface_pattern

MAX_SENTENCE_CHARS = 600
ALT_MAX_GAP = 60
MAX_EXAMPLES_PER_PAIR = 2
MAX_EXAMPLE_CHARS = 220
MAX_ISA_PER_NAME = 6

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_ALT = re.compile(
    r"alternative to|alternatives to|\bvs\.?\b|versus|compared (?:to|with)|competitor"
    r"|rival|instead of|replacement for|like",
    re.I,
)
_SEP = re.compile(r"^\s*(?:,|\band\b|\bor\b|/|,\s*and|,\s*or)\s*$", re.I)
_ISA = re.compile(r"(?:\s+is|,)\s+(?:a|an|the)\s+((?:[\w-]+\s+){0,5}[\w-]+)")

NamePatterns = Sequence[tuple[re.Pattern[str], str]]
Pair = tuple[str, str]


def name_patterns(surfaces_by_key: Mapping[str, Iterable[str]]) -> list[tuple[re.Pattern[str], str]]:
    """{대표 키: [표기...]} -> [(정규식, 대표 키)]. 별칭 표기도 대표 키로 센다."""
    out: list[tuple[re.Pattern[str], str]] = []
    for key, surfaces in surfaces_by_key.items():
        for surface in dict.fromkeys(surfaces):
            out.append((surface_pattern(surface), key))
    return out


def find_mentions(sentence: str, patterns: NamePatterns) -> list[tuple[int, int, str]]:
    """문장 안의 이름 위치 (시작, 끝, 대표 키). 더 긴 이름 안에 든 짧은 이름은 버린다."""
    raw: list[tuple[int, int, str]] = []
    for pattern, key in patterns:
        for m in pattern.finditer(sentence):
            raw.append((m.start(), m.end(), key))
    raw.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: list[tuple[int, int, str]] = []
    for start, end, key in raw:
        if kept and start < kept[-1][1]:
            continue  # 앞 이름과 겹친다(더 긴 쪽이 먼저 들어가 있다)
        kept.append((start, end, key))
    return kept


@dataclass(frozen=True)
class Neighbor:
    key: str
    count: int
    examples: tuple[str, ...]


@dataclass
class PatternEvidence:
    colist: Counter[Pair] = field(default_factory=Counter)
    alt: Counter[Pair] = field(default_factory=Counter)
    isa: dict[str, list[str]] = field(default_factory=dict)
    examples: dict[Pair, list[str]] = field(default_factory=dict)

    def _example(self, pair: Pair, sentence: str) -> None:
        bucket = self.examples.setdefault(pair, [])
        text = sentence.strip()[:MAX_EXAMPLE_CHARS]
        if len(bucket) < MAX_EXAMPLES_PER_PAIR and text not in bucket:
            bucket.append(text)

    def add_text(self, text: str, patterns: NamePatterns) -> None:
        for sentence in _SENT.split(text or ""):
            if not sentence or len(sentence) > MAX_SENTENCE_CHARS:
                continue
            self._add_sentence(sentence, patterns)

    def _add_sentence(self, s: str, patterns: NamePatterns) -> None:
        hits = find_mentions(s, patterns)
        if not hits:
            return
        # 나열: 구분자만 사이에 둔 이름 연속을 한 목록으로.
        runs: list[list[str]] = [[hits[0][2]]]
        for (_s1, e1, _a), (s2, _e2, b) in zip(hits, hits[1:]):
            if _SEP.match(s[e1:s2]):
                runs[-1].append(b)
            else:
                runs.append([b])
        for run in runs:
            for a, b in itertools.combinations(sorted(set(run)), 2):
                self.colist[(a, b)] += 1
                self._example((a, b), s)
        # 대안: 두 이름 사이에 대안 표현.
        seen_alt: set[Pair] = set()
        for (_s1, e1, a), (s2, _e2, b) in itertools.combinations(hits, 2):
            if a == b or s2 - e1 >= ALT_MAX_GAP or not _ALT.search(s[e1:s2]):
                continue
            pair = (a, b) if a < b else (b, a)
            if pair in seen_alt:
                continue
            seen_alt.add(pair)
            self.alt[pair] += 1
            self._example(pair, s)
        # is-a.
        for _st, en, key in hits:
            m = _ISA.match(s[en:])
            if m:
                bucket = self.isa.setdefault(key, [])
                if len(bucket) < MAX_ISA_PER_NAME and m.group(1) not in bucket:
                    bucket.append(m.group(1))

    def neighbors(self, key: str, limit: int = 5) -> list[Neighbor]:
        """``key``와 나열·대안으로 이어진 이름들(많은 순, 동률은 이름 순)."""
        counts: Counter[str] = Counter()
        examples: dict[str, list[str]] = {}
        for counter in (self.colist, self.alt):
            for (a, b), n in counter.items():
                if key not in (a, b):
                    continue
                other = b if a == key else a
                counts[other] += n
                examples.setdefault(other, [])
                for ex in self.examples.get((a, b), []):
                    if ex not in examples[other] and len(examples[other]) < MAX_EXAMPLES_PER_PAIR:
                        examples[other].append(ex)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [Neighbor(k, n, tuple(examples.get(k, ()))) for k, n in ranked]
