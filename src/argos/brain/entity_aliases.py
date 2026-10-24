"""같은 기술의 다른 이름을 하나로 잇는 별칭 판정 — ARG-312.

저장된 이름(``entities.normalized_key``)은 건드리지 않는다. 그 키는 사건 묶기와
ARG-307이 읽는 입력이라, 여기서 바꾸면 사건 묶기 결과가 달라진다. 대신 "이 이름은
저 이름의 다른 이름"이라는 별칭 층(``entity_aliases``)을 위에 얹는다.

판정에 LLM을 쓰지 않는다. 규칙은 둘이다.

- ``Spacing``: 공백만 다른 이름(``huggingface`` / ``hugging face``).
- ``VendorPrefix``: 회사 접두어 유무(``openai codex`` / ``codex``). 접두어 뿌리와
  고유성은 코퍼스 통계에서 뽑고, 증거가 모자라면 합치지 않는다.

모든 병합 간선은 **버전 서명이 같아야** 한다(``claude opus 5`` != ``claude opus 5.5``).
잘못 합치느니 덜 합치는 쪽이다. 이 모듈의 판정 함수는 전부 순수 함수다 — DB
읽기·쓰기는 ``alias_store``가 맡는다.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from argos.config import AliasConfig
from argos.models.entity_alias import AliasRule

# 문턱값은 config(``[aliases]``)에서 온다. 판정 함수는 순수 함수로 남기려고 값을
# 인자로 받고, 안 넘기면 기본값(AliasConfig())을 쓴다.
_DEFAULT = AliasConfig()


@dataclass(frozen=True)
class AliasEdge:
    """``a``와 ``b``가 같은 기술이라는 한 줄 증거. ``a``가 더 긴/접두어 쪽이다."""

    a: str
    b: str
    rule: AliasRule
    evidence: Mapping[str, object] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class AliasGroup:
    canonical: str
    aliases: tuple[str, ...]
    # 별칭 키 -> 규칙 / 근거. 키마다 하나만 남긴다.
    rules: Mapping[str, AliasRule] = field(default_factory=dict, hash=False, compare=False)
    evidence: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict, hash=False, compare=False
    )


def spacing_key(key: str) -> str:
    """공백을 뺀 비교 키. ``canonical_name``은 건드리지 않는다."""
    return "".join(key.split())


_NUMBER = re.compile(r"\d+(?:\.\d+)*")


def version_signature(key: str) -> tuple[str, ...]:
    """키에 든 숫자 덩어리들. 서명이 다르면 다른 버전이라 합치지 않는다.

    공백을 남긴 채 훑는다 — ``opus 5 5``(5, 5)와 ``opus 55``(55)를 가르려는 것이다.
    ``flux2``와 ``flux 2``는 둘 다 ("2",)다.
    """
    return tuple(_NUMBER.findall(key))


def spacing_edges(keys: Iterable[str]) -> list[AliasEdge]:
    """공백만 다른 이름끼리의 간선. 서명이 같은 이름끼리만 잇는다."""
    buckets: dict[str, list[str]] = {}
    for key in sorted(set(keys)):
        buckets.setdefault(spacing_key(key), []).append(key)
    edges: list[AliasEdge] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        by_signature: dict[tuple[str, ...], list[str]] = {}
        for member in members:
            by_signature.setdefault(version_signature(member), []).append(member)
        for same in by_signature.values():
            anchor = same[0]
            for other in same[1:]:
                edges.append(
                    AliasEdge(other, anchor, AliasRule.SPACING, {"spacing_key": spacing_key(anchor)})
                )
    return edges


def build_alias_groups(
    edges: Sequence[AliasEdge],
    doc_counts: Mapping[str, int],
) -> list[AliasGroup]:
    """간선을 union-find로 묶고 묶음마다 대표 이름 하나를 정한다. 결정적이다.

    서명이 다른 간선은 받지 않는다 — 어떤 규칙에서 와도 버전이 다르면 합치지 않는다.
    대표는 문서 수가 가장 많은 이름이고, 동률이면 짧은 키, 그다음 사전순이다.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    accepted: list[AliasEdge] = []
    for edge in sorted(edges, key=lambda e: (e.a, e.b, e.rule.value)):
        if version_signature(edge.a) != version_signature(edge.b):
            continue
        accepted.append(edge)
        root_a, root_b = find(edge.a), find(edge.b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    members: dict[str, set[str]] = {}
    for key in list(parent):
        members.setdefault(find(key), set()).add(key)

    groups: list[AliasGroup] = []
    for names in members.values():
        if len(names) < 2:
            continue
        canonical = min(names, key=lambda n: (-doc_counts.get(n, 0), len(n), n))
        aliases = tuple(sorted(names - {canonical}))
        rules: dict[str, AliasRule] = {}
        evidence: dict[str, dict[str, object]] = {}
        for alias in aliases:
            touching = [e for e in accepted if alias in (e.a, e.b)]
            # 접두어 증거가 있으면 그쪽 규칙을 우선한다.
            chosen = next((e for e in touching if e.rule is AliasRule.VENDOR_PREFIX), touching[0])
            rules[alias] = chosen.rule
            evidence[alias] = {
                "edges": [
                    {"rule": e.rule.value, "a": e.a, "b": e.b, **dict(e.evidence)} for e in touching
                ]
            }
        groups.append(AliasGroup(canonical, aliases, rules, evidence))
    return sorted(groups, key=lambda g: g.canonical)


# --------------------------------------------------------------------------- #
# 규칙 B: 회사 접두어 (VendorPrefix)
# --------------------------------------------------------------------------- #

HOLD_BASE_MISSING = "base_missing"
HOLD_VERSION_MISMATCH = "version_mismatch"
HOLD_BASE_IS_VENDOR = "base_is_vendor"
HOLD_BASE_TOO_SHORT = "base_too_short"
HOLD_FEW_LITERAL_DOCS = "few_literal_docs"
HOLD_COMPETING_VENDOR = "competing_vendor"
HOLD_BASE_LOWERCASE = "base_lowercase"
HOLD_BASE_UNVERIFIED = "base_unverified"
HOLD_REVERSED_PAIR = "reversed_pair"
HOLD_BASE_HAS_OTHER_VENDOR = "base_has_other_vendor"
HOLD_SHORT_BASE_LOWERCASE = "short_base_lowercase"

# 살아남은 B 집합 -> 소문자 비율. 비율을 못 구한 B는 빠져도 된다(보류된다).
RatioFn = Callable[[frozenset[str]], Mapping[str, float]]


@dataclass(frozen=True)
class VendorCandidate:
    """``alias`` = "``vendor`` ``base``" 꼴의 이름. ``base``로 합칠지가 판정 대상이다."""

    alias: str
    vendor: str
    base: str


@dataclass(frozen=True)
class HeldPair:
    """합치지 않고 따로 둔 쌍과 그 사유."""

    alias: str
    base: str
    reason: str


@dataclass(frozen=True)
class VendorStats:
    """접두어 판정에 필요한 코퍼스 통계. DB가 아니라 값으로 받는다."""

    keys: frozenset[str]
    roots: frozenset[str]
    doc_counts: Mapping[str, int]
    lowercase_ratios: RatioFn
    # 믿을 만한 접두어(trusted_vendors). B가 다른 접두어로 시작하는지 가를 때 쓴다.
    vendors: frozenset[str] = frozenset()


@dataclass(frozen=True)
class VendorDecision:
    edges: tuple[AliasEdge, ...]
    held: tuple[HeldPair, ...]


def find_roots(keys: Iterable[str], cfg: AliasConfig = _DEFAULT) -> frozenset[str]:
    """접두어 뿌리 — 자신이 이름이고, "뿌리 X" 꼴 이름 중 **X도 이름인 것**이 3개 이상인 이름.

    하드코딩 목록이 아니라 코퍼스에서 뽑는다. 낱말 경계로 센다(``open``은
    ``openai x``를 따르는 이름으로 치지 않는다).

    나머지 X도 이름이어야 센다는 조건이 회사/제품 접두어를 가른다. ``openai``는
    ``openai codex``(codex도 이름), ``openai sora`` ...처럼 따르는 이름의 나머지가 각자
    독립된 이름이라 뿌리지만, ``codex``는 ``codex cli``의 ``cli`` 같은 짧은 조각이
    따를 뿐이라 뿌리가 아니다. 그래서 ``microsoft openai -> openai``(회사끼리)는 막히고
    ``openai codex -> codex``(회사와 제품)는 통과한다. 사람 이름의 이름 쪽(``adam
    harvey``)도 같은 이유로 뿌리가 못 된다.
    """
    key_set = set(keys)
    followers: dict[str, int] = {}
    for key in key_set:
        tokens = key.split()
        for cut in range(1, len(tokens)):
            prefix = " ".join(tokens[:cut])
            if " ".join(tokens[cut:]) in key_set:
                followers[prefix] = followers.get(prefix, 0) + 1
    return frozenset(
        prefix
        for prefix, count in followers.items()
        if prefix in key_set and count >= cfg.min_root_followers
    )


def trusted_vendors(
    roots: frozenset[str],
    doc_counts: Mapping[str, int],
    ratio_fn: RatioFn,
    cfg: AliasConfig = _DEFAULT,
) -> frozenset[str]:
    """접두어 뿌리 중 **회사/제품 접두어로 믿을 만한** 것만 남긴다.

    따르는 이름이 3개라는 사실만으로는 뿌리가 일반 낱말(``open``, ``read``, ``us``)이나
    사람 이름(``adam``)이어도 통과한다. 실측 미리보기에서 ``open the url -> the url``,
    ``adam harvey -> harvey`` 같은 오탐이 나와서 두 조건을 더 건다. 둘 다 "덜 합치는
    쪽"으로만 작용한다.

    - 뿌리 자신이 ``min_literal_docs`` 문서 이상에 이름으로 나왔다(한 문서짜리는 접두어로
      확립된 게 아니다).
    - 뿌리가 본문에서 소문자로 쓰인 비율이 ``max_lowercase_ratio`` 이하다(일반 낱말 거름).
      비율을 못 구하면 믿지 않는다.
    """
    seen = frozenset(r for r in roots if doc_counts.get(r, 0) >= cfg.min_literal_docs)
    ratios = ratio_fn(seen) if seen else {}
    return frozenset(r for r in seen if r in ratios and ratios[r] <= cfg.max_lowercase_ratio)


def vendor_candidates(keys: Iterable[str], roots: frozenset[str]) -> list[VendorCandidate]:
    """접두어 뿌리로 시작하는 모든 (이름, 뿌리, 나머지) 분해. 정렬해 결정적이다."""
    found: list[VendorCandidate] = []
    for key in sorted(set(keys)):
        tokens = key.split()
        for cut in range(1, len(tokens)):
            vendor = " ".join(tokens[:cut])
            if vendor in roots:
                found.append(VendorCandidate(key, vendor, " ".join(tokens[cut:])))
    return found


def _is_distinctive_length(base: str, cfg: AliasConfig = _DEFAULT) -> bool:
    """짧은 약어는 접두어를 떼면 다른 말과 부딪힌다(``api``, ``i o``). 낱말 사이 공백은
    글자 수에 세지 않는다 — ``i o``(I/O)가 두 낱말이라는 이유로 통과하면 안 된다."""
    return len(base.replace(" ", "")) >= cfg.min_single_word_length or any(
        ch.isdigit() for ch in base
    )


def _is_short_plain_word(base: str, cfg: AliasConfig = _DEFAULT) -> bool:
    """숫자 없는 한 낱말이고 ``short_word_length``보다 짧은가(``omni``, ``opus``, ``duet``).

    이런 B는 일반 낱말과 겹치기 쉬워(``omni``-modal) 고유명이라는 증거를 더 세게 요구한다.
    """
    return (
        " " not in base
        and not any(ch.isdigit() for ch in base)
        and len(base) < cfg.short_word_length
    )


def decide_vendor_prefix(
    candidates: Sequence[VendorCandidate],
    stats: VendorStats,
    cfg: AliasConfig = _DEFAULT,
) -> VendorDecision:
    """후보마다 합칠지 정한다. 한 조건이라도 모자라면 합치지 않는다.

    싼 검사를 먼저 하고, 마지막에 살아남은 B들에 대해서만 소문자 비율을 한 번에
    구한다(본문 전체를 훑는 비싼 계산이라서).
    """
    held: list[HeldPair] = []
    survivors: list[VendorCandidate] = []

    # 같은 B를 노리는 접두어 후보들의 문서 수 — 경쟁 판정용.
    rivals: dict[str, dict[str, int]] = {}
    for cand in candidates:
        rivals.setdefault(cand.base, {})[cand.alias] = stats.doc_counts.get(cand.alias, 0)

    for cand in candidates:
        reason = _cheap_hold_reason(cand, stats, rivals[cand.base], cfg)
        if reason is None:
            survivors.append(cand)
        else:
            held.append(HeldPair(cand.alias, cand.base, reason))

    ratios = stats.lowercase_ratios(frozenset(c.base for c in survivors)) if survivors else {}
    edges: list[AliasEdge] = []
    for cand in survivors:
        ratio = ratios.get(cand.base)
        if ratio is None:
            held.append(HeldPair(cand.alias, cand.base, HOLD_BASE_UNVERIFIED))
        elif ratio > cfg.max_lowercase_ratio:
            held.append(HeldPair(cand.alias, cand.base, HOLD_BASE_LOWERCASE))
        elif _is_short_plain_word(cand.base, cfg) and ratio > cfg.max_short_word_lowercase_ratio:
            # `gemini omni -> omni`(0.12)는 막고 `claude opus -> opus`(0.02)는 통과시킨다.
            held.append(HeldPair(cand.alias, cand.base, HOLD_SHORT_BASE_LOWERCASE))
        else:
            others = [n for alias, n in rivals[cand.base].items() if alias != cand.alias]
            edges.append(
                AliasEdge(
                    cand.alias,
                    cand.base,
                    AliasRule.VENDOR_PREFIX,
                    {
                        "vendor": cand.vendor,
                        "literal_docs": stats.doc_counts.get(cand.alias, 0),
                        "lowercase_ratio": round(ratio, 4),
                        "rival_docs": max(others, default=0),
                    },
                )
            )
    return VendorDecision(
        edges=tuple(sorted(edges, key=lambda e: (e.a, e.b))),
        held=tuple(sorted(held, key=lambda h: (h.alias, h.base))),
    )


def _cheap_hold_reason(
    cand: VendorCandidate,
    stats: VendorStats,
    rivals: Mapping[str, int],
    cfg: AliasConfig = _DEFAULT,
) -> str | None:
    base = cand.base
    if base not in stats.keys:
        return HOLD_BASE_MISSING
    if version_signature(cand.alias) != version_signature(base):
        return HOLD_VERSION_MISMATCH
    if f"{base} {cand.vendor}" in stats.keys:
        # "V B"와 "B V"가 둘 다 이름이면 한쪽이 다른 쪽의 제품이 아니라 나란히 적힌
        # 이름(`openai anthropic` / `anthropic openai`)이다.
        return HOLD_REVERSED_PAIR
    base_tokens = base.split()
    if any(
        " ".join(base_tokens[:cut]) in stats.vendors and " ".join(base_tokens[:cut]) != cand.vendor
        for cut in range(1, len(base_tokens))
    ):
        # B가 이미 다른 접두어의 이름이다(`codex claude code`의 claude code는 claude 쪽).
        # V를 떼면 남는 건 다른 회사·제품이지 V의 다른 이름이 아니다.
        return HOLD_BASE_HAS_OTHER_VENDOR
    if base in stats.roots and stats.doc_counts.get(cand.vendor, 0) <= stats.doc_counts.get(base, 0):
        # B도 뿌리면 V가 B보다 확립된 이름(문서가 더 많음)일 때만 접두어로 본다.
        # `microsoft openai -> openai`는 막히고(openai가 더 크다) `openai codex -> codex`는
        # 통과한다. 제품(codex, cuda, opus 5)도 자기 이름이 따르는 이름을 거느려 뿌리가
        # 되기 때문에, "B는 뿌리가 아니다"만으로는 가장 흔한 사례가 다 막힌다.
        return HOLD_BASE_IS_VENDOR
    if not _is_distinctive_length(base, cfg):
        return HOLD_BASE_TOO_SHORT
    # 읽기 시점 "흔한 이름"(문서 비율)은 거르지 않는다 — codex·claude처럼 흔한 건
    # 그만큼 많이 언급되는 제품이라서, 문서 비율만으로는 일반 낱말과 가를 수 없다.
    # 일반 낱말은 위의 길이 검사와 아래 소문자 비율 검사가 가른다.
    own = stats.doc_counts.get(cand.alias, 0)
    if own < cfg.min_literal_docs:
        return HOLD_FEW_LITERAL_DOCS
    others = [n for alias, n in rivals.items() if alias != cand.alias]
    if others and own < cfg.competitor_margin * max(others):
        return HOLD_COMPETING_VENDOR
    return None


_WORD = re.compile(r"[^\W_]+")


# --------------------------------------------------------------------------- #
# 계획 계산 — DB 없이 입력 값만으로
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EntityRow:
    id: uuid.UUID
    key: str
    name: str


@dataclass(frozen=True)
class ExistingAlias:
    alias_id: uuid.UUID
    canonical_id: uuid.UUID
    rule: AliasRule


@dataclass(frozen=True)
class AliasInputs:
    entities: Sequence[EntityRow]
    doc_counts: Mapping[str, int]
    # 문서 본문(title + raw_content). 소문자 비율 계산에만 쓴다.
    texts: Sequence[str]
    existing: Sequence[ExistingAlias]


@dataclass(frozen=True)
class AliasLink:
    alias_id: uuid.UUID
    canonical_id: uuid.UUID
    alias_key: str
    canonical_key: str
    rule: AliasRule
    evidence: Mapping[str, object] = field(default_factory=dict, hash=False, compare=False)


@dataclass(frozen=True)
class AliasPlan:
    added: tuple[AliasLink, ...]
    removed: tuple[AliasLink, ...]
    unchanged: int
    held: tuple[HeldPair, ...]
    groups: int

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)

    def held_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pair in self.held:
            counts[pair.reason] = counts.get(pair.reason, 0) + 1
        return dict(sorted(counts.items()))


def compute_alias_plan(inputs: AliasInputs, cfg: AliasConfig = _DEFAULT) -> AliasPlan:
    """저장된 이름과 코퍼스 통계로 별칭을 다시 계산하고 기존 행과 견준다.

    ``Manual`` 행(신고 교정, ARG-317)은 건드리지 않는다 — 그 행에 걸린 이름은
    자동 판정 대상에서 아예 뺀다. 같은 입력이면 항상 같은 계획이 나오고, 방금 쓴
    결과를 다시 넣으면 ``added``/``removed``가 비어 있다(멱등).
    """
    manual_ids: set[uuid.UUID] = set()
    for row in inputs.existing:
        if row.rule is AliasRule.MANUAL:
            manual_ids.update((row.alias_id, row.canonical_id))
    entity_by_key = {e.key: e for e in inputs.entities if e.id not in manual_ids}
    entity_by_id = {e.id: e for e in inputs.entities}
    keys = frozenset(entity_by_key)

    roots = find_roots(keys, cfg)

    def ratio_fn(names: frozenset[str]) -> Mapping[str, float]:
        return lowercase_ratios(inputs.texts, names)

    vendors = trusted_vendors(roots, inputs.doc_counts, ratio_fn, cfg)
    stats = VendorStats(
        keys=keys,
        roots=roots,  # B가 뿌리인지 막는 검사는 믿을 만하지 않은 뿌리까지 전부 쓴다
        doc_counts=inputs.doc_counts,
        lowercase_ratios=ratio_fn,
        vendors=vendors,
    )
    decision = decide_vendor_prefix(vendor_candidates(keys, vendors), stats, cfg)
    groups = build_alias_groups(
        [*spacing_edges(keys), *decision.edges],
        inputs.doc_counts,
    )

    desired: dict[uuid.UUID, AliasLink] = {}
    for group in groups:
        canonical = entity_by_key[group.canonical]
        for alias_key in group.aliases:
            alias = entity_by_key[alias_key]
            desired[alias.id] = AliasLink(
                alias_id=alias.id,
                canonical_id=canonical.id,
                alias_key=alias_key,
                canonical_key=group.canonical,
                rule=group.rules[alias_key],
                evidence=group.evidence[alias_key],
            )

    existing_auto = {
        row.alias_id: row for row in inputs.existing if row.rule is not AliasRule.MANUAL
    }
    added: list[AliasLink] = []
    unchanged = 0
    for alias_id, link in desired.items():
        current = existing_auto.get(alias_id)
        if current is not None and (current.canonical_id, current.rule) == (
            link.canonical_id,
            link.rule,
        ):
            unchanged += 1
        else:
            added.append(link)

    removed: list[AliasLink] = []
    for alias_id, row in existing_auto.items():
        link = desired.get(alias_id)
        if link is not None and (link.canonical_id, link.rule) == (row.canonical_id, row.rule):
            continue
        alias = entity_by_id.get(alias_id)
        canonical = entity_by_id.get(row.canonical_id)
        removed.append(
            AliasLink(
                alias_id=alias_id,
                canonical_id=row.canonical_id,
                alias_key=alias.key if alias else str(alias_id),
                canonical_key=canonical.key if canonical else str(row.canonical_id),
                rule=row.rule,
            )
        )

    return AliasPlan(
        added=tuple(sorted(added, key=lambda link: link.alias_key)),
        removed=tuple(sorted(removed, key=lambda link: link.alias_key)),
        unchanged=unchanged,
        held=decision.held,
        groups=len(groups),
    )


def lowercase_ratios(texts: Iterable[str], targets: Iterable[str]) -> dict[str, float]:
    """각 target 이름이 본문에서 **전부 소문자로** 쓰인 비율.

    ``code``·``agent``는 일반 낱말이라 소문자로 많이 쓰이고(0.75, 0.88), ``codex``·
    ``cuda``는 고유명이라 거의 안 그렇다(0.07, 0.0). 본문에 한 번도 안 나온 target은
    결과에서 빠진다 — 확인할 수 없으니 호출자가 보류해야 한다.
    """
    by_tokens: dict[tuple[str, ...], list[str]] = {}
    for target in targets:
        tokens = tuple(w.casefold() for w in _WORD.findall(target))
        if tokens:
            by_tokens.setdefault(tokens, []).append(target)
    if not by_tokens:
        return {}
    lengths = sorted({len(t) for t in by_tokens})
    firsts = {t[0] for t in by_tokens}
    total: dict[str, int] = {}
    lower: dict[str, int] = {}
    for text in texts:
        words = _WORD.findall(text)
        folded = [w.casefold() for w in words]
        for i, word in enumerate(folded):
            if word not in firsts:
                continue
            for n in lengths:
                if i + n > len(folded):
                    break
                keys = by_tokens.get(tuple(folded[i : i + n]))
                if not keys:
                    continue
                is_lower = all(w == w.lower() for w in words[i : i + n])
                for key in keys:
                    total[key] = total.get(key, 0) + 1
                    if is_lower:
                        lower[key] = lower.get(key, 0) + 1
    return {key: lower.get(key, 0) / count for key, count in total.items()}
