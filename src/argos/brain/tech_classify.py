"""새 기술 자동 갈래 배정 — 판단 로직 (ARG-313).

하루 한 번 배치(``argos taxonomy assign``)가 쓴다. 사람의 승인 없이 대표 이름마다
(1) 기술인지 거르고, (2) 기술이면 무엇인지 한 줄로 설명하고, (3) 확정된 역할
트리의 **아래 갈래** 하나에 배정한다. 시험판(2026-10-22)의 D 방식 그대로다:
LLM이 만든 설명에 **LLM 없이 모은 증거**(나열·대안 이웃의 현재 갈래, is-a 구,
외부 태그)를 붙여 배정을 묻고, 이웃 갈래가 바뀐 이름만 한 번 더 묻는다.

이 모듈은 DB를 모른다. 입력(``TargetInput``)은 ``tech_classify_store``가 모으고,
결과는 ``AssignPlan``으로 돌려준다 — 미리보기와 기록이 **같은 계획**을 쓰므로
확인 뒤에 LLM을 다시 부르지 않는다. 갈래 트리(``role_branches``)는 읽기만 한다.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from argos.brain.taxonomy import Tree, iter_branches
from argos.brain.taxonomy_draft import _parse_json
from argos.brain.tech_evidence import Neighbor
from argos.config import TaxonomyConfig
from argos.models.technology_profile import NotTechReason, TechVerdict

logger = logging.getLogger(__name__)


class UnclassifiedReason:
    """``role_assignments.unclassified_reason`` 값."""

    INSUFFICIENT = "InsufficientEvidence"  # 설명할 문장도, LLM 없는 증거도 없음
    NO_FIT = "NoFittingBranch"  # LLM이 맞는 갈래가 없다고 함(-1)
    # LLM 인프라 실패(연결·타임아웃·응답 파싱 불가)는 미분류로 저장하지 않는다 —
    # 계획에서 빼고(``AssignPlan.skipped``) 다음 실행에서 다시 대상이 된다.


# --------------------------------------------------------------------- #
# 1. 일반어·장소·회사 이름 거르기 (LLM 없음)
# --------------------------------------------------------------------- #
_CONTRACTION = re.compile(r"(?:['’](?:m|s|re|ll|ve|d)|n['’]t)$", re.IGNORECASE)
_PLACE_LABELS = frozenset({"GPE", "LOC", "NORP"})
_PERSON_LABELS = frozenset({"PERSON"})


@dataclass(frozen=True)
class ScreenSignals:
    """한 이름의 거르기 신호. 모두 LLM 없이 코퍼스에서 계산한다."""

    key: str
    display: str
    common: bool  # entity_frequency.common_names에 듦
    lowercase_ratio: float | None  # 본문에서 전부 소문자로 쓰인 비율(못 구하면 None)
    # entity_aliases.trusted_vendors에 듦(접두어 뿌리). **거르기에는 쓰지 않는다** — 운영 DB
    # 실측에서 claude code·vllm·mcp·gemma 4 같은 제품 계열도 뿌리였다. LLM 설명 프롬프트에
    # "회사일 수 있다"는 힌트로만 넘긴다.
    is_vendor_root: bool
    ner_labels: Mapping[str, int] = field(default_factory=dict)  # spaCy 라벨 수(없으면 빈 dict)
    stored_name: str = ""  # entities.name(첫 등장 표기, 예: "I'm") — 표시 이름과 다를 수 있다


def majority_ner_label(labels: Mapping[str, int], *, min_mentions: int) -> str | None:
    """라벨이 ``min_mentions`` 이상이고 한 라벨이 과반이면 그 라벨."""
    total = sum(labels.values())
    if total < min_mentions:
        return None
    label, count = max(labels.items(), key=lambda kv: (kv[1], kv[0]))
    return label if count * 2 > total else None


def screen(signals: ScreenSignals, cfg: TaxonomyConfig) -> str | None:
    """기술이 아니면 ``NotTechReason`` 값, 통과면 ``None``. 덜 거르는 쪽으로 둔다 —
    여기서 놓친 일반어·회사 이름은 LLM 설명 단계(``is_technology``)가 한 번 더 거른다.

    흔한 이름은 **짧은 것(3자 이하: ai, api, llm)만** 여기서 거른다. 운영 DB 실측에서
    흔한 이름 목록에 codex·chatgpt·claude도 들어 있었다 — 자주 나온다고 기술이 아닌 게
    아니므로 나머지는 LLM이 판정한다.
    """
    key = signals.key.strip()
    if signals.common and len(key) <= cfg.common_short_max_chars:
        return NotTechReason.COMMON
    names = [n.strip() for n in (signals.display, signals.stored_name) if n and n.strip()]
    if len(key) < 2 or any(_CONTRACTION.search(n) for n in names):
        return NotTechReason.GENERIC
    if (
        signals.lowercase_ratio is not None
        and signals.lowercase_ratio > cfg.generic_lowercase_ratio
    ):
        return NotTechReason.GENERIC
    label = majority_ner_label(signals.ner_labels, min_mentions=cfg.ner_min_mentions)
    if label in _PLACE_LABELS:
        return NotTechReason.PLACE
    if label in _PERSON_LABELS:
        return NotTechReason.PERSON
    return None


# --------------------------------------------------------------------- #
# 갈래
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Leaf:
    index: int
    branch_id: str | None
    top: str
    name: str
    description: str

    @property
    def path(self) -> str:
        return f"{self.top} > {self.name}"


def leaves_of(tree: Tree) -> list[Leaf]:
    """배정 후보 = 아래 갈래만. 위 갈래에는 배정하지 않는다."""
    out: list[Leaf] = []
    for depth, branch, parent in iter_branches(tree):
        if depth == 2 and parent is not None:
            out.append(Leaf(len(out), branch.id, parent.name, branch.name, branch.description))
    return out


def size_suggestions(counts: Mapping[str, int], cfg: TaxonomyConfig) -> list[str]:
    """아래 갈래가 커졌을 때의 나누기 **제안** 문장. 갈래는 바꾸지 않는다."""
    total = sum(counts.values())
    out: list[str] = []
    for path, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        too_many = n > cfg.split_suggest_size
        too_big = total > 0 and n / total > cfg.split_suggest_ratio and n > 2
        if too_many or too_big:
            why = f"{n}개" + (f", 전체의 {n / total:.0%}" if total else "")
            name = path.split(" > ", 1)[-1]
            out.append(
                f"'{path}' 갈래가 커졌습니다({why}) — 나누려면: argos taxonomy export → "
                f"argos taxonomy split {name!r} --part ... → argos taxonomy apply"
            )
    return out


# --------------------------------------------------------------------- #
# 입력·출력
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExistingProfile:
    verdict: TechVerdict
    not_tech_reason: str | None
    kind: str | None
    description: str | None
    role_en: str | None
    described_doc_count: int


@dataclass(frozen=True)
class TargetInput:
    entity_id: Any  # uuid.UUID (테스트에서는 문자열도 받는다)
    key: str
    display: str
    doc_count: int
    sentences: tuple[str, ...]
    signals: ScreenSignals
    neighbors: tuple[Neighbor, ...] = ()
    isa: tuple[str, ...] = ()
    external_tags: tuple[str, ...] = ()
    existing: ExistingProfile | None = None
    needs_describe: bool = True

    @property
    def evidence_count(self) -> int:
        return len(self.neighbors) + len(self.isa) + len(self.external_tags)


@dataclass(frozen=True)
class ProfileWrite:
    entity_id: Any
    key: str
    display: str
    verdict: TechVerdict
    not_tech_reason: str | None = None
    kind: str | None = None
    description: str | None = None
    role_en: str | None = None
    described_doc_count: int = 0
    model: str | None = None


@dataclass(frozen=True)
class AssignmentWrite:
    entity_id: Any
    key: str
    display: str
    branch_id: str | None
    leaf_path: str | None
    unclassified_reason: str | None
    evidence: dict[str, Any]


@dataclass(frozen=True)
class AssignPlan:
    run_id: uuid.UUID
    profiles: tuple[ProfileWrite, ...] = ()
    assignments: tuple[AssignmentWrite, ...] = ()
    # 기술이 아니게 된 이름: 자동 배정 행을 지운다(Manual 행은 그대로).
    unassign: tuple[tuple[Any, str], ...] = ()  # (entity_id, display)
    screened: tuple[tuple[str, str], ...] = ()  # (display, NotTechReason)
    # LLM 인프라 실패로 이번 계획에서 뺀 이름(아무것도 쓰지 않는다 → 다음 실행에서 다시 대상).
    skipped: tuple[tuple[str, str], ...] = ()  # (display, "describe" | "assign")
    llm_calls: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.profiles or self.assignments or self.unassign)

    def preview_lines(self, *, sample: int | None = None) -> list[str]:
        lines: list[str] = []
        by_leaf: dict[str, list[AssignmentWrite]] = {}
        unclassified: list[AssignmentWrite] = []
        for a in self.assignments:
            if a.leaf_path:
                by_leaf.setdefault(a.leaf_path, []).append(a)
            else:
                unclassified.append(a)
        descs = {p.key: p.description for p in self.profiles}

        def cut(items: list) -> list:
            return items if sample is None else items[:sample]

        lines.append(f"배정 {sum(len(v) for v in by_leaf.values())}개:")
        for path in sorted(by_leaf):
            items = by_leaf[path]
            lines.append(f"  ■ {path} ({len(items)})")
            for a in cut(items):
                d = descs.get(a.key) or a.evidence.get("description") or ""
                lines.append(f"      - {a.display}" + (f" — {d}" if d else ""))
            if sample is not None and len(items) > sample:
                lines.append(f"      … 외 {len(items) - sample}개")
        lines.append(f"미분류 {len(unclassified)}개:")
        for a in cut(unclassified):
            lines.append(f"  - {a.display} ({a.unclassified_reason})")
        lines.append(f"기술 아님으로 거른 이름 {len(self.screened)}개:")
        for display, reason in cut(list(self.screened)):
            lines.append(f"  - {display} ({reason})")
        if self.unassign:
            lines.append(f"기존 자동 배정을 지울 이름 {len(self.unassign)}개:")
            for _eid, display in cut(list(self.unassign)):
                lines.append(f"  - {display}")
        if self.skipped:
            lines.append(
                f"LLM 실패로 건너뜀 {len(self.skipped)}개 (아무것도 쓰지 않음 — 다음 실행에서 다시 시도):"
            )
            for display, stage in cut(list(self.skipped)):
                lines.append(f"  - {display} ({'설명' if stage == 'describe' else '배정'} 단계)")
        return lines


# --------------------------------------------------------------------- #
# 2. LLM 단계
# --------------------------------------------------------------------- #
DESCRIBE_PROMPT = """Below are sentences from tech news articles that mention "{name}".
Describe what {name} IS, based only on these sentences. Return JSON:
{{"is_technology": true if {name} is a technology (a product, model, tool, library, framework, protocol, service, platform or benchmark); false if it is a company, organization, person, place, event or a generic word,
 "not_tech_kind": one of "company", "person", "place", "generic", "none",
 "enough": false if the sentences do not say what {name} is or does (for example only funding, pricing, adoption or partnership news),
 "kind": short noun phrase for the type of thing (e.g. "coding agent", "open-weight LLM", "inference engine", "database", "benchmark"),
 "does_what": one sentence on what it does for its user,
 "for_whom": who uses it,
 "one_liner": one English sentence describing its ROLE, without the company name,
 "description_ko": one Korean sentence (under 60 characters) describing its role, without the company name}}
If the sentences are not enough, set "enough" to false instead of guessing.

Sentences:
{sents}"""

VENDOR_HINT = """

Note: "{name}" is often used in front of other product names in the corpus, so it may be a company or a product family. If it is a company, set "is_technology" to false and "not_tech_kind" to "company"."""

ASSIGN_PROMPT = """Pick the single best leaf category for this technology.
Technology: {name}
LLM description (may be wrong if source sentences were thin): {one_liner} (kind: {kind})
Evidence from text patterns and external lists (more reliable than the description when present):
{ev}
Leaf categories:
{leaves}
If no leaf category fits the technology's role, use -1.
Return JSON: {{"leaf_index": <number>, "reason": short}}"""

_NOT_TECH_KIND = {
    "company": NotTechReason.VENDOR,
    "organization": NotTechReason.VENDOR,
    "person": NotTechReason.PERSON,
    "place": NotTechReason.PLACE,
    "generic": NotTechReason.GENERIC,
}


def _truthy(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return default


def describe_result(raw: str) -> dict[str, Any] | None:
    """설명 응답 -> dict. 읽을 수 없으면 None."""
    try:
        data = _parse_json(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def evidence_lines(target: TargetInput, current: Mapping[str, str | None]) -> list[str]:
    """시험판 assignD.py와 같은 증거 문장. ``current``: 이웃 키 -> 현재 갈래 경로."""
    ev: list[str] = []
    for nb in target.neighbors:
        where = current.get(nb.key)
        state = f'is currently in "{where}"' if where else "is not classified yet"
        ev.append(f"- listed side by side with {nb.key} ({nb.count} times); {nb.key} {state}")
    for x in target.isa[:2]:
        ev.append(f'- article says: "{target.display} is a {x}"')
    if target.external_tags:
        ev.append("- external tags: " + ", ".join(target.external_tags[:6]))
    return ev


def parse_leaf_index(raw: str, n_leaves: int) -> tuple[int | None, str, bool]:
    """(갈래 번호 | None=미분류, 이유, 읽기 성공 여부)."""
    try:
        data = _parse_json(raw)
        idx = int(data.get("leaf_index", -1))
    except (ValueError, TypeError, AttributeError):
        return None, "", False
    reason = str(data.get("reason") or "")[:300]
    return (idx if 0 <= idx < n_leaves else None), reason, True


async def _query(llm: Any, prompt: str, cfg: TaxonomyConfig) -> str:
    return await llm.query(
        "large",
        prompt,
        keep_alive="10m",
        num_ctx=cfg.num_ctx,
        think=False,
        temperature=0,
    )


async def build_plan(
    targets: Sequence[TargetInput],
    leaves: Sequence[Leaf],
    llm: Any,
    cfg: TaxonomyConfig,
    *,
    current: Mapping[str, str | None] | None = None,
    model_name: str | None = None,
    run_id: uuid.UUID | None = None,
) -> AssignPlan:
    """대상들의 판정·설명·배정 계획. LLM은 한 번에 하나씩 부른다(순차).

    ``current``는 DB에 이미 있는 배정(대표 키 -> 갈래 경로)이다. 이웃 증거가
    "그 이웃이 지금 어느 갈래인가"를 말하려고 쓴다.
    """
    cur: dict[str, str | None] = dict(current or {})
    calls = 0
    profiles: list[ProfileWrite] = []
    screened: list[tuple[str, str]] = []
    unassign: list[tuple[Any, str]] = []
    to_assign: list[tuple[TargetInput, ProfileWrite]] = []
    unclassified: list[AssignmentWrite] = []

    # --- 판정·설명
    skipped: list[tuple[str, str]] = []
    described: list[tuple[TargetInput, ProfileWrite, bool]] = []
    for t in targets:
        if not t.needs_describe and t.existing is not None:
            ex = t.existing
            prof = ProfileWrite(
                entity_id=t.entity_id, key=t.key, display=t.display, verdict=ex.verdict,
                not_tech_reason=ex.not_tech_reason, kind=ex.kind, description=ex.description,
                role_en=ex.role_en, described_doc_count=ex.described_doc_count,
            )
            described.append((t, prof, False))
        else:
            got, used_llm = await _describe(t, llm, cfg, model_name)
            calls += used_llm
            if got is None:
                # 인프라 실패: 프로필·배정을 쓰지 않는다. 기존 행(설명·described_doc_count)은
                # 그대로 남아 다음 실행에서 다시 대상이 된다.
                skipped.append((t.display, "describe"))
                continue
            described.append((t, got, True))

    # 이번 회에 기술이 아니라고 판정된 이름은 이웃 증거에서도 뺀다(OpenAI가 이웃이면 안 된다).
    not_tech = {t.key for t, p, _w in described if p.verdict == TechVerdict.NOT_TECHNOLOGY}
    for t, prof, written in described:
        if not_tech and any(nb.key in not_tech for nb in t.neighbors):
            t = replace(t, neighbors=tuple(nb for nb in t.neighbors if nb.key not in not_tech))
        if prof.verdict == TechVerdict.NOT_TECHNOLOGY:
            screened.append((t.display, prof.not_tech_reason or NotTechReason.LLM))
            unassign.append((t.entity_id, t.display))
            if written:
                profiles.append(prof)
            continue
        if written:
            profiles.append(prof)
        if prof.verdict == TechVerdict.INSUFFICIENT and t.evidence_count == 0:
            unclassified.append(
                AssignmentWrite(
                    entity_id=t.entity_id, key=t.key, display=t.display, branch_id=None,
                    leaf_path=None, unclassified_reason=UnclassifiedReason.INSUFFICIENT,
                    evidence=_evidence_dict(t, prof, cur, reason="", round_no=0),
                )
            )
            cur[t.key] = None
            continue
        to_assign.append((t, prof))

    # --- 배정 (라운드 1: 전부, 라운드 2+: 이웃 갈래가 바뀐 이름만)
    results: dict[str, AssignmentWrite] = {}
    pending = list(to_assign)
    for round_no in range(1, cfg.assign_rounds + 1):
        if not pending or not leaves:
            break
        changed: set[str] = set()
        new_cur: dict[str, str | None] = {}
        for t, prof in pending:
            write, used = await _assign_one(t, prof, leaves, llm, cfg, cur, round_no)
            calls += used
            if write is None:
                # 인프라 실패: 앞 라운드 결과가 있으면 그것을 두고, 없으면 배정을 쓰지 않는다
                # (기존 Auto 배정을 미분류로 덮어쓰지 않는다 — 아래에서 skipped로 보고).
                continue
            results[t.key] = write
            new_cur[t.key] = write.leaf_path
            if cur.get(t.key) != write.leaf_path or t.key not in cur:
                changed.add(t.key)
        cur.update(new_cur)
        pending = [
            (t, prof)
            for t, prof in to_assign
            if any(nb.key in changed and nb.key != t.key for nb in t.neighbors)
        ]
    if not leaves:
        for t, prof in to_assign:
            results[t.key] = AssignmentWrite(
                entity_id=t.entity_id, key=t.key, display=t.display, branch_id=None,
                leaf_path=None, unclassified_reason=UnclassifiedReason.NO_FIT,
                evidence=_evidence_dict(t, prof, cur, reason="no leaves", round_no=0),
            )

    for t, _p in to_assign:
        if t.key not in results:
            skipped.append((t.display, "assign"))
    assignments = [results[t.key] for t, _p in to_assign if t.key in results] + unclassified
    return AssignPlan(
        run_id=run_id or uuid.uuid4(),
        profiles=tuple(profiles),
        assignments=tuple(assignments),
        unassign=tuple(unassign),
        screened=tuple(screened),
        skipped=tuple(skipped),
        llm_calls=calls,
    )


async def _describe(
    t: TargetInput, llm: Any, cfg: TaxonomyConfig, model_name: str | None
) -> tuple[ProfileWrite | None, int]:
    """(프로필 | None=LLM 인프라 실패, LLM 호출 수). 문장 부족·LLM의 enough=false는
    정상 Insufficient로 프로필을 만든다. 연결 오류·타임아웃·읽을 수 없는 응답은 None."""
    reason = screen(t.signals, cfg)
    base = dict(entity_id=t.entity_id, key=t.key, display=t.display, described_doc_count=t.doc_count)
    if reason is not None:
        return ProfileWrite(verdict=TechVerdict.NOT_TECHNOLOGY, not_tech_reason=reason, **base), 0
    if len(t.sentences) < cfg.min_sentences:
        return ProfileWrite(verdict=TechVerdict.INSUFFICIENT, **base), 0

    prompt = DESCRIBE_PROMPT.format(
        name=t.display, sents="\n".join(f"- {s}" for s in t.sentences)
    )
    if t.signals.is_vendor_root:
        prompt += VENDOR_HINT.format(name=t.display)
    try:
        raw = await _query(llm, prompt, cfg)
    except Exception as exc:  # noqa: BLE001 — 한 이름 실패가 배치 전체를 막지 않게
        logger.warning("taxonomy assign: describe %s failed: %r", t.display, exc)
        return None, 1
    data = describe_result(raw)
    if data is None:
        logger.warning("taxonomy assign: describe %s returned unreadable JSON", t.display)
        return None, 1
    kind = str(data.get("kind") or "").strip()[:120] or None
    desc = str(data.get("description_ko") or "").strip()[:200] or None
    role_en = str(data.get("one_liner") or "").strip()[:400] or None
    if not _truthy(data.get("is_technology"), True):
        reason = _NOT_TECH_KIND.get(str(data.get("not_tech_kind") or "").lower(), NotTechReason.LLM)
        return (
            ProfileWrite(
                verdict=TechVerdict.NOT_TECHNOLOGY, not_tech_reason=reason, kind=kind,
                description=desc, role_en=role_en, model=model_name, **base,
            ),
            1,
        )
    verdict = TechVerdict.TECHNOLOGY if _truthy(data.get("enough"), True) else TechVerdict.INSUFFICIENT
    if verdict == TechVerdict.INSUFFICIENT:
        # 근거 없는 추측 설명은 저장하지 않는다(문서가 늘면 다시 설명한다).
        kind = desc = role_en = None
    return (
        ProfileWrite(
            verdict=verdict, kind=kind, description=desc, role_en=role_en,
            model=model_name, **base,
        ),
        1,
    )


def _evidence_dict(
    t: TargetInput,
    prof: ProfileWrite,
    cur: Mapping[str, str | None],
    *,
    reason: str,
    round_no: int,
) -> dict[str, Any]:
    return {
        "neighbors": [
            {
                "name": nb.key,
                "count": nb.count,
                "branch": cur.get(nb.key),
                "examples": list(nb.examples),
            }
            for nb in t.neighbors
        ],
        "isa": list(t.isa[:2]),
        "external_tags": list(t.external_tags[:6]),
        "sentence_count": len(t.sentences),
        "description": prof.description,
        "kind": prof.kind,
        "verdict": prof.verdict.value,
        "llm_reason": reason,
        "round": round_no,
    }


async def _assign_one(
    t: TargetInput,
    prof: ProfileWrite,
    leaves: Sequence[Leaf],
    llm: Any,
    cfg: TaxonomyConfig,
    cur: Mapping[str, str | None],
    round_no: int,
) -> tuple[AssignmentWrite | None, int]:
    """(배정 | None=LLM 인프라 실패, 호출 수). -1(맞는 갈래 없음)은 정상 미분류다."""
    one_liner = prof.role_en or prof.description or "(not enough source sentences)"
    prompt = ASSIGN_PROMPT.format(
        name=t.display,
        one_liner=one_liner,
        kind=prof.kind or "unknown",
        ev="\n".join(evidence_lines(t, cur)) or "- (none)",
        leaves="\n".join(f"{lf.index}. {lf.top} > {lf.name}: {lf.description}" for lf in leaves),
    )
    try:
        raw = await _query(llm, prompt, cfg)
    except Exception as exc:  # noqa: BLE001 — 한 이름 실패가 배치 전체를 막지 않게
        logger.warning("taxonomy assign: assign %s failed: %r", t.display, exc)
        return None, 1
    idx, reason, ok = parse_leaf_index(raw, len(leaves))
    if not ok:
        logger.warning("taxonomy assign: assign %s returned unreadable JSON", t.display)
        return None, 1
    evidence = _evidence_dict(t, prof, cur, reason=reason, round_no=round_no)
    if idx is None:
        return (
            AssignmentWrite(
                entity_id=t.entity_id, key=t.key, display=t.display, branch_id=None,
                leaf_path=None,
                unclassified_reason=UnclassifiedReason.NO_FIT,
                evidence=evidence,
            ),
            1,
        )
    leaf = leaves[idx]
    return (
        AssignmentWrite(
            entity_id=t.entity_id, key=t.key, display=t.display, branch_id=leaf.branch_id,
            leaf_path=leaf.path, unclassified_reason=None, evidence=evidence,
        ),
        1,
    )
