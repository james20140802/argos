"""자동 갈래 배정의 DB 읽기·쓰기 — ARG-313.

판단은 ``tech_classify``(DB 모름)가 하고, 여기서는 (1) 대상 이름을 고르고 입력을
모으고, (2) 계산된 ``AssignPlan``을 ``technology_profiles``·``role_assignments``에
쓴다. 쓰는 테이블은 그 둘뿐이다 — ``entities``·``document_entities``·
``entity_aliases``는 읽기만 하고, 갈래 확정본(``role_branches``)은
``taxonomy_store.load_confirmed``로 읽기만 한다(``test_taxonomy_isolation``).

대상은 **대표 이름**이다(별칭은 대표로 모은다). 문서 수도 별칭 문서를 합쳐 센다.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_aliases import find_roots, lowercase_ratios, trusted_vendors
from argos.brain.entity_frequency import common_names
from argos.brain.entity_names import canonical_name
from argos.brain.taxonomy import Tree, iter_branches
from argos.brain.taxonomy_draft import extract_sentences
from argos.brain.tech_classify import (
    AssignPlan,
    ExistingProfile,
    ScreenSignals,
    TargetInput,
)
from argos.brain.tech_evidence import PatternEvidence, name_patterns
from argos.config import AliasConfig, TaxonomyConfig, settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.entity_alias import EntityAlias
from argos.models.role_assignment import AssignmentSource, RoleAssignment
from argos.models.tech_item import TechItem
from argos.models.technology_profile import TechnologyProfile, TechVerdict
from argos.services.technology_names import pick_display_names

logger = logging.getLogger(__name__)

DOCS_PER_TARGET = 30
MAX_SENTENCES = 15
MAX_SENTENCE_CHARS = 400
MAX_NEIGHBORS = 5


class ClassifyTablesMissing(RuntimeError):
    """``technology_profiles``/``role_assignments``가 없다 — ``alembic upgrade head``가 먼저."""


async def tables_exist(session: AsyncSession) -> bool:
    found = await session.scalar(
        text(
            "SELECT to_regclass('public.technology_profiles') IS NOT NULL "
            "AND to_regclass('public.role_assignments') IS NOT NULL"
        )
    )
    return found is True


# --------------------------------------------------------------------- #
# 대상 고르기
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class NameRow:
    """대표 이름 하나의 현재 상태."""

    entity_id: Any
    key: str
    name: str
    doc_count: int
    profile: ExistingProfile | None = None
    assignment_source: AssignmentSource | None = None


def needs_refine(row: NameRow, cfg: TaxonomyConfig) -> bool:
    """문서가 충분히 늘었으면 다시 설명한다(기사 하나의 틀린 설명이 굳지 않게)."""
    if row.profile is None:
        return True
    before = row.profile.described_doc_count
    return row.doc_count >= before * cfg.refine_ratio or row.doc_count >= before + cfg.refine_add


def select_targets(
    rows: Iterable[NameRow],
    cfg: TaxonomyConfig,
    *,
    all_names: bool = False,
    limit: int | None = None,
) -> list[tuple[NameRow, bool]]:
    """(대상, 설명 필요 여부) 목록. 문서 많은 순. 순수 함수.

    - 문서 수 < ``min_documents``와 ``Manual`` 배정은 대상이 아니다.
    - 하루 배치: 새 이름(설명 없음), 문서가 늘어 다시 설명할 이름, 설명은 있는데
      배정 행이 없는 기술(갈래가 지워진 경우)만.
    - ``all_names``(소급): 위 조건 없이 전부 다시 배정한다(설명은 필요할 때만).
    """
    out: list[tuple[NameRow, bool]] = []
    for row in rows:
        if row.doc_count < cfg.min_documents:
            continue
        if row.assignment_source == AssignmentSource.MANUAL:
            continue
        describe = needs_refine(row, cfg)
        if all_names:
            out.append((row, describe))
            continue
        if describe:
            out.append((row, True))
        elif (
            row.profile is not None
            and row.profile.verdict != TechVerdict.NOT_TECHNOLOGY
            and row.assignment_source is None
        ):
            out.append((row, False))
    out.sort(key=lambda rd: (-rd[0].doc_count, rd[0].key))
    return out if limit is None else out[:limit]


# --------------------------------------------------------------------- #
# 읽기
# --------------------------------------------------------------------- #
async def _alias_map(session: AsyncSession) -> dict[Any, Any]:
    rows = (
        await session.execute(select(EntityAlias.alias_entity_id, EntityAlias.canonical_entity_id))
    ).all()
    return {a: c for a, c in rows}


async def load_name_rows(session: AsyncSession) -> list[NameRow]:
    """모든 대표 이름의 문서 수(별칭 문서 합산)와 설명·배정 상태. 읽기 전용."""
    alias_of = await _alias_map(session)
    entities = {
        eid: (key, name)
        for eid, key, name in (
            await session.execute(select(Entity.id, Entity.normalized_key, Entity.name))
        ).all()
    }
    docs_of: dict[Any, set] = defaultdict(set)
    for eid, doc in (
        await session.execute(select(DocumentEntity.entity_id, DocumentEntity.tech_item_id))
    ).all():
        docs_of[alias_of.get(eid, eid)].add(doc)

    profiles = {
        p.entity_id: p
        for p in (await session.execute(select(TechnologyProfile))).scalars().all()
    }
    sources = {
        eid: AssignmentSource(src)
        for eid, src in (
            await session.execute(select(RoleAssignment.entity_id, RoleAssignment.source))
        ).all()
    }
    out: list[NameRow] = []
    for eid, docs in docs_of.items():
        if eid not in entities:
            continue
        key, name = entities[eid]
        p = profiles.get(eid)
        out.append(
            NameRow(
                entity_id=eid,
                key=key,
                name=name,
                doc_count=len(docs),
                profile=_existing(p) if p is not None else None,
                assignment_source=sources.get(eid),
            )
        )
    return out


def _existing(p: TechnologyProfile) -> ExistingProfile:
    return ExistingProfile(
        verdict=TechVerdict(p.verdict),
        not_tech_reason=p.not_tech_reason,
        kind=p.kind,
        description=p.description,
        role_en=p.role_en,
        described_doc_count=p.described_doc_count,
    )


async def current_assignments(session: AsyncSession, tree: Tree) -> dict[str, str | None]:
    """대표 키 -> 현재 갈래 경로("위 > 아래") 또는 None(미분류)."""
    paths = {
        b.id: f"{p.name} > {b.name}" for d, b, p in iter_branches(tree) if d == 2 and p and b.id
    }
    rows = (
        await session.execute(
            select(Entity.normalized_key, RoleAssignment.branch_id).join(
                RoleAssignment, RoleAssignment.entity_id == Entity.id
            )
        )
    ).all()
    return {key: (paths.get(str(bid)) if bid else None) for key, bid in rows}


def _ner_labels(sentences: Sequence[str], surfaces: Iterable[str]) -> dict[str, int]:
    """spaCy가 있으면 이름 언급의 라벨 수. 없거나 실패하면 빈 dict(그 단계를 건너뜀)."""
    try:
        from argos.brain.entity_spacy import load_pipeline

        pipeline = load_pipeline()
    except Exception:  # noqa: BLE001
        return {}
    if pipeline is None or not sentences:
        return {}
    wanted = {s.casefold() for s in surfaces}
    counts: dict[str, int] = {}
    try:
        for parsed in pipeline.pipe(list(sentences)):
            for ent in parsed.ents:
                if ent.text.casefold() in wanted:
                    counts[ent.label_] = counts.get(ent.label_, 0) + 1
    except Exception:  # noqa: BLE001
        return {}
    return counts


async def build_inputs(
    session: AsyncSession,
    selected: Sequence[tuple[NameRow, bool]],
    cfg: TaxonomyConfig,
    *,
    all_rows: Sequence[NameRow],
    external: Any | None = None,
    alias_cfg: AliasConfig | None = None,
) -> list[TargetInput]:
    """대상마다 문장·거르기 신호·패턴 이웃·외부 태그를 모은다. 읽기 전용."""
    if not selected:
        return []
    alias_cfg = alias_cfg or settings.user.aliases
    ed = settings.user.event_detection
    alias_of = await _alias_map(session)
    members_of: dict[Any, list[Any]] = defaultdict(list)
    for alias_id, canon_id in alias_of.items():
        members_of[canon_id].append(alias_id)
    ent_rows = (
        await session.execute(select(Entity.id, Entity.normalized_key, Entity.name))
    ).all()
    key_of = {eid: key for eid, key, _n in ent_rows}
    name_of = {eid: name for eid, _k, name in ent_rows}

    # 흔한 이름 · 회사 접두어 뿌리 (코퍼스 전체 통계)
    doc_counts = {r.key: r.doc_count for r in all_rows}
    named_docs = int(
        await session.scalar(select(func.count(func.distinct(DocumentEntity.tech_item_id)))) or 0
    )
    common = common_names(
        document_frequency=doc_counts,
        named_documents=named_docs,
        max_ratio=ed.entity_common_doc_ratio,
        min_documents=ed.entity_common_min_documents,
    )
    # 패턴·이웃에서 뺄 흔한 이름은 짧은 것(ai·api·llm)만 — codex·chatgpt도 흔한 이름이지만
    # 기술이라 나열 증거에 남아야 한다(screen과 같은 기준).
    common_short = frozenset(k for k in common if len(k) <= cfg.common_short_max_chars)
    roots = find_roots(key_of.values(), alias_cfg)
    not_tech = {
        r.key for r in all_rows if r.profile and r.profile.verdict == TechVerdict.NOT_TECHNOLOGY
    }
    canon_key = {eid: key_of[alias_of.get(eid, eid)] for eid in key_of if alias_of.get(eid, eid) in key_of}

    # 대상 문서
    docs_by_target: dict[Any, list[tuple[Any, str]]] = {}
    all_doc_ids: set = set()
    for row, _d in selected:
        ids = [row.entity_id, *members_of.get(row.entity_id, [])]
        docs = (
            await session.execute(
                select(
                    TechItem.id,
                    TechItem.title,
                    TechItem.summary,
                    TechItem.raw_content,
                    TechItem.created_at,
                )
                .join(DocumentEntity, DocumentEntity.tech_item_id == TechItem.id)
                .where(DocumentEntity.entity_id.in_(ids))
                .distinct()
                .order_by(TechItem.created_at.desc(), TechItem.id)
                .limit(DOCS_PER_TARGET)
            )
        ).all()
        docs_by_target[row.entity_id] = [
            (did, "\n".join(t for t in (title, summary, raw) if t))
            for did, title, summary, raw, _created in docs
        ]
        all_doc_ids.update(did for did, *_ in docs)

    # 패턴 증거: 문서마다 그 문서에 붙은 이름만 찾는다(공출현은 세지 않는다).
    names_in_doc: dict[Any, set[Any]] = defaultdict(set)
    if all_doc_ids:
        for did, eid in (
            await session.execute(
                select(DocumentEntity.tech_item_id, DocumentEntity.entity_id).where(
                    DocumentEntity.tech_item_id.in_(list(all_doc_ids))
                )
            )
        ).all():
            names_in_doc[did].add(eid)
    evidence = PatternEvidence()
    seen_docs: set = set()
    for docs in docs_by_target.values():
        for did, body in docs:
            if did in seen_docs:
                continue
            seen_docs.add(did)
            surfaces: dict[str, list[str]] = defaultdict(list)
            for eid in names_in_doc.get(did, ()):
                ck = canon_key.get(eid)
                if ck is None or ck in common_short:
                    continue
                surfaces[ck].append(name_of[eid])
            if len(surfaces) >= 1:
                evidence.add_text(body, name_patterns(surfaces))

    out: list[TargetInput] = []
    for row, describe in selected:
        texts = [body for _did, body in docs_by_target.get(row.entity_id, [])]
        member_ids = [row.entity_id, *members_of.get(row.entity_id, [])]
        members = [(key_of[m], name_of[m]) for m in member_ids if m in key_of]
        display, others = pick_display_names(members, row.key, texts) if members else (row.name, ())
        display = prefer_full_name(display, others, texts)
        sentences = extract_sentences(
            texts, display, limit=MAX_SENTENCES, max_chars=MAX_SENTENCE_CHARS
        )
        ratios = lowercase_ratios(texts, [row.key])
        vendor = row.key in roots and row.key in trusted_vendors(
            frozenset({row.key}), doc_counts, lambda keys: ratios, alias_cfg
        )
        signals = ScreenSignals(
            key=row.key,
            display=display,
            common=row.key in common,
            lowercase_ratio=ratios.get(row.key),
            is_vendor_root=vendor,
            ner_labels=_ner_labels(sentences, [display, *(n for _k, n in members)]) if describe else {},
            stored_name=row.name,
        )
        # 이웃도 대상과 같은 문턱을 넘은 이름만(한 문서짜리 잡음 "pi 4714" 등을 뺀다).
        neighbors = tuple(
            nb for nb in evidence.neighbors(row.key, limit=MAX_NEIGHBORS * 4)
            if nb.key not in not_tech
            and nb.key not in common_short
            and doc_counts.get(nb.key, 0) >= cfg.min_documents
        )[:MAX_NEIGHBORS]
        tags: tuple[str, ...] = ()
        # 설명을 재사용하는 회(--all)에도 모은다 — 외부 태그로 배정된 문장 부족 기술이
        # 다음 회에 근거 0으로 미분류가 되면 안 된다. 캐시(TTL)가 있어 재요청은 싸다.
        if external is not None:
            from argos.crawler.tech_external import links_for

            gh, hf = links_for(texts, display)
            if gh or hf:
                try:
                    tags = tuple(await external.tags_for(gh, hf))
                except Exception as exc:  # noqa: BLE001 — 외부 증거 실패는 그 증거만 비운다
                    logger.info("taxonomy assign: external evidence for %s failed: %r", display, exc)
        out.append(
            TargetInput(
                entity_id=row.entity_id,
                key=row.key,
                display=display,
                doc_count=row.doc_count,
                sentences=tuple(sentences),
                signals=signals,
                neighbors=neighbors,
                isa=tuple(evidence.isa.get(row.key, ())),
                external_tags=tags,
                existing=row.profile,
                needs_describe=describe,
            )
        )
    return out


def prefer_full_name(display: str, others: Sequence[str], texts: Sequence[str]) -> str:
    """표시 이름이 다른 표기의 조각이면("6 Astra" ⊂ "GPT-6 Astra") 긴 표기를 쓴다.

    ARG-312 별칭이 회사 접두어를 뗀 이름을 대표로 두면, 조각 표기는 긴 표기 안에서도
    잡혀 늘 더 많이 세어진다. 긴 표기가 본문에서 조각의 절반 이상 쓰였으면 그쪽이 LLM에
    넘길 이름으로 맞다. 순수 함수.
    """
    def count(name: str) -> int:
        pat = re.compile(
            rf"(?<![0-9A-Za-z]){re.escape(name)}(?![0-9A-Za-z])", re.IGNORECASE
        )
        return sum(len(pat.findall(t or "")) for t in texts)

    base = count(display)
    best = display
    for other in others:
        if len(other) > len(best) and display.casefold() in other.casefold():
            if base and count(other) * 2 >= base:
                best = other
    return best


# --------------------------------------------------------------------- #
# 쓰기
# --------------------------------------------------------------------- #
async def write_plan(session: AsyncSession, plan: AssignPlan) -> int:
    """계획을 쓴다. 커밋은 호출자 몫. ``Manual`` 배정 행은 바꾸거나 지우지 않는다."""
    if plan.is_empty:
        return 0
    if not await tables_exist(session):
        raise ClassifyTablesMissing(
            "technology_profiles/role_assignments 테이블이 없습니다. "
            "`uv run alembic upgrade head`를 먼저 실행하세요."
        )
    now = func.now()
    for p in plan.profiles:
        values = dict(
            verdict=p.verdict,
            not_tech_reason=p.not_tech_reason,
            kind=p.kind,
            description=p.description,
            role_en=p.role_en,
            described_doc_count=p.described_doc_count,
            described_at=now,
            model=p.model,
        )
        stmt = pg_insert(TechnologyProfile).values(id=uuid.uuid4(), entity_id=p.entity_id, **values)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[TechnologyProfile.entity_id],
                set_={**values, "updated_at": now},
            )
        )
    for a in plan.assignments:
        values = dict(
            branch_id=uuid.UUID(a.branch_id) if a.branch_id else None,
            evidence=a.evidence,
            unclassified_reason=a.unclassified_reason,
            source=AssignmentSource.AUTO,
            assigned_at=now,
            run_id=plan.run_id,
        )
        stmt = pg_insert(RoleAssignment).values(id=uuid.uuid4(), entity_id=a.entity_id, **values)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[RoleAssignment.entity_id],
                set_={**values, "updated_at": now},
                where=RoleAssignment.source == AssignmentSource.AUTO,
            )
        )
    if plan.unassign:
        await session.execute(
            delete(RoleAssignment).where(
                RoleAssignment.entity_id.in_([eid for eid, _d in plan.unassign]),
                RoleAssignment.source == AssignmentSource.AUTO,
            )
        )
    return len(plan.profiles) + len(plan.assignments) + len(plan.unassign)


# --------------------------------------------------------------------- #
# why / eval
# --------------------------------------------------------------------- #
async def resolve_canonical(session: AsyncSession, name: str) -> tuple[Any, str, str] | None:
    """어떤 표기든 -> (대표 entity id, 대표 키, 저장된 이름). 없으면 None."""
    row = (
        await session.execute(
            select(Entity.id, Entity.normalized_key, Entity.name).where(
                Entity.normalized_key == canonical_name(name)
            )
        )
    ).first()
    if row is None:
        return None
    canon = await session.scalar(
        select(EntityAlias.canonical_entity_id).where(EntityAlias.alias_entity_id == row[0])
    )
    if canon is None:
        return row[0], row[1], row[2]
    found = (
        await session.execute(
            select(Entity.id, Entity.normalized_key, Entity.name).where(Entity.id == canon)
        )
    ).first()
    return (found[0], found[1], found[2]) if found else (row[0], row[1], row[2])


@dataclass(frozen=True)
class Explanation:
    name: str
    key: str
    profile: TechnologyProfile | None
    assignment: RoleAssignment | None
    leaf_path: str | None


async def explain(session: AsyncSession, name: str, tree: Tree | None) -> Explanation | None:
    found = await resolve_canonical(session, name)
    if found is None:
        return None
    eid, key, stored = found
    profile = (
        await session.execute(select(TechnologyProfile).where(TechnologyProfile.entity_id == eid))
    ).scalar_one_or_none()
    assignment = (
        await session.execute(select(RoleAssignment).where(RoleAssignment.entity_id == eid))
    ).scalar_one_or_none()
    path = None
    if assignment is not None and assignment.branch_id and tree:
        for d, b, p in iter_branches(tree):
            if d == 2 and p and b.id == str(assignment.branch_id):
                path = f"{p.name} > {b.name}"
    return Explanation(stored, key, profile, assignment, path)


async def assigned_labels(
    session: AsyncSession, names: Iterable[str]
) -> dict[str, str | None]:
    """평가용: 정답 이름 -> 배정된 갈래 id(미분류 None). DB에 없는 이름은 빠진다."""
    out: dict[str, str | None] = {}
    for name in names:
        found = await resolve_canonical(session, name)
        if found is None:
            continue
        bid = await session.scalar(
            select(RoleAssignment.branch_id).where(RoleAssignment.entity_id == found[0])
        )
        out[name] = str(bid) if bid else None
    return out
