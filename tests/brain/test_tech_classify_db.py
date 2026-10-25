"""자동 갈래 배정 저장 왕복의 DB 통합 테스트 (ARG-313).

스크래치 DB(argos_test*)만 쓰고, Postgres가 없으면 통째로 skip(릴리즈 CI 대비).
이 모듈의 키는 전부 ``zzcls`` 접두어, 문서 URL은 고유 접두어를 달고 끝나면 지운다.
LLM은 가짜다.
"""
from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos.brain import taxonomy as tx
from argos.brain import tech_classify_store as store
from argos.brain.taxonomy_store import apply_plan, load_confirmed, plan_apply
from argos.brain.tech_classify import build_plan, leaves_of
from argos.config import TaxonomyConfig, settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.entity_alias import AliasRule, EntityAlias
from argos.models.role_assignment import AssignmentSource, RoleAssignment
from argos.models.role_branch import RoleBranch
from argos.models.tech_item import CategoryType, TechItem
from argos.models.technology_profile import TechnologyProfile, TechVerdict
from tests.conftest import db_reachable as _db_reachable

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-313-classify-test.example.com/"
_KEY_LIKE = "zzcls%"
CFG = TaxonomyConfig(min_documents=2)


@pytest.fixture(scope="module", autouse=True)
def _require_db():
    if not _db_reachable(_DB_URL):
        pytest.skip("pgvector DB not reachable — skipping ARG-313 classify DB test")


async def _cleanup(factory):
    async with factory() as session:
        await session.execute(delete(TechItem).where(TechItem.source_url.like(f"{_URL_PREFIX}%")))
        await session.execute(delete(Entity).where(Entity.normalized_key.like(_KEY_LIKE)))
        await session.execute(delete(RoleBranch))
        await session.commit()


@pytest.fixture
async def factory():
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    f = async_sessionmaker(bind=engine, expire_on_commit=False)
    await _cleanup(f)
    try:
        yield f
    finally:
        await _cleanup(f)
        await engine.dispose()


TEXT = (
    "Developers compare Zzclscode, Zzclscursor, and Zzclspilot daily. "
    "Zzclscode is a terminal coding agent for software teams. "
    "Zzclscode writes and edits code in your repository. "
    "Zzclscode runs tests and fixes failures automatically."
)


async def _seed(session):
    ids = {}
    for key in ("zzclscode", "zzclscursor", "zzclspilot", "zzclsacme zzclspilot", "zzclsmanual"):
        ids[key] = uuid.uuid4()
        session.add(Entity(id=ids[key], name=key.title(), normalized_key=key))
    await session.flush()
    # 별칭: "zzclsacme zzclspilot" -> "zzclspilot" (대표로 모은다)
    session.add(
        EntityAlias(
            alias_entity_id=ids["zzclsacme zzclspilot"],
            canonical_entity_id=ids["zzclspilot"],
            rule=AliasRule.VENDOR_PREFIX,
            evidence={},
        )
    )
    items = []
    for i in range(3):
        item = TechItem(
            id=uuid.uuid4(), title=f"ARG-313 classify {i}", source_url=f"{_URL_PREFIX}{i}",
            raw_content=TEXT, category=CategoryType.ALPHA,
        )
        session.add(item)
        items.append(item.id)
    await session.flush()
    links = {
        "zzclscode": items,
        "zzclscursor": items[:2],
        "zzclspilot": items[:1],
        "zzclsacme zzclspilot": items[1:2],
        "zzclsmanual": items,
    }
    for key, its in links.items():
        for it in its:
            session.add(DocumentEntity(id=uuid.uuid4(), tech_item_id=it, entity_id=ids[key]))
    await session.flush()
    return ids


class FakeLLM:
    def __init__(self, leaf_index):
        self.leaf_index = leaf_index
        self.calls = 0

    async def query(self, role, prompt, **kw):
        self.calls += 1
        if prompt.startswith("Below are sentences"):
            return json.dumps({
                "is_technology": True, "not_tech_kind": "none", "enough": True,
                "kind": "coding agent", "one_liner": "Coding agent.", "description_ko": "코딩 에이전트",
            })
        return json.dumps({"leaf_index": self.leaf_index, "reason": "listed with peers"})


async def test_assign_round_trip_respects_manual_and_aliases(factory):
    async with factory() as session:
        await apply_plan(session, plan_apply(None, tx.SPIKE_SEED))
        ids = await _seed(session)
        await session.commit()

    async with factory() as session:
        tree = await load_confirmed(session)
        leaves = leaves_of(tree)
        coding = next(lf for lf in leaves if lf.name.startswith("코딩"))
        # 사람이 고친 배정(Manual)이 이미 있다.
        session.add(RoleAssignment(
            entity_id=ids["zzclsmanual"], branch_id=uuid.UUID(leaves[0].branch_id),
            evidence={}, source=AssignmentSource.MANUAL,
        ))
        await session.commit()

    async with factory() as session:
        rows = [r for r in await store.load_name_rows(session) if r.key.startswith("zzcls")]
        by = {r.key: r for r in rows}
        assert "zzclsacme zzclspilot" not in by  # 별칭은 대표로 모인다
        assert by["zzclspilot"].doc_count == 2
        assert by["zzclsmanual"].assignment_source == AssignmentSource.MANUAL
        selected = store.select_targets(rows, CFG)
        assert {r.key for r, _d in selected} == {"zzclscode", "zzclscursor", "zzclspilot"}
        inputs = await store.build_inputs(session, selected, CFG, all_rows=rows)
        code = next(t for t in inputs if t.key == "zzclscode")
        assert len(code.sentences) >= 3
        assert {nb.key for nb in code.neighbors} == {"zzclscursor", "zzclspilot"}
        assert code.isa and code.isa[0].startswith("terminal coding agent")
        plan = await build_plan(inputs, leaves, FakeLLM(coding.index), CFG,
                                current=await store.current_assignments(session, tree))
        await store.write_plan(session, plan)
        await session.commit()

    async with factory() as session:
        got = dict((await session.execute(
            select(Entity.normalized_key, RoleAssignment.branch_id)
            .join(RoleAssignment, RoleAssignment.entity_id == Entity.id)
            .where(Entity.normalized_key.like(_KEY_LIKE))
        )).all())
        assert str(got["zzclscode"]) == coding.branch_id
        assert str(got["zzclsmanual"]) == leaves[0].branch_id  # Manual 그대로
        prof = (await session.execute(
            select(TechnologyProfile).where(TechnologyProfile.entity_id == ids["zzclscode"])
        )).scalar_one()
        assert prof.verdict == TechVerdict.TECHNOLOGY and prof.described_doc_count == 3
        exp = await store.explain(session, "Zzclscode", tree)
        assert exp.leaf_path == coding.path
        assert exp.assignment.evidence["neighbors"]
        # 별칭 표기로 물어도 대표의 배정이 나온다.
        labels = await store.assigned_labels(session, ["Zzclsacme Zzclspilot", "nope-zz"])
        assert labels == {"Zzclsacme Zzclspilot": coding.branch_id}

        # 다음 날: 새 것이 없으면 대상이 없다(멱등).
        rows = [r for r in await store.load_name_rows(session) if r.key.startswith("zzcls")]
        assert store.select_targets(rows, CFG) == []

        # 기술이 아니게 된 이름: 자동 배정만 지우고 Manual 행은 남긴다.
        from argos.brain.tech_classify import AssignPlan

        await store.write_plan(session, AssignPlan(
            run_id=uuid.uuid4(),
            unassign=((ids["zzclscode"], "Zzclscode"), (ids["zzclsmanual"], "Zzclsmanual")),
        ))
        await session.commit()
        left = set((await session.execute(
            select(Entity.normalized_key)
            .join(RoleAssignment, RoleAssignment.entity_id == Entity.id)
            .where(Entity.normalized_key.like(_KEY_LIKE))
        )).scalars())
        assert "zzclscode" not in left and "zzclsmanual" in left


async def test_external_tags_are_collected_when_the_profile_is_reused(factory, monkeypatch):
    """--all이 기존 프로필을 재사용(설명 안 함)해도 외부 태그 증거는 다시 모은다(Codex P2).
    안 그러면 외부 태그로 배정됐던 문장 부족 기술이 다음 --all에서 근거 0 -> 미분류로 풀린다."""
    from argos.crawler import tech_external

    async with factory() as session:
        await _seed(session)
        await session.commit()

    monkeypatch.setattr(
        tech_external, "links_for",
        lambda texts, name: (["acme/zzclscode"], []) if name.lower() == "zzclscode" else ([], []),
    )

    class External:
        def __init__(self):
            self.asked = []

        async def tags_for(self, gh, hf):
            self.asked.append(list(gh))
            return ["gh:coding-agent"]

    ext = External()
    async with factory() as session:
        rows = [r for r in await store.load_name_rows(session) if r.key == "zzclscode"]
        inputs = await store.build_inputs(
            session, [(rows[0], False)], CFG, all_rows=rows, external=ext,
        )
    (code,) = inputs
    assert code.needs_describe is False
    assert code.external_tags == ("gh:coding-agent",)
    assert ext.asked == [["acme/zzclscode"]]
