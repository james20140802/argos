"""대표 이름·다른 이름 조회 — ARG-312. 순수 부분은 DB 없이, 조회는 DB가 있을 때만."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from argos import cli
from argos.brain import alias_store
from argos.config import settings
from argos.models.document_entity import DocumentEntity
from argos.models.entity import Entity
from argos.models.entity_alias import AliasRule, EntityAlias
from argos.models.tech_item import CategoryType, TechItem
from argos.services import technology_names as tn
from tests.conftest import db_reachable as _db_reachable


def test_display_name_is_the_most_common_surface_not_the_first_stored_one():
    members = [("openclaw", "Openclaw")]
    texts = [
        "OpenClaw is out. OpenClaw agents. Openclaw once.",
        "I tried OpenClaw today.",
    ]
    display, others = tn.pick_display_names(members, "openclaw", texts)
    assert display == "OpenClaw"
    assert "Openclaw" in others
    assert "OpenClaw" not in others


def test_other_names_include_alias_surfaces_and_stored_names():
    members = [("codex", "Codex"), ("openai codex", "Openai Codex")]
    texts = ["OpenAI Codex ships. Codex is fast. Codex again. Codex!"]
    display, others = tn.pick_display_names(members, "codex", texts)
    assert display == "Codex"
    assert "OpenAI Codex" in others


def test_display_falls_back_to_the_canonical_stored_name_without_text():
    display, others = tn.pick_display_names(
        [("codex", "Codex"), ("openai codex", "Openai Codex")], "codex", []
    )
    assert display == "Codex"
    assert others == ("Openai Codex",)


def test_surface_pattern_respects_word_boundaries_and_symbols():
    assert tn.surface_pattern("c").search("C++ is old") is None  # c++ 는 c가 아니다
    assert tn.surface_pattern("c++").search("I like C++ a lot")
    assert tn.surface_pattern("node js").search("Node.js rocks")
    assert tn.surface_pattern("code").search("Barcode") is None


def test_version_numbers_are_not_conflated_with_digit_runs():
    texts = ["Opus 5.5 vs Opus 55. Opus 55 is unrelated. Opus 55."]
    display, others = tn.pick_display_names([("opus 5.5", "Opus 5.5")], "opus 5.5", texts)
    assert display == "Opus 5.5"
    assert "Opus 55" not in others


def test_version_match_does_not_end_or_start_mid_number():
    pat = tn.surface_pattern("opus 5.5")
    assert pat.search("Opus 5.5 is out")
    assert pat.search("Opus 5.5.") and pat.search("Opus 5.5, ok")
    assert pat.search("Opus 5.55") is None
    assert pat.search("Opus 5.5.1") is None
    assert pat.search("Opus 55") is None
    assert tn.surface_pattern("gpt 6").search("GPT-6 ships")
    assert tn.surface_pattern("5.5").search("v1.5.5") is None


def test_internal_name_symbols_are_kept_in_compound_names():
    display, aliases = tn.pick_display_names(
        [("c++ cli", "C++/CLI")],
        "c++ cli",
        ["C++/CLI is .NET. C CLI is different. C CLI."],
    )
    assert display == "C++/CLI"
    assert "C CLI" not in aliases
    assert tn.surface_pattern("asp .net").search("ASP .NET Core")
    assert not tn.surface_pattern("asp .net").search("ASP NET")
    assert tn.surface_pattern("f# interactive").search("F# Interactive")
    assert not tn.surface_pattern("f interactive").search("F# Interactive")


def test_cli_show_prints_display_and_other_names(monkeypatch, capsys):
    session = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: ctx)
    monkeypatch.setattr(alias_store, "alias_table_exists", AsyncMock(return_value=True))
    monkeypatch.setattr(
        tn,
        "names_for",
        AsyncMock(return_value=tn.TechnologyNames("OpenClaw", ("Openclaw",), "openclaw", ("openclaw",))),
    )
    assert cli.main(["alias-names", "--show", "Openclaw"]) == 0
    assert "OpenClaw — 다른 이름: Openclaw" in capsys.readouterr().out


def test_cli_show_unknown_name_fails(monkeypatch, capsys):
    session = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: ctx)
    monkeypatch.setattr(alias_store, "alias_table_exists", AsyncMock(return_value=True))
    monkeypatch.setattr(tn, "names_for", AsyncMock(return_value=None))
    assert cli.main(["alias-names", "--show", "nope"]) != 0
    assert "저장된 이름에 없습니다" in capsys.readouterr().err


def test_cli_show_without_the_alias_table_explains_how_to_migrate(monkeypatch, capsys):
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(cli, "AsyncSessionLocal", lambda: ctx)
    monkeypatch.setattr(alias_store, "alias_table_exists", AsyncMock(return_value=False))
    assert cli.main(["alias-names", "--show", "codex"]) != 0
    assert "alembic upgrade head" in capsys.readouterr().err


# ----- DB 조회 ---------------------------------------------------------------- #

_DB_URL: str = settings.database_url
_URL_PREFIX = "https://arg-312-technology-names-test.example.com/"


@pytest.fixture
async def session_factory():
    if not _db_reachable(_DB_URL):
        pytest.skip("pgvector DB not reachable — skipping ARG-312 technology_names DB test")
    engine = create_async_engine(_DB_URL, poolclass=NullPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with factory() as session:
            await session.execute(delete(TechItem).where(TechItem.source_url.like(f"{_URL_PREFIX}%")))
            await session.execute(delete(Entity).where(Entity.normalized_key.like("zztn%")))
            await session.commit()
        await engine.dispose()


async def test_names_for_follows_alias_rows_and_excludes_other_versions(session_factory):
    async with session_factory() as session:
        ids = {k: uuid.uuid4() for k in ["zztn opus 5", "zztn claude opus 5", "zztn opus x"]}
        for key, entity_id in ids.items():
            session.add(Entity(id=entity_id, name=key.title(), normalized_key=key))
        item = TechItem(
            id=uuid.uuid4(),
            title="ARG-312 names",
            source_url=f"{_URL_PREFIX}1",
            raw_content="Zztn Claude Opus 5 is here. Zztn Opus 5 too. Zztn Opus 5 again.",
            category=CategoryType.ALPHA,
        )
        session.add(item)
        await session.flush()
        for key in ("zztn opus 5", "zztn claude opus 5"):
            session.add(DocumentEntity(id=uuid.uuid4(), tech_item_id=item.id, entity_id=ids[key]))
        session.add(
            EntityAlias(
                alias_entity_id=ids["zztn claude opus 5"],
                canonical_entity_id=ids["zztn opus 5"],
                rule=AliasRule.VENDOR_PREFIX,
                evidence={},
            )
        )
        await session.commit()

    async with session_factory() as session:
        from_alias = await tn.names_for(session, "ZZTN Claude Opus 5")
        from_canonical = await tn.names_for(session, "zztn opus 5")
        assert from_alias is not None and from_canonical is not None
        assert from_alias == from_canonical
        assert from_canonical.display == "Zztn Opus 5"
        assert "Zztn Claude Opus 5" in from_canonical.aliases
        assert set(from_canonical.member_keys) == {"zztn opus 5", "zztn claude opus 5"}
        # 다른 이름(별칭 없는 다른 항목)은 같은 묶음에 나오지 않는다
        assert "zztn opus x" not in from_canonical.member_keys
        alone = await tn.names_for(session, "zztn opus x")
        assert alone is not None and alone.member_keys == ("zztn opus x",)
        assert await tn.names_for(session, "zztn does not exist") is None
