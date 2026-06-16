"""Route tests for 🧬 기술 계보 section on /item/{id} (ARG-159)."""
from __future__ import annotations

import uuid

from starlette.testclient import TestClient

from argos.models.tech_succession import RelationType
from argos.web.app import _get_session, build_web_app
from argos.web.services.detail import GenealogyEntry, ItemDetailView


def _view_with(
    *,
    predecessors: list[GenealogyEntry] | None = None,
    successors: list[GenealogyEntry] | None = None,
) -> ItemDetailView:
    return ItemDetailView(
        id=uuid.uuid4(),
        title="Anchor item",
        source_url="https://example.com/anchor",
        image_url=None,
        summary=None,
        category=None,
        trust_score=None,
        published_at=None,
        predecessors=predecessors or [],
        successors=successors or [],
    )


def _client(monkeypatch, view: ItemDetailView) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_fetch_item_detail(session, item_id):
        return view

    monkeypatch.setattr("argos.web.app.fetch_item_detail", _fake_fetch_item_detail)
    return TestClient(app)


def test_genealogy_section_omitted_when_no_predecessors_or_successors(monkeypatch):
    view = _view_with()
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "기술 계보" not in body
    assert "detail-genealogy" not in body


def test_genealogy_renders_predecessor_with_reasoning(monkeypatch):
    pred_id = uuid.uuid4()
    view = _view_with(
        predecessors=[
            GenealogyEntry(
                tech_id=pred_id,
                title="GPT-4o",
                relation_type=RelationType.REPLACE,
                reasoning="Replaced by next-gen multimodal stack.",
            )
        ]
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "기술 계보" in body
    assert "선행" in body
    assert "GPT-4o" in body
    assert "Replace" in body
    assert "Replaced by next-gen multimodal stack." in body
    assert f"/item/{pred_id}" in body
    # PascalCase badge class.
    assert "genealogy-relation--replace" in body


def test_genealogy_renders_successor_without_reasoning(monkeypatch):
    succ_id = uuid.uuid4()
    view = _view_with(
        successors=[
            GenealogyEntry(
                tech_id=succ_id,
                title="LangChain-next",
                relation_type=RelationType.ENHANCE,
                reasoning=None,
            )
        ]
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "후속" in body
    assert "LangChain-next" in body
    assert "Enhance" in body
    assert f"/item/{succ_id}" in body
    assert "genealogy-relation--enhance" in body
    # No reasoning paragraph should be rendered.
    assert "genealogy-reasoning" not in body


def test_genealogy_renders_both_groups(monkeypatch):
    view = _view_with(
        predecessors=[
            GenealogyEntry(
                tech_id=uuid.uuid4(),
                title="ParentA",
                relation_type=RelationType.FORK,
                reasoning="Fork from original design",
            )
        ],
        successors=[
            GenealogyEntry(
                tech_id=uuid.uuid4(),
                title="ChildB",
                relation_type=RelationType.ENHANCE,
                reasoning="Adds streaming",
            )
        ],
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "선행" in body
    assert "후속" in body
    assert "ParentA" in body
    assert "ChildB" in body
    assert "Fork" in body
    assert "Enhance" in body
