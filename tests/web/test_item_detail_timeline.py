"""Route tests for the unified timeline on /item/{id} for Keep assets (ARG-208).

Keep assets replace the older track_history-derived subsections (새 신호 /
최근 변화) with the same `_timeline.html` fragment the portfolio card
accordion uses. Non-asset items must render exactly as before.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.models.tech_succession import RelationType
from argos.models.user_asset import AssetStatus
from argos.web.app import _get_session, build_web_app
from argos.web.services.detail import (
    GenealogyEntry,
    HistoryEntry,
    ItemDetailView,
    SignalAlert,
    SimilarItem,
)
from argos.web.services.timeline import TimelineEvent


def _view_with(
    *,
    timeline: list[TimelineEvent] | None = None,
    status: AssetStatus | None = None,
    asset_id: uuid.UUID | None = None,
    similar: list[SimilarItem] | None = None,
    signal_alerts: list[SignalAlert] | None = None,
    related_history: list[HistoryEntry] | None = None,
    successors: list[GenealogyEntry] | None = None,
) -> ItemDetailView:
    return ItemDetailView(
        id=uuid.uuid4(),
        title="Anchor",
        source_url="https://example.com/anchor",
        image_url=None,
        summary=None,
        category=None,
        trust_score=None,
        published_at=None,
        status=status,
        asset_id=asset_id,
        similar=similar or [],
        signal_alerts=signal_alerts or [],
        related_history=related_history or [],
        timeline=timeline or [],
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


def test_keep_asset_renders_unified_timeline_and_hides_old_subsections(monkeypatch):
    """AC1: a Keep item shows the unified timeline; the old 최근 변화 /
    새 신호 subsections must not appear at all, even as empty markers."""
    matched_id = uuid.uuid4()
    view = _view_with(
        status=AssetStatus.KEEP,
        asset_id=uuid.uuid4(),
        timeline=[
            TimelineEvent(
                kind="signal",
                changed_at=datetime(2026, 6, 12, 8, 0, tzinfo=timezone.utc),
                title="Mistral-Embed v2",
                link_tech_id=matched_id,
                changed_from=None,
                changed_to=None,
                relation_type=None,
                reasoning=None,
                label="새 신호: Mistral-Embed v2",
            )
        ],
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert 'class="timeline"' in body
    assert "Mistral-Embed v2" in body
    assert f"/item/{matched_id}" in body
    # Old track_history-derived subsections must be entirely gone (their
    # unified timeline replacement's own event label can legitimately reuse
    # the "새 신호:" phrase, so we assert on the subsection markup/heading
    # rather than the plain substring).
    assert "signals-history" not in body
    assert "signals-new" not in body
    assert "최근 변화" not in body
    assert "signals-subsection__label" not in body


def test_keep_asset_still_shows_similarity_subsection(monkeypatch):
    """Similarity recommendations must keep rendering for Keep assets too."""
    sim_id = uuid.uuid4()
    view = _view_with(
        status=AssetStatus.KEEP,
        asset_id=uuid.uuid4(),
        similar=[SimilarItem(tech_id=sim_id, title="LangChain-next")],
        timeline=[
            TimelineEvent(
                kind="succession",
                changed_at=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
                title="Successor Tech",
                link_tech_id=uuid.uuid4(),
                changed_from=None,
                changed_to=None,
                relation_type=RelationType.ENHANCE,
                reasoning="Adds streaming",
                label="Enhance: Successor Tech",
            )
        ],
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "유사 신호" in body
    assert "LangChain-next" in body
    assert 'class="timeline"' in body
    assert "Successor Tech" in body


def test_non_asset_item_never_renders_timeline_markup(monkeypatch):
    """AC2: non-asset items keep the existing similarity/genealogy layout;
    item.timeline stays empty so the old signals wrapper renders unchanged."""
    tech_id = uuid.uuid4()
    view = _view_with(
        status=None,
        asset_id=None,
        related_history=[
            HistoryEntry(
                changed_from="Tracking",
                changed_to="Keep",
                changed_at=datetime(2026, 6, 10, 9, 30, tzinfo=timezone.utc),
                tech_id=tech_id,
                tech_title="Claude Opus 4.7",
            )
        ],
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert 'class="timeline"' not in body
    assert "최근 변화" in body
    assert "signals-history" in body


# ------------------------------------------------------------------ #
# ARG-209 — detail-page handoff banner (Replace successor)
# ------------------------------------------------------------------ #

def test_detail_page_renders_handoff_banner_for_replace_successor(monkeypatch):
    """A Keep item with a Replace successor gets the handoff banner in its
    action bar — derived from item.successors (no new query, AC1/AC3)."""
    asset_id = uuid.uuid4()
    succ_id = uuid.uuid4()
    view = _view_with(
        status=AssetStatus.KEEP,
        asset_id=asset_id,
        successors=[
            GenealogyEntry(
                tech_id=succ_id,
                title="Next-Gen Model",
                relation_type=RelationType.REPLACE,
                reasoning="superseded",
            )
        ],
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "handoff-banner" in body
    assert "Next-Gen Model" in body
    assert "이어받기" in body
    assert (
        f'hx-post="/assets/{asset_id}/handoff'
        f'?successor_tech_id={succ_id}&context=detail"' in body
    )
    # The handoff swaps the whole detail action area (banner + bar) so the two
    # can't drift out of sync after the predecessor is archived (ARG-209).
    assert f'hx-target="#detail-actions-{view.id}"' in body


def test_detail_page_omits_handoff_banner_for_enhance_successor(monkeypatch):
    """A Keep item whose only successor is Enhance (not Replace) gets no
    banner — that relation only surfaces the timeline's Keep button (AC3)."""
    view = _view_with(
        status=AssetStatus.KEEP,
        asset_id=uuid.uuid4(),
        successors=[
            GenealogyEntry(
                tech_id=uuid.uuid4(),
                title="Enhanced Variant",
                relation_type=RelationType.ENHANCE,
                reasoning=None,
            )
        ],
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    assert "handoff-banner" not in resp.text


def test_detail_page_omits_handoff_banner_for_non_kept_item(monkeypatch):
    """A Replace successor on a non-Keep item (e.g. Archived, Tracking-only,
    or untriaged) is not an active "hand off my Keep asset" situation, so no
    banner renders even though a Replace successor exists."""
    view = _view_with(
        status=AssetStatus.ARCHIVED,
        asset_id=uuid.uuid4(),
        successors=[
            GenealogyEntry(
                tech_id=uuid.uuid4(),
                title="Next-Gen Model",
                relation_type=RelationType.REPLACE,
                reasoning=None,
            )
        ],
    )
    client = _client(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    assert "handoff-banner" not in resp.text
