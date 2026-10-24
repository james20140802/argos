"""Route + template tests for the 상세 보기 screen (ARG-158).

These exercise GET /item/{id} without a live database: the per-request
session dependency is overridden and ``fetch_item_detail`` is monkeypatched
to return a canned ``ItemDetailView`` (or ``None`` for the 404 path). This
keeps the tests runnable on release.yml CI (no Postgres).
"""
from __future__ import annotations

import uuid

from starlette.testclient import TestClient

from argos.models.tech_item import CategoryType
from argos.models.user_asset import AssetStatus
from argos.web.app import _get_session, build_web_app
from argos.web.services.detail import ItemDetailView, ItemEvent
from argos.web.services.event_sections import EventSection


def _view(
    *,
    title: str = "GPT-5 launches with multimodal reasoning",
    image_url: str | None = None,
    summary: str | None = "It's the next milestone.",
    event: ItemEvent | None = None,
    category: CategoryType | None = CategoryType.MAINSTREAM,
    trust_score: float | None = 0.82,
    source_url: str = "https://example.com/gpt5",
    status: AssetStatus | None = None,
    asset_id: uuid.UUID | None = None,
    trust_rubric: dict | None = None,
    corroboration_count: int | None = None,
    has_body: bool = True,
) -> ItemDetailView:
    return ItemDetailView(
        id=uuid.uuid4(),
        title=title,
        source_url=source_url,
        image_url=image_url,
        summary=summary,
        event=event,
        category=category,
        trust_score=trust_score,
        published_at=None,
        status=status,
        asset_id=asset_id,
        trust_rubric=trust_rubric,
        corroboration_count=corroboration_count,
        has_body=has_body,
    )


def _client_with_detail(
    monkeypatch, view: ItemDetailView | None, capture: list | None = None
) -> TestClient:
    """Build a TestClient whose detail route returns ``view`` without DB access."""
    app = build_web_app()

    async def _fake_session():
        yield None

    app.dependency_overrides[_get_session] = _fake_session

    async def _fake_fetch_item_detail(session, item_id):
        if capture is not None:
            capture.append({"item_id": item_id})
        return view

    monkeypatch.setattr("argos.web.app.fetch_item_detail", _fake_fetch_item_detail)
    return TestClient(app, raise_server_exceptions=False)


def test_item_detail_renders_title_summary_and_source_link(monkeypatch):
    view = _view(
        title="Claude Opus 4.8 ships",
        summary="A leap in coding reliability.",
        source_url="https://example.com/opus-4-8",
    )
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "Claude Opus 4.8 ships" in body
    assert "A leap in coding reliability." in body
    assert "https://example.com/opus-4-8" in body
    # The 원문 보기 link must open in a new tab with safe rel attributes.
    assert 'target="_blank"' in body
    assert "noopener" in body
    assert "noreferrer" in body
    assert "원문 보기" in body


def test_item_detail_renders_hero_image_when_url_present(monkeypatch):
    view = _view(image_url="https://cdn.example.com/hero.jpg")
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "https://cdn.example.com/hero.jpg" in body
    assert "<img" in body
    assert "detail-hero__img" in body
    # Fallback marker must NOT appear when we have an image.
    assert "detail-hero--fallback" not in body


def test_item_detail_renders_category_tinted_fallback_when_no_image(monkeypatch):
    view = _view(image_url=None, category=CategoryType.ALPHA)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "detail-hero--fallback" in body
    assert "detail-hero--alpha" in body
    assert "<img" not in body or "detail-hero__img" not in body


def test_item_detail_renders_trust_score_dial(monkeypatch):
    view = _view(trust_score=0.73)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "trust-dial" in body
    # 0.73 → 73% on the conic dial.
    assert "73" in body
    assert "신뢰도" in body


def test_item_detail_omits_trust_dial_when_score_is_none(monkeypatch):
    view = _view(trust_score=None)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    assert "trust-dial" not in resp.text


def test_item_detail_returns_404_for_unknown_id(monkeypatch):
    capture: list = []
    client = _client_with_detail(monkeypatch, view=None, capture=capture)

    item_id = uuid.uuid4()
    resp = client.get(f"/item/{item_id}")

    assert resp.status_code == 404
    body = resp.text
    assert "404" in body
    assert "관측 대상이 없습니다" in body
    # Route must call the service exactly once with the parsed UUID.
    assert capture == [{"item_id": item_id}]


def test_item_detail_returns_404_for_malformed_uuid(monkeypatch):
    capture: list = []
    client = _client_with_detail(monkeypatch, view=None, capture=capture)

    resp = client.get("/item/not-a-uuid")

    assert resp.status_code == 404
    assert "관측 대상이 없습니다" in resp.text
    # Service must never be called with a non-UUID.
    assert capture == []


def _event(**overrides) -> ItemEvent:
    values = dict(
        id=uuid.uuid4(),
        title="OpenAI가 GPT-5를 공개했다",
        summary="사건 요약 한 단락.",
        event_type_label="출시",
        sections=(
            EventSection("무엇이 새로운가", "멀티모달 추론을 한다."),
            EventSection("쓸 수 있는 곳·시점", "오늘부터 API로 쓸 수 있다."),
        ),
    )
    values.update(overrides)
    return ItemEvent(**values)


def test_detail_shows_the_event_sections_and_links_to_the_event(monkeypatch):
    """ARG-308: 기사 상세는 digest 대신 그 기사가 속한 사건의 소제목 상세를 보여 준다."""
    event = _event()
    view = _view(event=event)
    resp = _client_with_detail(monkeypatch, view).get(f"/item/{view.id}")
    assert resp.status_code == 200
    assert "이 기사가 속한 사건" in resp.text
    assert f'href="/event/{event.id}"' in resp.text
    assert "무엇이 새로운가" in resp.text and "멀티모달 추론을 한다." in resp.text
    assert "쓸 수 있는 곳·시점" in resp.text
    assert "출시" in resp.text
    assert resp.text.index("무엇이 새로운가") < resp.text.index("쓸 수 있는 곳·시점")
    assert "detail-digest" not in resp.text


def test_detail_without_event_shows_no_event_block(monkeypatch):
    view = _view(summary="한 줄 리드.", event=None)
    resp = _client_with_detail(monkeypatch, view).get(f"/item/{view.id}")
    assert resp.status_code == 200
    assert "한 줄 리드." in resp.text
    assert "이 기사가 속한 사건" not in resp.text


def test_detail_event_without_sections_falls_back_to_its_summary(monkeypatch):
    view = _view(event=_event(sections=(), event_type_label=None))
    resp = _client_with_detail(monkeypatch, view).get(f"/item/{view.id}")
    assert "사건 요약 한 단락." in resp.text
    assert "event-section__heading" not in resp.text


def test_detail_event_sections_are_html_escaped(monkeypatch):
    view = _view(event=_event(sections=(EventSection("방법", "<script>alert(1)</script>"),)))
    resp = _client_with_detail(monkeypatch, view).get(f"/item/{view.id}")
    assert "<script>alert(1)</script>" not in resp.text
    assert "&lt;script&gt;" in resp.text


def test_item_detail_inherits_base_layout(monkeypatch):
    view = _view()
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    # Masthead + tabbar from base.html must be present.
    assert "ARGOS" in body
    assert "관측 피드" in body
    assert "포트폴리오" in body


# --- ARG-184: Keep/Pass/Untrack action bar on the detail page ------------ #


def test_item_detail_untriaged_renders_keep_and_pass_buttons(monkeypatch):
    """An item with no user_asset shows plain (inactive) Keep + Pass buttons,
    each hx-posting to the existing feed action routes with ``context=detail``
    so the response re-renders this action bar, not a feed-card fragment."""
    view = _view(status=None, asset_id=None)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert f"/items/{view.id}/keep?context=detail" in body
    assert f"/items/{view.id}/pass?context=detail" in body
    assert "✓ Keep" not in body
    assert "✓ Pass" not in body
    assert "Untrack" not in body
    assert f"item-actions-{view.id}" in body


def test_item_detail_kept_item_shows_untrack_instead_of_keep(monkeypatch):
    """Once Kept, the Keep button is replaced by Untrack (mirrors the
    portfolio's one-way release action) and posts to /assets/{asset_id}/untrack
    with both context=detail and the tech_id needed to re-resolve on a stale
    user_asset_id."""
    asset_id = uuid.uuid4()
    view = _view(status=AssetStatus.KEEP, asset_id=asset_id)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert f"/assets/{asset_id}/untrack" in body
    assert "context=detail" in body
    assert f"tech_id={view.id}" in body
    assert f"/items/{view.id}/keep" not in body
    # Pass stays available (inactive) so the user can switch the decision.
    assert f"/items/{view.id}/pass?context=detail" in body
    assert "✓ Pass" not in body


def test_item_detail_passed_item_shows_active_pass_button(monkeypatch):
    """A previously-Passed item renders the Pass button pressed (✓ Pass) with
    ``active=1`` so a re-click clears it, mirroring the feed card contract."""
    view = _view(status=AssetStatus.ARCHIVED, asset_id=uuid.uuid4())
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "✓ Pass" in body
    # Jinja autoescapes the ``&`` joiner in the query string (matches the
    # feed card's own ``join('&')`` pattern).
    assert f"/items/{view.id}/pass?context=detail&amp;active=1" in body
    # Not Kept, so the Keep button (not Untrack) is shown.
    assert f"/items/{view.id}/keep?context=detail" in body
    assert "Untrack" not in body


# --- ARG-211: 신뢰도 구성 요소 내역 partial ------------------------------- #


def test_item_detail_renders_trust_breakdown_when_rubric_present(monkeypatch):
    rubric = {
        "is_primary_source": True,
        "has_evidence_links": True,
        "has_concrete_numbers": False,
        "claim_evidence_balance": "balanced",
        "marketing_intensity": "low",
    }
    view = _view(
        source_url="https://example.com/rubric-item",
        trust_rubric=rubric,
        corroboration_count=2,
    )
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    body = resp.text
    assert "신뢰도 구성" in body
    assert "예" in body
    assert "있음" in body
    assert "없음" in body
    assert "balanced" in body
    assert "low" in body
    # Full source URL (not a bare domain substring) — asserts the detail page
    # renders the item's source link, and avoids CodeQL's incomplete-URL-
    # sanitization false positive on a lone "example.com" check.
    assert "https://example.com/rubric-item" in body
    assert "2건" in body


def test_item_detail_omits_trust_breakdown_for_legacy_row(monkeypatch):
    """Legacy rows (trust_rubric IS NULL) render no breakdown section."""
    view = _view(trust_rubric=None, corroboration_count=None)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    assert "신뢰도 구성" not in resp.text
    assert "detail-trust-breakdown" not in resp.text


def test_item_detail_shows_no_body_note_for_thin_items(monkeypatch):
    """ARG-309: 본문을 못 가져온 기사는 상세에 '본문 없음'이 보인다."""
    view = _view(has_body=False)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    assert "본문 없음" in resp.text


def test_item_detail_hides_no_body_note_when_body_exists(monkeypatch):
    view = _view(has_body=True)
    client = _client_with_detail(monkeypatch, view)

    resp = client.get(f"/item/{view.id}")

    assert resp.status_code == 200
    assert "본문 없음" not in resp.text
