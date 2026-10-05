"""Route + template tests for event cards and the 사건 상세 page (ARG-243).

No live database: the session dependency is overridden and every DB-touching
helper is monkeypatched, so these run on release CI (no Postgres).
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from starlette.testclient import TestClient

from argos.models.tech_item import CategoryType
from argos.models.user_asset import AssetStatus
from argos.slack.services.asset_transition import ToggleOutcome
from argos.web.app import _get_session, build_web_app
from argos.web.services.event_detail import EventDetailView, EvidenceDoc
from argos.web.services.feed import FeedItem, FeedPage

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


class _FakeSession:
    def add(self, obj) -> None:
        return None

    async def commit(self) -> None:
        return None


def _client(monkeypatch, **patches) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield _FakeSession()

    app.dependency_overrides[_get_session] = _fake_session

    async def _no_successors(session, tech_id):
        return []

    monkeypatch.setattr("argos.web.app._load_item_successors", _no_successors)
    for path, fn in patches.items():
        monkeypatch.setattr(path, fn)
    return TestClient(app, raise_server_exceptions=False)


def _event_entry(**kw) -> FeedItem:
    defaults = dict(
        id=uuid.uuid4(),
        title="Claude Sonnet 5 출시",
        source_url="https://www.anthropic.com/news/sonnet-5",
        category=CategoryType.MAINSTREAM,
        image_url=None,
        summary="다섯 매체가 같은 날 보도했다.",
        status=None,
        trust_score=0.8,
        sort_at=T0,
        feed_score=0.9,
        kind="event",
        rep_id=uuid.uuid4(),
        doc_count=5,
        source_domains=(
            "anthropic.com", "theverge.com", "techcrunch.com",
            "news.ycombinator.com", "arstechnica.com",
        ),
    )
    defaults.update(kw)
    return FeedItem(**defaults)


def _lone_entry(**kw) -> FeedItem:
    defaults = dict(
        id=uuid.uuid4(),
        title="혼자인 기사",
        source_url="https://blog.example.com/post",
        category=CategoryType.ALPHA,
        image_url=None,
        summary=None,
        status=None,
        trust_score=None,
        sort_at=T0,
    )
    defaults.update(kw)
    return FeedItem(**defaults)


def _feed_client(monkeypatch, items):
    async def _fetch_feed(session, *, category=None, cursor=None, limit=20, sort="recommended"):
        return FeedPage(items=list(items), next_cursor=None)

    async def _none(*a, **k):
        return None

    async def _empty(*a, **k):
        return []

    return _client(
        monkeypatch,
        **{
            "argos.web.app.fetch_feed": _fetch_feed,
            "argos.web.app.select_hero": _none,
            "argos.web.app.latest_feed_cursor": _none,
            "argos.web.app.fetch_activity": _empty,
        },
    )


def _card_html(body: str, entry_id: uuid.UUID) -> str:
    start = body.index(f'id="feed-card-{entry_id}"')
    return body[start: body.index("</article>", start)]


# --------------------------------------------------------------------- #
# Feed cards
# --------------------------------------------------------------------- #


def test_event_card_shows_source_count_and_links_to_event_page(monkeypatch):
    ev = _event_entry()
    body = _feed_client(monkeypatch, [ev]).get("/feed").text
    card = _card_html(body, ev.id)
    assert "출처 5곳" in card
    assert f'href="/event/{ev.id}"' in card
    assert f"/item/{ev.id}" not in card
    # Keep/Pass act on the representative document and re-render this card.
    assert f'hx-post="/items/{ev.rep_id}/keep?' in card
    assert f"entry={ev.id}" in card
    # Only the button row (+ the eyebrow, out of band) is swapped — never the
    # whole card, whose cover image would re-load and flash.
    assert f'hx-target="#actions-{ev.id}"' in card
    assert f'hx-select-oob="#eyebrow-{ev.id}"' in card
    assert f'hx-target="#feed-card-{ev.id}"' not in card


def test_lone_document_card_is_the_same_component_on_its_old_url(monkeypatch):
    ev, lone = _event_entry(), _lone_entry()
    body = _feed_client(monkeypatch, [ev, lone]).get("/feed").text
    card = _card_html(body, lone.id)
    assert "출처 1곳" in card
    assert f'href="/item/{lone.id}"' in card
    assert f'hx-post="/items/{lone.id}/keep"' in card  # bare URL, no entry=
    # Same building blocks as the event card — no second card component.
    for part in ('class="cover', 'class="headline"', 'class="source-stack"', 'class="card-actions"'):
        assert part in card
        assert part in _card_html(body, ev.id)


def test_source_stack_caps_discs_and_counts_overflow(monkeypatch):
    ev = _event_entry(source_domains=tuple(f"s{i}.example" for i in range(7)), doc_count=9)
    card = _card_html(_feed_client(monkeypatch, [ev]).get("/feed").text, ev.id)
    assert card.count('class="source-disc"') == 4
    assert "+3" in card
    assert "출처 7곳" in card and "기사 9건" in card


def test_feed_impressions_attribute_to_representative_document(monkeypatch):
    ev = _event_entry()
    card = _card_html(_feed_client(monkeypatch, [ev]).get("/feed").text, ev.id)
    assert f'data-item-id="{ev.rep_id}"' in card
    assert f'data-event-id="{ev.id}"' in card


# --------------------------------------------------------------------- #
# Keep/Pass on an event card
# --------------------------------------------------------------------- #


def test_keep_on_event_card_rerenders_the_event_card(monkeypatch):
    ev = _event_entry(status=AssetStatus.KEEP)
    toggled = {}

    async def _toggle(session, tech_id, target_status, *, currently_active=False):
        toggled["tech_id"] = tech_id
        return ToggleOutcome.SET

    async def _ctx(session, tech_id):
        return {"id": tech_id, "title": "대표 문서", "status": AssetStatus.KEEP,
                "category": None, "image_url": None, "summary": None,
                "trust_score": None, "source_url": "https://x", "asset_id": None}

    async def _entry(session, entry_id):
        assert entry_id == ev.id
        return ev

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _toggle,
            "argos.web.app._load_feed_card_context": _ctx,
            "argos.web.app.fetch_feed_entry": _entry,
        },
    )
    resp = client.post(f"/items/{ev.rep_id}/keep?entry={ev.id}")
    assert resp.status_code == 200
    assert toggled["tech_id"] == ev.rep_id
    assert f'id="feed-card-{ev.id}"' in resp.text
    assert "출처 5곳" in resp.text
    assert "✓ Keep" in resp.text
    assert "card--swapped" in resp.text


def test_keep_with_malformed_entry_is_404(monkeypatch):
    client = _client(monkeypatch)
    resp = client.post(f"/items/{uuid.uuid4()}/keep?entry=nope")
    assert resp.status_code == 404


# --------------------------------------------------------------------- #
# Event detail page
# --------------------------------------------------------------------- #


def _event_view(event_id: uuid.UUID, rep_id: uuid.UUID) -> EventDetailView:
    docs = [
        EvidenceDoc(id=rep_id, title="Anthropic 공식 발표", source_url="https://anthropic.com/a",
                    domain="anthropic.com", image_url=None, summary=None,
                    reported_at=T0, is_first=True),
        EvidenceDoc(id=uuid.uuid4(), title="The Verge 기사", source_url="https://theverge.com/b",
                    domain="theverge.com", image_url=None, summary=None,
                    reported_at=T0.replace(hour=11), is_first=False),
        EvidenceDoc(id=uuid.uuid4(), title="TechCrunch 기사", source_url="https://techcrunch.com/c",
                    domain="techcrunch.com", image_url=None, summary=None,
                    reported_at=T0.replace(hour=13), is_first=False),
    ]
    return EventDetailView(
        id=event_id, title="Claude Sonnet 5 출시", summary="사건 요약", documents=docs,
        rep_id=rep_id, category=CategoryType.MAINSTREAM, trust_score=0.8, image_url=None,
    )


def _detail_client(monkeypatch, *, resolve_to=None, view=None):
    async def _resolve(session, event_id):
        return resolve_to or event_id

    async def _fetch(session, event_id):
        return view

    async def _ctx(session, tech_id):
        return {"id": tech_id, "title": "대표", "status": None, "category": None,
                "image_url": None, "summary": None, "trust_score": None,
                "source_url": "https://x", "asset_id": None}

    return _client(
        monkeypatch,
        **{
            "argos.web.app.resolve_event": _resolve,
            "argos.web.app.fetch_event_detail": _fetch,
            "argos.web.app._load_feed_card_context": _ctx,
        },
    )


def test_event_page_lists_every_source_in_report_order_with_first_marked(monkeypatch):
    eid, rep = uuid.uuid4(), uuid.uuid4()
    resp = _detail_client(monkeypatch, view=_event_view(eid, rep)).get(f"/event/{eid}")
    assert resp.status_code == 200
    body = resp.text
    order = [body.index(t) for t in ("Anthropic 공식 발표", "The Verge 기사", "TechCrunch 기사")]
    assert order == sorted(order)
    for domain in ("anthropic.com", "theverge.com", "techcrunch.com"):
        assert domain in body
    assert body.count("최초 보도") == 1
    first_item = body[body.index('evidence-item--first'):]
    assert first_item.index("최초 보도") < first_item.index("The Verge 기사")
    # Each report links out to its source, safely.
    assert 'href="https://theverge.com/b" target="_blank" rel="noopener noreferrer"' in body
    # Keep/Pass act on the representative, in the event context.
    assert f'hx-post="/items/{rep}/keep?context=event"' in body


def test_old_link_to_merged_event_redirects_to_survivor(monkeypatch):
    old, survivor = uuid.uuid4(), uuid.uuid4()
    client = _detail_client(monkeypatch, resolve_to=survivor, view=None)
    resp = client.get(f"/event/{old}", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == f"/event/{survivor}"


def test_unknown_or_malformed_event_is_404_page(monkeypatch):
    client = _detail_client(monkeypatch, view=None)
    assert client.get(f"/event/{uuid.uuid4()}").status_code == 404
    assert client.get("/event/not-a-uuid").status_code == 404


def test_event_context_action_rerenders_bar_without_signals_oob(monkeypatch):
    rep = uuid.uuid4()
    called = {"detail": 0}

    async def _toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.SET

    async def _ctx(session, tech_id):
        return {"id": tech_id, "title": "대표", "status": AssetStatus.ARCHIVED, "category": None,
                "image_url": None, "summary": None, "trust_score": None,
                "source_url": "https://x", "asset_id": None}

    async def _detail(session, tech_id):
        called["detail"] += 1
        return None

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.toggle_asset": _toggle,
            "argos.web.app._load_feed_card_context": _ctx,
            "argos.web.app.fetch_item_detail": _detail,
        },
    )
    resp = client.post(f"/items/{rep}/pass?context=event")
    assert resp.status_code == 200
    assert f'id="detail-actions-{rep}"' in resp.text
    assert "hx-swap-oob" not in resp.text
    assert called["detail"] == 0
    # Follow-up actions stay in the event context.
    assert re.search(r'/items/[0-9a-f-]+/pass\?context=event', resp.text)
