"""Route + template tests for 기술 칩 on event cards / 사건 상세 (ARG-319). No live DB."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

from argos.models.tech_item import CategoryType
from argos.web.app import _get_session, build_web_app
from argos.web.services.event_detail import EventDetailView, EvidenceDoc
from argos.web.services.feed import FeedItem, FeedPage
from argos.web.services.technology import TechChip

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
CSS = Path(__file__).resolve().parents[2] / "src/argos/web/static/css/argos.css"


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
        id=uuid.uuid4(), title="Claude Sonnet 5 출시",
        source_url="https://www.anthropic.com/news/sonnet-5",
        category=CategoryType.MAINSTREAM, image_url=None, summary="요약",
        status=None, trust_score=0.8, sort_at=T0, feed_score=0.9,
        kind="event", rep_id=uuid.uuid4(), doc_count=5,
        source_domains=("anthropic.com", "theverge.com"),
    )
    defaults.update(kw)
    return FeedItem(**defaults)


def _lone_entry(**kw) -> FeedItem:
    defaults = dict(
        id=uuid.uuid4(), title="혼자인 기사", source_url="https://blog.example.com/post",
        category=CategoryType.ALPHA, image_url=None, summary=None,
        status=None, trust_score=None, sort_at=T0,
    )
    defaults.update(kw)
    return FeedItem(**defaults)


def _chips(n: int) -> list[TechChip]:
    return [TechChip(entity_id=uuid.uuid4(), name=f"Tech{i}", doc_count=n - i) for i in range(n)]


def _feed_client(monkeypatch, items, chips_by_event=None, calls=None):
    async def _fetch_feed(session, *, category=None, cursor=None, limit=20, sort="recommended"):
        return FeedPage(items=list(items), next_cursor=None)

    async def _none(*a, **k):
        return None

    async def _empty(*a, **k):
        return []

    async def _chips_fn(session, event_ids):
        if calls is not None:
            calls.append(list(event_ids))
        return {k: v for k, v in (chips_by_event or {}).items() if k in event_ids}

    return _client(
        monkeypatch,
        **{
            "argos.web.app.fetch_feed": _fetch_feed,
            "argos.web.app.select_hero": _none,
            "argos.web.app.latest_feed_cursor": _none,
            "argos.web.app.fetch_activity": _empty,
            "argos.web.app.fetch_event_tech_chips": _chips_fn,
        },
    )


def _card_html(body: str, entry_id: uuid.UUID) -> str:
    start = body.index(f'id="feed-card-{entry_id}"')
    return body[start: body.index("</article>", start)]


def test_feed_event_card_shows_at_most_three_chips(monkeypatch):
    ev, chips = _event_entry(), _chips(5)
    card = _card_html(_feed_client(monkeypatch, [ev], {ev.id: chips}).get("/feed").text, ev.id)
    assert card.count('href="/tech/') == 3
    for c in chips[:3]:
        assert c.name in card
    for c in chips[3:]:
        assert c.name not in card


def test_feed_chip_links_to_tech_page(monkeypatch):
    ev, chips = _event_entry(), _chips(1)
    card = _card_html(_feed_client(monkeypatch, [ev], {ev.id: chips}).get("/feed").text, ev.id)
    assert f'<a class="tech-chip" href="/tech/{chips[0].entity_id}"' in card
    # 칩 링크는 제목 링크 안에 중첩되지 않는다.
    headline = card[card.index('class="headline"'): card.index("</h2>")]
    assert "tech-chip" not in headline


def test_feed_lone_item_card_has_no_chip_area(monkeypatch):
    ev, lone = _event_entry(), _lone_entry()
    body = _feed_client(monkeypatch, [ev, lone], {ev.id: _chips(2)}).get("/feed").text
    assert 'class="tech-chips"' not in _card_html(body, lone.id)


def test_event_card_without_chips_has_no_chip_area(monkeypatch):
    ev = _event_entry()
    card = _card_html(_feed_client(monkeypatch, [ev], {}).get("/feed").text, ev.id)
    assert 'class="tech-chips"' not in card


def test_feed_order_unchanged_by_chips(monkeypatch):
    ev1, lone, ev2 = _event_entry(), _lone_entry(), _event_entry()
    items = [ev1, lone, ev2]
    order = lambda body: re.findall(r'id="feed-card-([0-9a-f-]{36})"', body)  # noqa: E731
    calls: list[list[uuid.UUID]] = []
    with_chips = _feed_client(
        monkeypatch, items, {ev1.id: _chips(3), ev2.id: _chips(2)}, calls
    ).get("/feed").text
    without = _feed_client(monkeypatch, items, {}).get("/feed").text
    assert order(with_chips) == order(without) == [str(i.id) for i in items]
    assert calls == [[ev1.id, ev2.id]]  # 사건 엔트리 id만, 한 번


def test_feed_items_fragment_shows_chips(monkeypatch):
    ev, chips = _event_entry(), _chips(2)
    body = _feed_client(monkeypatch, [ev], {ev.id: chips}).get("/feed/items").text
    assert body.count('href="/tech/') == 2


def _detail_view(**kw) -> EventDetailView:
    defaults = dict(
        id=uuid.uuid4(), title="사건", summary="요약",
        documents=[
            EvidenceDoc(id=uuid.uuid4(), title="기사", source_url="https://a.example/x",
                        domain="a.example", image_url=None, summary=None,
                        reported_at=T0, is_first=True),
        ],
        rep_id=None, category=CategoryType.MAINSTREAM, trust_score=None, image_url=None,
    )
    defaults.update(kw)
    return EventDetailView(**defaults)


def _detail_client(monkeypatch, view, chips):
    async def _resolve(session, event_id):
        return event_id

    async def _fetch(session, event_id):
        return view

    async def _chips_fn(session, event_ids):
        return {view.id: chips} if chips else {}

    return _client(
        monkeypatch,
        **{
            "argos.web.app.resolve_event": _resolve,
            "argos.web.app.fetch_event_detail": _fetch,
            "argos.web.app.fetch_event_tech_chips": _chips_fn,
        },
    )


def test_event_detail_shows_all_chips_below_evidence(monkeypatch):
    view, chips = _detail_view(), _chips(5)
    body = _detail_client(monkeypatch, view, chips).get(f"/event/{view.id}").text
    assert body.count('href="/tech/') == 5
    for c in chips:
        assert c.name in body
    assert body.index('class="evidence"') < body.index('class="tech-chips')


def test_event_without_chips_has_no_chip_area(monkeypatch):
    view = _detail_view()
    body = _detail_client(monkeypatch, view, []).get(f"/event/{view.id}").text
    assert 'class="tech-chips' not in body
    assert "detail-techs" not in body


def test_keep_rerender_without_chips_ok(monkeypatch):
    from argos.models.user_asset import AssetStatus
    from argos.slack.services.asset_transition import ToggleOutcome

    ev = _event_entry(status=AssetStatus.KEEP)

    async def _toggle(session, tech_id, target_status, *, currently_active=False):
        return ToggleOutcome.SET

    async def _ctx(session, tech_id):
        return {"id": tech_id, "title": "대표", "status": AssetStatus.KEEP, "category": None,
                "image_url": None, "summary": None, "trust_score": None,
                "source_url": "https://x", "asset_id": None}

    async def _entry(session, entry_id):
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
    assert 'class="tech-chips"' not in resp.text


def test_chip_css_wraps_on_phone_width():
    css = CSS.read_text()
    chips_rule = re.search(r"\.tech-chips\s*\{([^}]*)\}", css)
    chip_rule = re.search(r"\.tech-chip\s*\{([^}]*)\}", css)
    assert chips_rule and "flex-wrap: wrap" in chips_rule.group(1)
    assert chip_rule and "max-width: 100%" in chip_rule.group(1)
    assert "overflow-wrap: anywhere" in chip_rule.group(1)


def test_event_detail_survives_click_commit_failure(monkeypatch):
    """Click 기록 commit이 실패해도 롤백 후 칩과 함께 200 (ARG-319 최종 리뷰)."""
    rep_id = uuid.uuid4()
    view, chips = _detail_view(rep_id=rep_id), _chips(2)
    state = {"rolled_back": False, "commit_failed": False}

    class _FailingCommitSession(_FakeSession):
        async def commit(self) -> None:
            state["commit_failed"] = True
            raise RuntimeError("commit failed")

        async def rollback(self) -> None:
            state["rolled_back"] = True

    async def _resolve(session, event_id):
        return event_id

    async def _fetch(session, event_id):
        return view

    async def _chips_fn(session, event_ids):
        # 세션이 롤백 필요 상태(commit 실패 후, rollback 전)면 SQLAlchemy는 PendingRollbackError.
        if not state["rolled_back"] and state["commit_failed"]:
            raise RuntimeError("PendingRollbackError")
        return {view.id: chips}

    async def _ctx(session, tech_id):
        return {"id": tech_id, "title": "대표", "status": None, "category": None,
                "image_url": None, "summary": None, "trust_score": None,
                "source_url": "https://x", "asset_id": None}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.resolve_event": _resolve,
            "argos.web.app.fetch_event_detail": _fetch,
            "argos.web.app.fetch_event_tech_chips": _chips_fn,
            "argos.web.app._load_feed_card_context": _ctx,
        },
    )

    async def _fake_session():
        yield _FailingCommitSession()

    client.app.dependency_overrides[_get_session] = _fake_session
    resp = client.get(f"/event/{view.id}")
    assert resp.status_code == 200
    assert state["commit_failed"] and state["rolled_back"]
    for c in chips:
        assert c.name in resp.text


def test_event_detail_chip_failure_rolls_back_before_click_log(monkeypatch):
    """칩 조회가 실패하면 롤백해 이어지는 Click 기록이 망가진 트랜잭션에 걸리지 않게 한다."""
    view = _detail_view(rep_id=uuid.uuid4())
    calls: list[str] = []

    class _OrderSession(_FakeSession):
        def add(self, obj) -> None:
            calls.append("add")

        async def rollback(self) -> None:
            calls.append("rollback")

    async def _resolve(session, event_id):
        return event_id

    async def _fetch(session, event_id):
        return view

    async def _boom(session, event_ids):
        raise RuntimeError("chip query failed")

    async def _ctx(session, tech_id):
        return {"id": tech_id, "title": "대표", "status": None, "category": None,
                "image_url": None, "summary": None, "trust_score": None,
                "source_url": "https://x", "asset_id": None}

    client = _client(
        monkeypatch,
        **{
            "argos.web.app.resolve_event": _resolve,
            "argos.web.app.fetch_event_detail": _fetch,
            "argos.web.app.fetch_event_tech_chips": _boom,
            "argos.web.app._load_feed_card_context": _ctx,
        },
    )

    async def _fake_session():
        yield _OrderSession()

    client.app.dependency_overrides[_get_session] = _fake_session
    resp = client.get(f"/event/{view.id}")
    assert resp.status_code == 200
    assert calls[:2] == ["rollback", "add"]


def test_feed_survives_chip_fetch_failure(monkeypatch):
    ev = _event_entry()
    client = _feed_client(monkeypatch, [ev], {})

    async def _boom(session, event_ids):
        raise RuntimeError("chip query failed")

    monkeypatch.setattr("argos.web.app.fetch_event_tech_chips", _boom)
    resp = client.get("/feed")
    assert resp.status_code == 200
    assert 'class="tech-chips"' not in _card_html(resp.text, ev.id)
