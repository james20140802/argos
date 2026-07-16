"""Route tests for POST /events/batch and the Click-on-view recording on
GET /item/{id} (ARG-207).

These exercise the routes without a live database: the per-request session
dependency is overridden with an in-memory fake that records ``add()``/
``commit()`` calls, and ``fetch_item_detail`` is monkeypatched for the
detail-page Click test. Keeps this suite runnable on release.yml CI (no
Postgres) — mirrors the pattern in test_actions_route.py / test_item_detail_route.py.
"""
from __future__ import annotations

import uuid

from starlette.testclient import TestClient

from argos.models.feed_event import FeedEventType
from argos.models.tech_item import CategoryType
from argos.web.app import _get_session, build_web_app
from argos.web.services.detail import ItemDetailView


class _FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    """Minimal stand-in for an AsyncSession that records add()/commit().

    ``execute`` models the /events/batch existence pre-filter: it pulls the
    UUIDs out of the ``TechItem.id.in_(...)`` guard query and reports them as
    existing, minus any ids in ``missing_ids`` (so a test can simulate a
    dangling item_id that must be skipped rather than sinking the batch).
    """

    def __init__(self, missing_ids: set | None = None) -> None:
        self.added: list = []
        self.commit_count = 0
        self.missing_ids = missing_ids or set()

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commit_count += 1

    async def execute(self, statement):
        wanted: set = set()
        for value in statement.compile().params.values():
            if isinstance(value, (list, tuple, set)):
                wanted.update(value)
            elif isinstance(value, uuid.UUID):
                wanted.add(value)
        return _FakeResult([i for i in wanted if i not in self.missing_ids])


def _client(monkeypatch, session: _FakeSession) -> TestClient:
    app = build_web_app()

    async def _fake_session():
        yield session

    app.dependency_overrides[_get_session] = _fake_session
    return TestClient(app, raise_server_exceptions=False)


# --- POST /events/batch ---------------------------------------------------- #


def test_events_batch_inserts_valid_events(monkeypatch):
    session = _FakeSession()
    client = _client(monkeypatch, session)
    item_id = str(uuid.uuid4())

    resp = client.post(
        "/events/batch",
        json={
            "events": [
                {"type": "Impression", "item_id": item_id},
                {"type": "Dwell", "item_id": item_id, "value": 4.2},
            ]
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"inserted": 2}
    assert len(session.added) == 2
    assert session.commit_count == 1
    types = {ev.event_type for ev in session.added}
    assert types == {FeedEventType.IMPRESSION, FeedEventType.DWELL}
    dwell = next(ev for ev in session.added if ev.event_type == FeedEventType.DWELL)
    assert dwell.value == 4.2


def test_events_batch_skips_unknown_type(monkeypatch):
    session = _FakeSession()
    client = _client(monkeypatch, session)
    item_id = str(uuid.uuid4())

    resp = client.post(
        "/events/batch",
        json={
            "events": [
                {"type": "Bogus", "item_id": item_id},
                {"type": "Click", "item_id": item_id},
            ]
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"inserted": 1}
    assert len(session.added) == 1
    assert session.added[0].event_type == FeedEventType.CLICK


def test_events_batch_skips_malformed_uuid(monkeypatch):
    session = _FakeSession()
    client = _client(monkeypatch, session)

    resp = client.post(
        "/events/batch",
        json={"events": [{"type": "Impression", "item_id": "not-a-uuid"}]},
    )

    assert resp.status_code == 200
    assert resp.json() == {"inserted": 0}
    assert session.added == []
    assert session.commit_count == 0


def test_events_batch_dangling_item_id_does_not_sink_good_events(monkeypatch):
    # P2 fix (proactive review): a well-formed but non-existent item_id must be
    # skipped, NOT roll the whole batch back on an FK violation at commit. The
    # client clears its queue before sending and never retries, so a co-batched
    # dangling id would otherwise silently drop every good Impression/Dwell.
    good_id = uuid.uuid4()
    dangling_id = uuid.uuid4()
    session = _FakeSession(missing_ids={dangling_id})
    client = _client(monkeypatch, session)

    resp = client.post(
        "/events/batch",
        json={
            "events": [
                {"type": "Impression", "item_id": str(good_id)},
                {"type": "Dwell", "item_id": str(dangling_id), "value": 3.0},
            ]
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {"inserted": 1}
    assert len(session.added) == 1
    assert session.added[0].tech_item_id == good_id
    assert session.added[0].event_type == FeedEventType.IMPRESSION
    assert session.commit_count == 1


def test_events_batch_all_dangling_skips_commit(monkeypatch):
    # If every id is dangling, nothing is added and commit is never called
    # (the empty-insert guard), so an all-adversarial payload is a clean no-op.
    dangling_id = uuid.uuid4()
    session = _FakeSession(missing_ids={dangling_id})
    client = _client(monkeypatch, session)

    resp = client.post(
        "/events/batch",
        json={"events": [{"type": "Click", "item_id": str(dangling_id)}]},
    )

    assert resp.status_code == 200
    assert resp.json() == {"inserted": 0}
    assert session.added == []
    assert session.commit_count == 0


def test_events_batch_empty_list_returns_zero(monkeypatch):
    session = _FakeSession()
    client = _client(monkeypatch, session)

    resp = client.post("/events/batch", json={"events": []})

    assert resp.status_code == 200
    assert resp.json() == {"inserted": 0}
    assert session.commit_count == 0


def test_events_batch_malformed_body_does_not_500(monkeypatch):
    session = _FakeSession()
    client = _client(monkeypatch, session)

    resp = client.post("/events/batch", content=b"not json", headers={"content-type": "application/json"})

    assert resp.status_code == 200
    assert resp.json() == {"inserted": 0}


# --- GET /item/{id} records a Click event ----------------------------------- #


def _view(item_id: uuid.UUID) -> ItemDetailView:
    return ItemDetailView(
        id=item_id,
        title="Test item",
        source_url="https://example.com/x",
        image_url=None,
        summary="s",
        category=CategoryType.MAINSTREAM,
        trust_score=0.5,
        published_at=None,
    )


def test_item_detail_records_click_event(monkeypatch):
    session = _FakeSession()
    client = _client(monkeypatch, session)
    item_id = uuid.uuid4()

    async def _fake_fetch(session, item_id_):
        return _view(item_id)

    monkeypatch.setattr("argos.web.app.fetch_item_detail", _fake_fetch)

    resp = client.get(f"/item/{item_id}")

    assert resp.status_code == 200
    assert len(session.added) == 1
    click_event = session.added[0]
    assert click_event.event_type == FeedEventType.CLICK
    assert click_event.tech_item_id == item_id
    assert session.commit_count == 1


def test_item_detail_404_does_not_record_click(monkeypatch):
    session = _FakeSession()
    client = _client(monkeypatch, session)
    item_id = uuid.uuid4()

    async def _fake_fetch(session, item_id_):
        return None

    monkeypatch.setattr("argos.web.app.fetch_item_detail", _fake_fetch)

    resp = client.get(f"/item/{item_id}")

    assert resp.status_code == 404
    assert session.added == []
    assert session.commit_count == 0
