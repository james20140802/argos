"""사건 상세 심층 분석 라우트·템플릿(ARG-336). DB 없음."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from starlette.testclient import TestClient

from argos.models.event_deep_analysis import DeepAnalysisStatus
from argos.models.tech_item import CategoryType
from argos.web import auth
from argos.web.app import _get_session, build_web_app
from argos.web.services.deep_analysis import DeepAnalysisSection, DeepAnalysisView
from argos.web.services.event_detail import EventDetailView, EvidenceDoc

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
HEADINGS = ("핵심 기술·접근", "성숙도", "생태계와의 관계", "위험·한계", "추적할 만한가")


class _FakeSession:
    def add(self, obj) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _FakeRunner:
    def __init__(self) -> None:
        self.started: list[uuid.UUID] = []

    def start(self, event_id):
        self.started.append(event_id)


def _doc(**kw) -> EvidenceDoc:
    defaults = dict(
        id=uuid.uuid4(),
        title="기사",
        source_url="https://example.com/a",
        domain="example.com",
        reported_at=T0,
        is_first=False,
        image_url=None,
        summary=None,
    )
    defaults.update(kw)
    return EvidenceDoc(**defaults)


def _event(event_id: uuid.UUID, n_docs: int = 2) -> EventDetailView:
    return EventDetailView(
        id=event_id,
        title="사건 제목",
        summary="요약",
        documents=[_doc() for _ in range(n_docs)],
        rep_id=None,
        category=CategoryType.MAINSTREAM,
        trust_score=0.8,
        image_url=None,
        sections=(),
    )


def _view(event_id, **kw) -> DeepAnalysisView:
    defaults = dict(
        event_id=event_id,
        sections=(),
        analyzed_at=None,
        doc_count=None,
        status=None,
        can_start=True,
    )
    defaults.update(kw)
    return DeepAnalysisView(**defaults)


def _client(monkeypatch, event_id, *, n_docs=2, view=None, claim=True, resolved=None):
    app = build_web_app()

    async def _fake_session():
        yield _FakeSession()

    app.dependency_overrides[_get_session] = _fake_session
    runner = _FakeRunner()
    app.state.deep_analysis_runner = runner
    claims: list[uuid.UUID] = []
    target = resolved or event_id

    async def _resolve(session, eid):
        return target

    async def _detail(session, eid):
        return _event(eid, n_docs)

    async def _load(session, eid, doc_ids):
        return view if view is not None else _view(eid, can_start=bool(doc_ids))

    async def _claim(session, eid, **kw):
        claims.append(eid)
        return claim

    async def _none(*a, **k):
        return {}

    async def _no_succ(*a, **k):
        return []

    monkeypatch.setattr("argos.web.app.resolve_event", _resolve)
    monkeypatch.setattr("argos.web.app.fetch_event_detail", _detail)
    monkeypatch.setattr("argos.web.app.load_deep_analysis_view", _load)
    monkeypatch.setattr("argos.web.app.claim_deep_analysis", _claim)
    monkeypatch.setattr("argos.web.app.fetch_event_tech_chips", _none)
    monkeypatch.setattr("argos.web.app._load_item_successors", _no_succ)
    return TestClient(app, raise_server_exceptions=False), runner, claims


def test_event_page_shows_start_button(monkeypatch) -> None:
    eid = uuid.uuid4()
    client, _, _ = _client(monkeypatch, eid)
    body = client.get(f"/event/{eid}").text
    assert "심층 분석" in body
    assert f'hx-post="/event/{eid}/deep-analysis"' in body
    assert "event-sections" in body or "evidence-list" in body
    assert "evidence-list" in body
    assert body.index('id="deep-analysis"') < body.index('class="evidence"')


def test_event_page_hides_button_without_docs(monkeypatch) -> None:
    eid = uuid.uuid4()
    client, _, _ = _client(monkeypatch, eid, n_docs=0)
    assert "deep-analysis__start" not in client.get(f"/event/{eid}").text


def test_event_page_shows_saved_result_with_meta(monkeypatch) -> None:
    eid = uuid.uuid4()
    view = _view(
        eid,
        sections=tuple(
            DeepAnalysisSection(h, "Claude Duet 은 새 제품이다." if i == 0 else "본문")
            for i, h in enumerate(HEADINGS)
        ),
        analyzed_at=T0,
        doc_count=2,
    )
    client, _, _ = _client(monkeypatch, eid, view=view)
    body = client.get(f"/event/{eid}").text
    for h in HEADINGS:
        assert h in body
    assert "기사 2건으로 분석" in body
    assert "Claude Duet" in body
    assert "<time" in body


def test_post_starts_once_and_returns_progress(monkeypatch) -> None:
    eid = uuid.uuid4()
    running = _view(eid, status=DeepAnalysisStatus.RUNNING, can_start=False)
    client, runner, claims = _client(monkeypatch, eid, view=running, claim=True)
    r = client.post(f"/event/{eid}/deep-analysis")
    assert r.status_code == 200
    assert runner.started == [eid]
    assert "분석 중…" in r.text
    assert 'hx-trigger="every 3s"' in r.text


def test_post_while_running_does_not_restart(monkeypatch) -> None:
    eid = uuid.uuid4()
    running = _view(eid, status=DeepAnalysisStatus.RUNNING, can_start=False)
    client, runner, _ = _client(monkeypatch, eid, view=running, claim=False)
    r = client.post(f"/event/{eid}/deep-analysis")
    assert runner.started == []
    assert "분석 중…" in r.text


def test_post_on_merged_event_uses_survivor(monkeypatch) -> None:
    eid, survivor = uuid.uuid4(), uuid.uuid4()
    running = _view(survivor, status=DeepAnalysisStatus.RUNNING, can_start=False)
    client, runner, claims = _client(
        monkeypatch, eid, view=running, claim=True, resolved=survivor
    )
    client.post(f"/event/{eid}/deep-analysis")
    assert claims == [survivor]
    assert runner.started == [survivor]


def test_post_without_docs_does_not_start(monkeypatch) -> None:
    eid = uuid.uuid4()
    client, runner, claims = _client(monkeypatch, eid, n_docs=0)
    client.post(f"/event/{eid}/deep-analysis")
    assert claims == []
    assert runner.started == []


def test_get_status_stops_polling_when_done(monkeypatch) -> None:
    eid = uuid.uuid4()
    view = _view(
        eid,
        sections=tuple(DeepAnalysisSection(h, "b") for h in HEADINGS),
        analyzed_at=T0,
        doc_count=2,
    )
    client, _, _ = _client(monkeypatch, eid, view=view)
    r = client.get(f"/event/{eid}/deep-analysis")
    assert r.status_code == 200
    assert "hx-trigger" not in r.text


@pytest.mark.real_security
def test_post_requires_login(monkeypatch) -> None:
    eid = uuid.uuid4()
    client, runner, _ = _client(monkeypatch, eid)
    r = client.post(
        f"/event/{eid}/deep-analysis", headers={"Sec-Fetch-Site": "same-origin"}
    )
    assert r.status_code == 401
    assert runner.started == []
    assert auth  # keep import used


@pytest.mark.parametrize(
    ("held", "expected"),
    [(True, DeepAnalysisStatus.WAITING), (False, DeepAnalysisStatus.RUNNING)],
)
def test_post_claims_status_by_turn_state(monkeypatch, held, expected) -> None:
    eid = uuid.uuid4()
    client, _, _ = _client(monkeypatch, eid)
    seen: list = []

    async def _claim(session, e, **kw):
        seen.append(kw.get("status"))
        return True

    monkeypatch.setattr("argos.web.app.claim_deep_analysis", _claim)
    monkeypatch.setattr("argos.web.app.turn_is_held", lambda: held)
    client.post(f"/event/{eid}/deep-analysis")
    assert seen == [expected]


def test_waiting_view_shows_waiting_text_and_keeps_polling(monkeypatch) -> None:
    eid = uuid.uuid4()
    waiting = _view(eid, status=DeepAnalysisStatus.WAITING, can_start=False)
    client, _, _ = _client(monkeypatch, eid, view=waiting)
    body = client.get(f"/event/{eid}/deep-analysis").text
    assert "대기 중(다른 작업이 모델 사용 중)" in body
    assert 'hx-trigger="every 3s"' in body
    assert "분석 중…" not in body
    assert "취소" not in body
    running = _view(eid, status=DeepAnalysisStatus.RUNNING, can_start=False)
    client, _, _ = _client(monkeypatch, eid, view=running)
    body = client.get(f"/event/{eid}/deep-analysis").text
    assert "분석 중…" in body
    assert "대기 중" not in body


def _five_sections():
    return tuple(DeepAnalysisSection(h, "본문") for h in HEADINGS)


def test_failed_view_shows_reason_and_retry_button(monkeypatch) -> None:
    eid = uuid.uuid4()
    view = _view(
        eid,
        status=DeepAnalysisStatus.FAILED,
        failure_message="모델에 연결하지 못했어요",
    )
    client, _, _ = _client(monkeypatch, eid, view=view)
    body = client.get(f"/event/{eid}").text
    assert "모델에 연결하지 못했어요" in body
    assert 'role="alert"' in body
    assert ">심층 분석</button>" in body
    assert "hx-trigger" not in body


def test_failed_view_keeps_previous_result(monkeypatch) -> None:
    eid = uuid.uuid4()
    view = _view(
        eid,
        sections=_five_sections(),
        analyzed_at=T0,
        doc_count=2,
        status=DeepAnalysisStatus.FAILED,
        failure_message="시간 안에 분석을 마치지 못했어요",
    )
    client, _, _ = _client(monkeypatch, eid, view=view)
    body = client.get(f"/event/{eid}").text
    for h in HEADINGS:
        assert h in body
    assert "시간 안에 분석을 마치지 못했어요" in body
    assert body.count(">다시 분석</button>") == 1
    assert "hx-trigger" not in body
    assert body.index("시간 안에") < body.index(HEADINGS[0])


def test_failure_text_has_no_raw_error(monkeypatch) -> None:
    from argos.web.services.deep_analysis import failure_code, failure_message
    from argos.brain.event_deep_analysis import DeepAnalysisError, DeepAnalysisFailure

    eid = uuid.uuid4()
    exc = DeepAnalysisError(DeepAnalysisFailure.BadFormat, "Traceback (most recent call last)")
    view = _view(
        eid,
        status=DeepAnalysisStatus.FAILED,
        failure_message=failure_message(failure_code(exc)),
    )
    client, _, _ = _client(monkeypatch, eid, view=view)
    body = client.get(f"/event/{eid}").text
    assert "Traceback" not in body
    assert "모델 답이 정해진 형식이 아니었어요" in body
