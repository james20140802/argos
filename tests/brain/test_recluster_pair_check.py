"""recluster_period의 애매한 짝 확인 배선 — ARG-307. DB·LLM 없이 조회와 확인을 갈아 끼운다."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import argos.brain.recluster as recluster
from argos.brain.event_scoring import DocumentFeatures
from argos.brain.pair_check import PairCheckReport
from argos.brain.recluster_input import NeighborPair, ReclusterDocument, ReclusterInput
from argos.config import EventDetectionConfig
from tests.conftest import requires_graph_libs

pytestmark = requires_graph_libs

_AT = datetime(2026, 8, 10, tzinfo=timezone.utc)
_START, _END = _AT, _AT


def _id(n: int) -> uuid.UUID:
    return uuid.UUID(int=n)


def _doc(n: int, event: int) -> ReclusterDocument:
    return ReclusterDocument(
        tech_item_id=_id(n),
        features=DocumentFeatures(
            embedding=(1.0, 0.0),
            names=frozenset({"x", "y"}),
            at=_AT,
            keywords=frozenset({"x"}),
            source="openai.com",
        ),
        event_ids=(_id(100 + event),),
    )


def _period() -> ReclusterInput:
    docs = (_doc(1, 1), _doc(2, 2))
    return ReclusterInput(
        documents=docs,
        neighbor_pairs=(NeighborPair(left_id=_id(1), right_id=_id(2)),),
        capped_document_ids=frozenset(),
    )


def _wire(monkeypatch, config, *, report=None, boom=False):
    async def fake_fetch(session, *, start, end):
        return _period()

    calls = []

    async def fake_check(session, documents, scored, *, config, cache, max_calls=None):
        if boom:
            raise AssertionError("pair_check가 꺼져 있는데 불렸다")
        calls.append(max_calls)
        return report

    monkeypatch.setattr(recluster, "fetch_period_input", fake_fetch)
    monkeypatch.setattr(recluster, "check_ambiguous_pairs", fake_check)
    monkeypatch.setattr(
        recluster, "settings", SimpleNamespace(user=SimpleNamespace(event_detection=config))
    )
    return calls


async def test_disabled_never_calls_the_check(monkeypatch):
    _wire(monkeypatch, EventDetectionConfig(pair_check_enabled=False), boom=True)
    result = await recluster.recluster_period(None, start=_START, end=_END)
    assert result.pair_check is None
    assert len(result.merges) == 1  # 이벤트 둘이 한 덩어리 → 합칠 후보 (기능 꺼짐 기준선)


async def test_vetoed_pair_removes_the_merge_candidate(monkeypatch):
    report = PairCheckReport(
        model="gemma4:26b", vetoed=frozenset({(_id(1), _id(2))}), band_pairs=1, asked=1
    )
    calls = _wire(monkeypatch, EventDetectionConfig(), report=report)
    result = await recluster.recluster_period(
        None, start=_START, end=_END, pair_check_max_calls=7
    )
    assert calls == [7]
    assert result.merges == ()  # 거부 반영 → 두 문서가 갈라져 합칠 후보가 사라진다
    assert result.pair_check is report


async def test_report_does_not_affect_candidate_equality(monkeypatch):
    first_report = PairCheckReport(model="m", elapsed_s=1.0)
    second_report = PairCheckReport(model="m", elapsed_s=9.0)
    _wire(monkeypatch, EventDetectionConfig(), report=first_report)
    first = await recluster.recluster_period(None, start=_START, end=_END)
    _wire(monkeypatch, EventDetectionConfig(), report=second_report)
    second = await recluster.recluster_period(None, start=_START, end=_END)
    assert first == second


def test_pair_check_defaults():
    config = EventDetectionConfig()
    assert config.pair_check_enabled is True
    assert config.pair_check_model == "gemma4:26b"
    assert config.pair_check_upper == pytest.approx(0.65)
    assert config.pair_check_same_source_only is True
    assert config.pair_check_max_calls == 300
