"""pair_check 순수 함수·오케스트레이션 — ARG-307. LLM은 전부 가짜, DB 없음."""
from __future__ import annotations

import asyncio
import uuid

import pytest

from argos.brain.event_scoring import DocumentFeatures
from argos.brain.ollama_client import OllamaInfraError
from argos.brain.pair_check import (
    PairVerdictCache,
    build_prompt,
    check_ambiguous_pairs,
    input_sha,
    parse_verdict,
    select_ambiguous_pairs,
)
from argos.brain.recluster_core import WeightedEdge
from argos.brain.recluster_input import ReclusterDocument
from argos.config import EventDetectionConfig


def _id(n: int) -> uuid.UUID:
    return uuid.UUID(int=n)


def _edge(a: int, b: int, weight: float) -> WeightedEdge:
    low, high = sorted([_id(a), _id(b)])
    return WeightedEdge(left_id=low, right_id=high, weight=weight)


SITE = {_id(1): "openai.com", _id(2): "openai.com", _id(3): "openai.com",
        _id(4): "kimi.com", _id(5): None, _id(6): None}


def _select(edges, **kw):
    kw.setdefault("lower", 0.55)
    kw.setdefault("upper", 0.65)
    kw.setdefault("same_source_only", True)
    return select_ambiguous_pairs(edges, SITE, **kw)


def test_band_includes_lower_and_excludes_upper():
    edges = [_edge(1, 2, 0.55), _edge(1, 3, 0.65), _edge(2, 3, 0.6499), _edge(1, 2, 0.5499)]
    picked = {(e.left_id, e.right_id, e.weight) for e in _select(edges)}
    assert (_id(1), _id(2), 0.55) in picked
    assert (_id(2), _id(3), 0.6499) in picked
    assert all(w != 0.65 and w != 0.5499 for _, _, w in picked)


def test_unknown_source_is_never_the_same_source():
    assert _select([_edge(5, 6, 0.6), _edge(1, 5, 0.6)]) == ()


def test_cross_source_pairs_toggle():
    edges = [_edge(1, 4, 0.6)]
    assert _select(edges) == ()
    assert len(_select(edges, same_source_only=False)) == 1


def test_selection_is_order_independent():
    edges = [_edge(2, 3, 0.6), _edge(1, 2, 0.58), _edge(1, 3, 0.62)]
    assert _select(edges) == _select(list(reversed(edges)))
    assert [(e.left_id, e.right_id) for e in _select(edges)] == sorted(
        (e.left_id, e.right_id) for e in edges
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("SAME - 같은 출시 발표", ("SAME", "같은 출시 발표")),
        ("different: 다른 고객", ("DIFFERENT", "다른 고객")),
        ("<think>SAME? no</think>\nDIFFERENT\n이유 없음", ("DIFFERENT", "이유 없음")),
        ("DIFFERENT", ("DIFFERENT", "")),
    ],
)
def test_parse_verdict_reads_one_word(text, expected):
    assert parse_verdict(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "maybe", "SAME or DIFFERENT?", "Both same and different"])
def test_parse_verdict_unparseable_is_none(text):
    assert parse_verdict(text) is None


def test_prompt_contains_both_texts_and_clips_summaries():
    prompt = build_prompt(("제목A", "가" * 1000), ("제목B", "요약B"))
    assert "제목A" in prompt and "제목B" in prompt and "요약B" in prompt
    assert "가" * 401 not in prompt
    assert "SAME" in prompt and "DIFFERENT" in prompt


def test_input_sha_tracks_text_and_is_order_insensitive():
    a, b = ("t1", "s1"), ("t2", "s2")
    assert input_sha(a, b) == input_sha(b, a)
    assert input_sha(a, b) != input_sha(a, ("t2", "changed"))


# --- 오케스트레이션 (가짜 ask · 가짜 본문 조회 · 임시 캐시) --------------------------

MODEL = "gemma4:26b"


def _doc(n: int, source: str | None = "openai.com") -> ReclusterDocument:
    return ReclusterDocument(
        tech_item_id=_id(n),
        features=DocumentFeatures(
            embedding=None, names=frozenset(), at=None, keywords=frozenset(), source=source
        ),
        event_ids=(),
    )


async def _texts(session, ids):
    return {i: (f"title-{i.int}", f"summary-{i.int}") for i in ids}


class _Ask:
    """호출을 기록하는 가짜 LLM. `answers`는 프롬프트에 포함된 제목 → 응답."""

    def __init__(self, default="SAME 같은 발표", answers=None, fail_after=None):
        self.calls: list[str] = []
        self.default = default
        self.answers = answers or {}
        self.fail_after = fail_after

    async def __call__(self, model, prompt):
        assert model == MODEL
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise OllamaInfraError("connection refused")
        self.calls.append(prompt)
        for needle, answer in self.answers.items():
            if needle in prompt:
                return answer
        return self.default


class _Unload:
    def __init__(self):
        self.models: list[str] = []

    async def __call__(self, model):
        self.models.append(model)


def _run(tmp_path, scored, *, ask, docs=None, max_calls=None, unload=None, config=None):
    cache = PairVerdictCache.load(tmp_path / "pc.jsonl")
    docs = docs or [_doc(n) for n in range(1, 8)]
    return asyncio.run(
        check_ambiguous_pairs(
            None,
            docs,
            scored,
            config=config or EventDetectionConfig(),
            cache=cache,
            ask=ask,
            load_texts=_texts,
            max_calls=max_calls,
            unload=unload or _Unload(),
        )
    ), cache


BAND = [_edge(1, 2, 0.6), _edge(1, 3, 0.58), _edge(2, 3, 0.62)]


def test_only_band_pairs_are_asked_and_different_is_vetoed(tmp_path):
    scored = BAND + [_edge(1, 4, 0.9), _edge(1, 5, 0.3)]
    ask = _Ask(answers={"title-1\n": "DIFFERENT 다른 고객"})
    report, _ = _run(tmp_path, scored, ask=ask)
    assert len(ask.calls) == 3 and report.asked == 3
    assert report.vetoed == {(_id(1), _id(2)), (_id(1), _id(3))}
    assert {v.verdict for v in report.verdicts} == {"SAME", "DIFFERENT"}
    veto = next(v for v in report.verdicts if v.left_id == _id(1) and v.right_id == _id(2))
    assert veto.reason == "다른 고객" and veto.left_title == "title-1"


def test_second_run_asks_nothing_and_reuses_the_cache(tmp_path):
    ask = _Ask(default="DIFFERENT x")
    first, cache = _run(tmp_path, BAND, ask=ask)
    assert first.asked == 3 and first.cache_hits == 0
    second_ask = _Ask()
    second, _ = _run(tmp_path, BAND, ask=second_ask)  # 같은 캐시 파일 재로딩
    assert second_ask.calls == []
    assert second.asked == 0 and second.cache_hits == 3
    assert second.vetoed == first.vetoed  # 캐시 히트도 거부를 적용한다


def test_only_new_pairs_are_asked_on_the_next_run(tmp_path):
    _run(tmp_path, BAND, ask=_Ask())
    ask = _Ask()
    report, _ = _run(tmp_path, BAND + [_edge(2, 4, 0.6)], ask=ask)
    assert len(ask.calls) == 1 and report.asked == 1 and report.cache_hits == 3


def test_connection_error_skips_the_rest_without_raising(tmp_path):
    ask = _Ask(fail_after=1)
    unload = _Unload()
    report, _ = _run(tmp_path, BAND, ask=ask, unload=unload)
    assert report.asked == 1 and report.skipped == 2
    assert report.skipped_reason and "Ollama" in report.skipped_reason
    assert len(ask.calls) == 1  # 연결 오류 뒤로는 더 부르지 않는다


def test_connection_error_still_applies_cached_vetoes(tmp_path):
    _run(tmp_path, [_edge(1, 2, 0.6)], ask=_Ask(default="DIFFERENT x"))
    report, _ = _run(tmp_path, BAND, ask=_Ask(fail_after=0))
    assert report.vetoed == {(_id(1), _id(2))}
    assert report.cache_hits == 1 and report.skipped == 2


def test_max_calls_leaves_the_overflow_unchecked(tmp_path):
    ask = _Ask()
    report, _ = _run(tmp_path, BAND, ask=ask, max_calls=2)
    assert len(ask.calls) == 2 and report.asked == 2
    assert report.skipped == 1 and "상한" in (report.skipped_reason or "")


def test_unparseable_answers_are_not_cached_or_vetoed(tmp_path):
    report, cache = _run(tmp_path, BAND, ask=_Ask(default="모르겠다"))
    assert report.vetoed == frozenset() and report.asked == 3
    assert len(cache) == 0
    again, _ = _run(tmp_path, BAND, ask=_Ask(default="SAME"))
    assert again.asked == 3  # 다음 밤에 다시 묻는다


def test_changed_text_asks_again(tmp_path):
    _run(tmp_path, [_edge(1, 2, 0.6)], ask=_Ask())

    async def changed(session, ids):
        return {i: (f"new-{i.int}", "보강된 요약") for i in ids}

    cache = PairVerdictCache.load(tmp_path / "pc.jsonl")
    ask = _Ask()
    report = asyncio.run(
        check_ambiguous_pairs(
            None, [_doc(1), _doc(2)], [_edge(1, 2, 0.6)],
            config=EventDetectionConfig(), cache=cache, ask=ask,
            load_texts=changed, unload=_Unload(),
        )
    )
    assert report.asked == 1 and report.cache_hits == 0


def test_elapsed_is_recorded_and_the_model_is_unloaded_once(tmp_path):
    unload = _Unload()
    report, _ = _run(tmp_path, BAND, ask=_Ask(), unload=unload)
    assert report.elapsed_s >= 0.0
    assert unload.models == [MODEL]


def test_nothing_asked_means_nothing_to_unload(tmp_path):
    unload = _Unload()
    report, _ = _run(tmp_path, [_edge(1, 4, 0.9)], ask=_Ask(), unload=unload)
    assert report.asked == 0 and unload.models == []


def test_cross_source_pairs_are_not_asked_by_default(tmp_path):
    docs = [_doc(1, "openai.com"), _doc(2, "kimi.com")]
    ask = _Ask()
    report, _ = _run(tmp_path, [_edge(1, 2, 0.6)], ask=ask, docs=docs)
    assert ask.calls == [] and report.asked == 0


def test_cache_is_persisted_after_the_batch(tmp_path):
    _run(tmp_path, [_edge(1, 2, 0.6)], ask=_Ask(default="DIFFERENT 다른 글"))
    assert len(PairVerdictCache.load(tmp_path / "pc.jsonl")) == 1
