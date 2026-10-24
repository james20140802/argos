"""pair_check 판단 캐시 — ARG-307. 파일만 쓴다(DB·LLM 없음)."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from argos.brain.pair_check import PROMPT_VERSION, PairVerdictCache

A = uuid.UUID(int=1)
B = uuid.UUID(int=2)
MODEL = "gemma4:26b"


def _cache(path: Path) -> PairVerdictCache:
    return PairVerdictCache.load(path)


def test_missing_file_is_an_empty_cache(tmp_path):
    cache = _cache(tmp_path / "nope" / "pair_checks.jsonl")
    assert cache.get(A, B, model=MODEL, input_sha="x") is None
    assert len(cache) == 0


def test_key_is_order_insensitive(tmp_path):
    cache = _cache(tmp_path / "c.jsonl")
    cache.put(B, A, model=MODEL, input_sha="x", verdict="DIFFERENT", reason="r")
    hit = cache.get(A, B, model=MODEL, input_sha="x")
    assert hit is not None and hit.verdict == "DIFFERENT" and hit.reason == "r"


def test_model_prompt_version_and_input_hash_each_invalidate(tmp_path):
    cache = _cache(tmp_path / "c.jsonl")
    cache.put(A, B, model=MODEL, input_sha="x", verdict="SAME", reason="")
    assert cache.get(A, B, model="qwen3:8b", input_sha="x") is None
    assert cache.get(A, B, model=MODEL, input_sha="y") is None
    assert (
        cache.get(A, B, model=MODEL, input_sha="x", prompt_version=PROMPT_VERSION + 1)
        is None
    )
    assert cache.get(A, B, model=MODEL, input_sha="x") is not None


def test_flush_then_reload_round_trips(tmp_path):
    path = tmp_path / "state" / "c.jsonl"
    cache = _cache(path)
    cache.put(A, B, model=MODEL, input_sha="x", verdict="DIFFERENT", reason="다른 고객")
    cache.flush()
    again = _cache(path)
    hit = again.get(B, A, model=MODEL, input_sha="x")
    assert hit is not None and hit.verdict == "DIFFERENT" and hit.reason == "다른 고객"
    assert not list(path.parent.glob("*.tmp"))  # 임시 파일이 남지 않는다


def test_flush_without_changes_writes_nothing(tmp_path):
    path = tmp_path / "c.jsonl"
    _cache(path).flush()
    assert not path.exists()


def test_corrupt_lines_are_skipped_with_a_warning(tmp_path, caplog):
    path = tmp_path / "c.jsonl"
    good = _cache(path)
    good.put(A, B, model=MODEL, input_sha="x", verdict="SAME", reason="")
    good.flush()
    path.write_text(path.read_text() + "{not json\n" + '{"l": "only"}\n')
    with caplog.at_level(logging.WARNING):
        cache = _cache(path)
    assert cache.get(A, B, model=MODEL, input_sha="x") is not None
    assert len(cache) == 1
    assert "pair_check" in caplog.text
