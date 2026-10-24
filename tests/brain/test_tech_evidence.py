"""LLM 없는 문장 패턴 증거 (ARG-313) — 시험판 patterns.py 이식, DB/LLM 불필요."""
from __future__ import annotations

from argos.brain.tech_evidence import PatternEvidence, find_mentions, name_patterns


def _pats(*keys):
    return name_patterns({k: [k] for k in keys})


def test_list_of_three_gives_three_pairs():
    ev = PatternEvidence()
    ev.add_text(
        "Developers now switch between Claude Code, Codex, and Cursor every day.",
        _pats("claude code", "codex", "cursor"),
    )
    assert set(ev.colist) == {
        ("claude code", "codex"),
        ("claude code", "cursor"),
        ("codex", "cursor"),
    }
    assert all(n == 1 for n in ev.colist.values())


def test_mere_co_occurrence_in_same_article_gives_nothing():
    ev = PatternEvidence()
    ev.add_text(
        "Claude Code shipped a new release. Separately, the Cursor team raised money.\n"
        "OpenRouter says Codex traffic doubled.",
        _pats("claude code", "codex", "cursor", "openrouter"),
    )
    assert not ev.colist
    assert not ev.alt


def test_alternative_phrases_link_two_names():
    ev = PatternEvidence()
    ev.add_text("OpenCode is an open alternative to Claude Code.", _pats("opencode", "claude code"))
    ev.add_text("We benchmarked vLLM vs SGLang on H100s.", _pats("vllm", "sglang"))
    assert ev.alt[("claude code", "opencode")] == 1
    assert ev.alt[("sglang", "vllm")] == 1
    assert ev.examples[("sglang", "vllm")] == ["We benchmarked vLLM vs SGLang on H100s."]


def test_is_a_phrase():
    ev = PatternEvidence()
    ev.add_text("DuckDB is an in-process analytical database.", _pats("duckdb"))
    ev.add_text("We used SQLite, a small embedded database engine.", _pats("sqlite"))
    assert ev.isa["duckdb"] == ["in-process analytical database"]
    assert ev.isa["sqlite"] == ["small embedded database engine"]


def test_nested_mentions_keep_the_longer_name():
    pats = _pats("claude", "claude code", "cursor")
    hits = find_mentions("Try Claude Code, Cursor today", pats)
    assert [k for _s, _e, k in hits] == ["claude code", "cursor"]


def test_neighbors_sorted_by_count_with_examples():
    ev = PatternEvidence()
    pats = _pats("vllm", "sglang", "ollama")
    for _ in range(3):
        ev.add_text("Engines like vLLM and SGLang matter.", pats)
    ev.add_text("Ollama vs vLLM for local use.", pats)
    nb = ev.neighbors("vllm")
    assert [(n.key, n.count) for n in nb] == [("sglang", 3), ("ollama", 1)]
    assert len(nb[0].examples) <= 2


def test_alias_surfaces_map_to_the_canonical_key():
    pats = name_patterns({"github copilot": ["github copilot", "copilot"], "cursor": ["cursor"]})
    ev = PatternEvidence()
    ev.add_text("Copilot, Cursor and friends.", pats)
    assert ev.colist[("cursor", "github copilot")] == 1
