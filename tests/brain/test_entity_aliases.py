"""같은 기술의 다른 이름 별칭 규칙 — ARG-312. 순수 함수만, DB 없이."""

from __future__ import annotations

import uuid

from argos.brain import entity_aliases as ea


def test_spacing_key_drops_spaces_only():
    assert ea.spacing_key("hugging face") == "huggingface"
    assert ea.spacing_key("huggingface") == "huggingface"
    assert ea.spacing_key("claude opus 5") == "claudeopus5"


def test_version_signature_separates_versions():
    assert ea.version_signature("claude opus 5") != ea.version_signature("claude opus 5.5")
    assert ea.version_signature("scion") != ea.version_signature("scion 35b")
    assert ea.version_signature("scion") == ()
    assert ea.version_signature("scion 35b") == ("35",)
    assert ea.version_signature("flux2") == ea.version_signature("flux 2") == ("2",)
    # 숫자 덩어리 경계는 보존한다: "5 5"는 "55"가 아니다.
    assert ea.version_signature("opus 5 5") != ea.version_signature("opus 55")


def test_spacing_edges_merge_same_signature_only():
    keys = ["huggingface", "hugging face", "flux2", "flux 2", "backend", "back end", "solo"]
    edges = ea.spacing_edges(keys)
    pairs = {frozenset((e.a, e.b)) for e in edges}
    assert pairs == {
        frozenset(("huggingface", "hugging face")),
        frozenset(("flux2", "flux 2")),
        frozenset(("backend", "back end")),
    }
    assert all(e.rule == ea.AliasRule.SPACING for e in edges)


def test_spacing_edges_reject_version_mismatch():
    # 공백을 빼면 같은 키지만 숫자 덩어리가 달라 합치지 않는다.
    assert ea.spacing_edges(["opus 55", "opus 5 5"]) == []


def test_spacing_variants_collapse_to_deterministic_canonical():
    edges = ea.spacing_edges(["a b c", "ab c", "abc"])
    groups = ea.build_alias_groups(edges, {"a b c": 1, "ab c": 1, "abc": 5})
    assert len(groups) == 1
    assert groups[0].canonical == "abc"
    assert set(groups[0].aliases) == {"a b c", "ab c"}
    again = ea.build_alias_groups(list(reversed(edges)), {"a b c": 1, "ab c": 1, "abc": 5})
    assert again == groups


def test_groups_reject_edges_with_different_version_signature():
    edge = ea.AliasEdge("claude opus 5", "opus 5.5", ea.AliasRule.VENDOR_PREFIX)
    assert ea.build_alias_groups([edge], {}) == []


def test_groups_pick_canonical_by_docs_then_short_key_then_alpha():
    edges = [ea.AliasEdge("openai codex", "codex", ea.AliasRule.VENDOR_PREFIX)]
    assert ea.build_alias_groups(edges, {"openai codex": 9, "codex": 30})[0].canonical == "codex"
    # 동률이면 짧은 키
    assert ea.build_alias_groups(edges, {"openai codex": 9, "codex": 9})[0].canonical == "codex"
    # 문서 수가 더 많으면 긴 쪽이 대표다
    assert (
        ea.build_alias_groups(edges, {"openai codex": 30, "codex": 9})[0].canonical
        == "openai codex"
    )


def test_chain_of_vendor_and_spacing_edges_folds_into_one_group():
    edges = [
        ea.AliasEdge("openai codex", "codex", ea.AliasRule.VENDOR_PREFIX),
        ea.AliasEdge("open ai codex", "openai codex", ea.AliasRule.SPACING),
    ]
    groups = ea.build_alias_groups(edges, {"codex": 20, "openai codex": 9, "open ai codex": 1})
    assert len(groups) == 1
    assert groups[0].canonical == "codex"
    assert set(groups[0].aliases) == {"openai codex", "open ai codex"}
    assert groups[0].rules["openai codex"] is ea.AliasRule.VENDOR_PREFIX


# ----- 규칙 B: 회사 접두어 ------------------------------------------------- #

_KEYS = [
    # 뿌리 openai / claude / nvidia / google / microsoft / windows 11
    "openai", "openai codex", "openai api", "openai anthropic",
    "claude", "claude opus 5", "claude opus 5.5", "claude code", "claude agent",
    "claude max", "claude pro",
    "nvidia", "nvidia cuda", "nvidia gpu", "nvidia gemma",
    "google", "google gemini", "google chrome", "google gemma", "google search",
    "microsoft", "microsoft openai", "microsoft azure", "microsoft copilot",
    "windows 11", "windows 11 pro", "windows 11 home", "windows 11 enterprise",
    # 접두어 없는 쪽(합쳐질 수 있는 대표 후보)
    "codex", "api", "opus 5", "opus 5.5", "code", "agent", "max", "pro", "cuda", "gpu",
    "gemini", "chrome", "gemma", "anthropic", "pro 11",
    "anthropic api", "anthropic sdk", "anthropic console",
]
_DOCS = {
    "openai codex": 9, "openai api": 7, "openai anthropic": 3,
    "claude opus 5": 18, "claude opus 5.5": 4, "claude code": 40, "claude agent": 6,
    "claude max": 5, "claude pro": 5,
    "nvidia cuda": 3, "nvidia gpu": 6, "nvidia gemma": 5,
    "google gemini": 2, "google chrome": 4, "google gemma": 5, "google search": 3,
    "microsoft openai": 8, "microsoft azure": 4, "microsoft copilot": 4,
    "windows 11 pro": 5,
    "codex": 12, "api": 90, "opus 5": 7, "opus 5.5": 3, "code": 80, "agent": 70,
    "max": 20, "pro": 20, "cuda": 6, "gpu": 40, "gemini": 30, "chrome": 15, "gemma": 8,
    "anthropic": 50, "openai": 100, "claude": 120, "nvidia": 60, "google": 90,
    "microsoft": 70, "windows 11": 10,
}
# 실측 소문자 비율
_RATIOS = {
    "codex": 0.07, "opus 5": 0.04, "opus 5.5": 0.04, "cuda": 0.0, "chrome": 0.13,
    "gemini": 0.05, "code": 0.75, "agent": 0.88, "max": 0.49, "gemma": 0.0,
    "gpu": 0.0, "api": 0.3, "anthropic": 0.0, "pro": 0.5,
}


def _decide(common=frozenset(), ratios=None, docs=None, keys=None):
    keys = frozenset(keys or _KEYS)
    docs = docs or _DOCS
    table = _RATIOS if ratios is None else ratios
    stats = ea.VendorStats(
        keys=keys,
        roots=ea.find_roots(keys),
        doc_counts=docs,
        common=frozenset(common),
        lowercase_ratios=lambda bases: {b: table[b] for b in bases if b in table},
    )
    cands = ea.vendor_candidates(keys, stats.roots)
    decision = ea.decide_vendor_prefix(cands, stats)
    merged = {(e.a, e.b) for e in decision.edges}
    held = {(h.alias, h.base): h.reason for h in decision.held}
    return merged, held


def test_find_roots_comes_from_corpus_not_a_list():
    roots = ea.find_roots(_KEYS)
    assert {"openai", "claude", "nvidia", "google", "microsoft"} <= roots
    # 따르는 이름이 3개 미만이면 뿌리가 아니다 (codex는 따르는 이름이 없다)
    assert "codex" not in roots and "gemini" not in roots


def test_vendor_prefix_merges_distinctive_names_with_literal_evidence():
    merged, _ = _decide()
    assert ("openai codex", "codex") in merged
    assert ("claude opus 5", "opus 5") in merged
    assert ("claude opus 5.5", "opus 5.5") in merged
    assert ("nvidia cuda", "cuda") in merged
    assert ("google chrome", "chrome") in merged
    assert ("google gemini", "gemini") in merged  # 문서 2개, 경계


def test_vendor_prefix_holds_common_words_by_lowercase_ratio():
    merged, held = _decide()
    assert ("claude code", "code") not in merged
    assert held[("claude code", "code")] == ea.HOLD_BASE_LOWERCASE
    assert held[("claude agent", "agent")] == ea.HOLD_BASE_LOWERCASE


def test_vendor_prefix_holds_short_single_words():
    merged, held = _decide()
    for alias, base in [("claude max", "max"), ("claude pro", "pro"), ("openai api", "api")]:
        assert (alias, base) not in merged
        assert held[(alias, base)] == ea.HOLD_BASE_TOO_SHORT
    assert held[("nvidia gpu", "gpu")] == ea.HOLD_BASE_TOO_SHORT


def test_vendor_prefix_never_collapses_a_vendor_into_its_partner():
    merged, held = _decide()
    assert ("microsoft openai", "openai") not in merged
    assert held[("microsoft openai", "openai")] == ea.HOLD_BASE_IS_VENDOR
    assert ("openai anthropic", "anthropic") not in merged
    assert held[("openai anthropic", "anthropic")] == ea.HOLD_BASE_IS_VENDOR


def test_vendor_prefix_holds_names_that_are_common_at_read_time():
    merged, held = _decide(common={"chrome"})
    assert ("google chrome", "chrome") not in merged
    assert held[("google chrome", "chrome")] == ea.HOLD_BASE_COMMON


def test_vendor_prefix_needs_two_literal_documents():
    docs = {**_DOCS, "google gemini": 1}
    merged, held = _decide(docs=docs)
    assert ("google gemini", "gemini") not in merged
    assert held[("google gemini", "gemini")] == ea.HOLD_FEW_LITERAL_DOCS


def test_vendor_prefix_ties_between_rival_vendors_are_held():
    # nvidia gemma 5 vs google gemma 5 — 동률이면 둘 다 보류(AC4)
    merged, held = _decide()
    assert ("nvidia gemma", "gemma") not in merged and ("google gemma", "gemma") not in merged
    assert held[("nvidia gemma", "gemma")] == ea.HOLD_COMPETING_VENDOR
    assert held[("google gemma", "gemma")] == ea.HOLD_COMPETING_VENDOR


def test_vendor_prefix_clear_winner_over_rival_is_merged():
    docs = {**_DOCS, "nvidia gemma": 1, "google gemma": 9}
    keys = _KEYS
    merged, held = _decide(docs=docs, keys=keys)
    # 9 >= 3 * 1 이면 google 쪽만 합친다. nvidia gemma는 문서 1개라 어차피 보류.
    assert ("google gemma", "gemma") in merged
    assert ("nvidia gemma", "gemma") not in merged
    # 3배 미만이면 보류
    docs = {**_DOCS, "nvidia gemma": 4, "google gemma": 9}
    merged, held = _decide(docs=docs)
    assert ("google gemma", "gemma") not in merged
    assert held[("google gemma", "gemma")] == ea.HOLD_COMPETING_VENDOR


def test_vendor_prefix_unverifiable_base_is_held():
    merged, held = _decide(ratios={})  # 비율을 하나도 못 구함
    assert merged == set()
    assert held[("openai codex", "codex")] == ea.HOLD_BASE_UNVERIFIED


def test_vendor_prefix_never_merges_across_versions():
    # "windows 11" 뿌리의 "windows 11 pro" -> "pro": 서명 ("11",) != ()
    merged, held = _decide()
    assert ("windows 11 pro", "pro") not in merged
    assert held[("windows 11 pro", "pro")] == ea.HOLD_VERSION_MISMATCH
    # claude opus 5.5 는 opus 5 로 가지 않는다 (base가 opus 5.5라서 서명이 같은 쪽만)
    assert ("claude opus 5.5", "opus 5") not in merged


def test_vendor_prefix_missing_base_is_held_not_created():
    keys = [k for k in _KEYS if k != "codex"]
    merged, held = _decide(keys=keys)
    assert ("openai codex", "codex") not in merged
    assert held[("openai codex", "codex")] == ea.HOLD_BASE_MISSING


def test_vendor_decision_is_deterministic():
    first = _decide()
    assert _decide() == first


def test_lowercase_ratios_counts_case_sensitive_surface_forms():
    texts = [
        "Use the Codex agent. codex is great. CODEX too.",
        "Write code, then more code. Code review.",
        "Claude Opus 5 shipped; opus 5 was fast.",
    ]
    ratios = ea.lowercase_ratios(texts, ["codex", "code", "opus 5", "missing"])
    assert ratios["codex"] == 1 / 3
    assert ratios["code"] == 2 / 3
    assert ratios["opus 5"] == 1 / 2
    assert "missing" not in ratios  # 한 번도 안 나오면 비율을 만들지 않는다


# ----- 계획 계산 ------------------------------------------------------------ #


def _inputs(existing=(), texts=None, keys=None, docs=None, common=frozenset()):
    keys = list(keys or _KEYS + ["hugging face", "huggingface"])
    ids = {key: uuid.uuid5(uuid.NAMESPACE_DNS, key) for key in keys}
    entities = tuple(ea.EntityRow(ids[key], key, key.title()) for key in keys)
    texts = texts or [
        # 소문자 비율은 이 본문에서 계산된다.
        "Codex and Opus 5 and Opus 5.5 and CUDA and Chrome and Gemini. Codex, Gemini.",
        "Write code. Run the agent. code code code agent agent.",
    ]
    return ea.AliasInputs(
        entities=entities,
        doc_counts=docs or {**_DOCS, "hugging face": 4, "huggingface": 9},
        texts=texts,
        common=common,
        existing=tuple(existing),
    ), ids


def test_plan_adds_expected_links_and_reports_held_reasons():
    inputs, ids = _inputs()
    plan = ea.compute_alias_plan(inputs)
    added = {(link.alias_key, link.canonical_key): link.rule for link in plan.added}
    assert added[("hugging face", "huggingface")] is ea.AliasRule.SPACING
    assert added[("openai codex", "codex")] is ea.AliasRule.VENDOR_PREFIX
    # 문서 수가 더 많은 쪽(claude opus 5, 18건)이 대표다
    assert added[("opus 5", "claude opus 5")] is ea.AliasRule.VENDOR_PREFIX
    assert ("claude code", "code") not in added
    assert ("claude opus 5.5", "opus 5") not in added
    assert plan.removed == () and plan.unchanged == 0
    assert plan.held_by_reason()[ea.HOLD_BASE_LOWERCASE] >= 2
    link = next(link for link in plan.added if link.alias_key == "openai codex")
    assert link.alias_id == ids["openai codex"] and link.canonical_id == ids["codex"]
    assert link.evidence["edges"][0]["literal_docs"] == 9


def test_plan_is_idempotent_after_applying_its_own_output():
    inputs, _ = _inputs()
    first = ea.compute_alias_plan(inputs)
    applied = [ea.ExistingAlias(link.alias_id, link.canonical_id, link.rule) for link in first.added]
    again = ea.compute_alias_plan(_inputs(existing=applied)[0])
    assert again.added == () and again.removed == ()
    assert again.unchanged == len(first.added)
    assert not again.changed


def test_plan_replaces_a_stale_auto_link_but_never_touches_manual_rows():
    _, ids = _inputs()
    stale = ea.ExistingAlias(ids["claude code"], ids["code"], ea.AliasRule.VENDOR_PREFIX)
    manual = ea.ExistingAlias(ids["pro 11"], ids["pro"], ea.AliasRule.MANUAL)
    plan = ea.compute_alias_plan(_inputs(existing=[stale, manual])[0])
    assert [link.alias_key for link in plan.removed] == ["claude code"]
    # 수동 행의 두 이름은 자동 판정에서 빠진다 -> 어떤 링크에도 나오지 않는다
    touched = set()
    for link in (*plan.added, *plan.removed):
        touched.update((link.alias_key, link.canonical_key))
    assert "pro 11" not in touched and "pro" not in touched


def test_plan_recanonicalizes_when_the_best_name_changes():
    _, ids = _inputs()
    old = ea.ExistingAlias(ids["codex"], ids["openai codex"], ea.AliasRule.VENDOR_PREFIX)
    plan = ea.compute_alias_plan(_inputs(existing=[old])[0])
    # 예전 방향(codex -> openai codex)은 지우고 새 방향으로 넣는다
    assert [link.alias_key for link in plan.removed] == ["codex"]
    assert any(link.alias_key == "openai codex" for link in plan.added)


def test_plan_is_deterministic():
    inputs, _ = _inputs()
    assert ea.compute_alias_plan(inputs) == ea.compute_alias_plan(inputs)
