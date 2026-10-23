"""같은 기술의 다른 이름 별칭 규칙 — ARG-312. 순수 함수만, DB 없이."""

from __future__ import annotations

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
