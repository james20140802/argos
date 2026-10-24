"""역할 갈래 트리 편집·검증 (ARG-311 T1/T2) — DB·LLM 없이 돈다."""
from __future__ import annotations

import pytest

from argos.brain import taxonomy as tx
from argos.brain.taxonomy import SPIKE_SEED, Branch, TaxonomyError

DESC = "설명은 열 글자를 넘는 한 문장."


def B(name, desc=DESC, children=(), id=None):
    return Branch(name=name, description=desc, children=tuple(children), id=id)


def small_tree():
    return (
        B("모델", children=[B("기초 모델", id="c1"), B("특화 모델", id="c2")], id="t1"),
        B("에이전트", children=[B("코딩 에이전트", id="c3")], id="t2"),
        B("인프라", children=[B("저장소", id="c4"), B("런타임", id="c5")], id="t3"),
    )


# ---- seed (T2) ----
def test_seed_is_valid_and_top_names_match_issue():
    assert tx.validate(SPIKE_SEED) == []
    assert [b.name for b in SPIKE_SEED] == [
        "AI 모델 및 엔진",
        "AI 에이전트 및 어시스턴트",
        "에이전트 인프라 및 도구",
        "데이터 및 개발 인프라",
        "평가",
    ]


def test_seed_splits_overbroad_dev_branch():
    data = next(b for b in SPIKE_SEED if b.name == "데이터 및 개발 인프라")
    names = [c.name for c in data.children]
    assert any("버전 관리" in n for n in names)
    assert any("런타임" in n for n in names)
    assert not any("개발 및 실행 환경" in n for n in names)


# ---- validate (T1) ----
def test_validate_top_count_bounds():
    t = small_tree()
    assert tx.validate(t) == []
    assert any("3~6" in e for e in tx.validate(t[:2]))
    many = tuple(B(f"갈래{i}", children=[B(f"하위{i}")]) for i in range(7))
    assert any("3~6" in e for e in tx.validate(many))


def test_validate_requires_children_depth_and_unique_names():
    t = list(small_tree())
    t[1] = B("에이전트")
    assert any("1개 이상" in e for e in tx.validate(tuple(t)))
    t[1] = B("에이전트", children=[B("코딩", children=[B("더 아래")])])
    assert any("2단" in e for e in tx.validate(tuple(t)))
    t[1] = B("에이전트", children=[B("모델")])
    assert any("중복" in e for e in tx.validate(tuple(t)))


def test_validate_name_and_description_rules():
    t = list(small_tree())
    t[0] = B("가", children=[B("기초 모델")])
    assert any("길이" in e for e in tx.validate(tuple(t)))
    t[0] = B("모델", desc="짧음", children=[B("기초 모델")])
    assert any("설명" in e for e in tx.validate(tuple(t)))
    t[0] = B("claude, went, seconds", children=[B("기초 모델")])
    assert any("단어 나열" in e for e in tx.validate(tuple(t)))


def test_word_list_detector_does_not_flag_normal_names():
    assert tx.looks_like_word_list("claude, went, seconds")
    assert not tx.looks_like_word_list("추론·특화 모델 (Reasoning, Specialized)")
    assert not tx.looks_like_word_list("모델 실행·서빙")


def test_vague_top_name_is_warning_not_error():
    t = list(small_tree())
    t[0] = B("AI", children=[B("기초 모델")])
    assert tx.validate(tuple(t)) == []
    assert tx.warnings(tuple(t))


# ---- edits ----
def test_rename_keeps_id_and_is_immutable():
    t = small_tree()
    out = tx.rename(t, "기초 모델", "범용 모델")
    assert out[0].children[0].name == "범용 모델"
    assert out[0].children[0].id == "c1"
    assert t[0].children[0].name == "기초 모델"


def test_rename_rejects_duplicate_and_word_list_and_unknown():
    t = small_tree()
    with pytest.raises(TaxonomyError):
        tx.rename(t, "기초 모델", "특화 모델")
    with pytest.raises(TaxonomyError):
        tx.rename(t, "기초 모델", "claude, went, seconds")
    with pytest.raises(TaxonomyError):
        tx.rename(t, "없는 갈래", "아무개")


def test_merge_children_keeps_survivor_id():
    out = tx.merge(small_tree(), "저장소", "런타임")
    infra = out[2]
    assert [c.name for c in infra.children] == ["런타임"]
    assert infra.children[0].id == "c5"


def test_merge_tops_moves_children_and_keeps_survivor_id():
    t = small_tree() + (B("평가", children=[B("벤치마크")], id="t4"),)
    out = tx.merge(t, "평가", "에이전트")
    assert [b.name for b in out] == ["모델", "에이전트", "인프라"]
    agent = out[1]
    assert agent.id == "t2"
    assert [c.name for c in agent.children] == ["코딩 에이전트", "벤치마크"]


def test_merge_rejects_cross_depth_and_self():
    t = small_tree()
    with pytest.raises(TaxonomyError):
        tx.merge(t, "저장소", "모델")
    with pytest.raises(TaxonomyError):
        tx.merge(t, "저장소", "저장소")


def test_split_child_into_siblings_in_place():
    out = tx.split(
        small_tree(),
        "저장소",
        [("데이터베이스", "데이터를 저장하고 질의하는 시스템."), ("버전 관리", "코드 이력을 관리하는 도구.")],
    )
    assert [c.name for c in out[2].children] == ["데이터베이스", "버전 관리", "런타임"]
    assert out[2].children[0].id is None


def test_split_top_requires_moves_for_every_child():
    t = small_tree()
    parts = [("모델 본체", "모델 자체를 가리키는 갈래."), ("모델 실행", "모델을 돌리는 갈래입니다.")]
    with pytest.raises(TaxonomyError) as ei:
        tx.split(t, "모델", parts)
    assert any("배치" in e for e in ei.value.errors)
    out = tx.split(t, "모델", parts, moves={"기초 모델": "모델 본체", "특화 모델": "모델 실행"})
    assert [b.name for b in out][:2] == ["모델 본체", "모델 실행"]
    assert out[0].children[0].id == "c1"


def test_split_needs_two_parts_and_fresh_names():
    t = small_tree()
    with pytest.raises(TaxonomyError):
        tx.split(t, "저장소", [("하나", DESC)])
    with pytest.raises(TaxonomyError):
        tx.split(t, "저장소", [("런타임", DESC), ("새것", DESC)])


def test_delete_leaf_and_top_force_rule():
    t = small_tree()
    out = tx.delete(t, "특화 모델")
    assert [c.name for c in out[0].children] == ["기초 모델"]
    with pytest.raises(TaxonomyError):
        tx.delete(t, "모델")
    four = t + (B("평가", children=[B("벤치마크")]),)
    out = tx.delete(four, "평가", force=True)
    assert len(out) == 3


def test_delete_that_breaks_rules_is_rejected():
    with pytest.raises(TaxonomyError):
        tx.delete(small_tree(), "코딩 에이전트")  # 위 갈래가 빈 채로 남음


# ---- file roundtrip ----
def test_file_roundtrip_preserves_korean_ids_examples(tmp_path):
    t = (
        B(
            "모델",
            children=[Branch("기초 모델", DESC, id="c1", examples=("GPT", "Llama"))],
            id="t1",
        ),
    ) + small_tree()[1:]
    p = tmp_path / "d.json"
    tx.save_file(t, p)
    assert "기초 모델" in p.read_text(encoding="utf-8")  # ensure_ascii=False
    assert tx.load_file(p) == t


def test_load_file_bad_json(tmp_path):
    p = tmp_path / "d.json"
    p.write_text("{nope", encoding="utf-8")
    with pytest.raises(TaxonomyError):
        tx.load_file(p)
