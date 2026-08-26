"""문서에 매다는 이름의 정규화 키와 결정성 — ARG-263."""
from __future__ import annotations

from unittest.mock import MagicMock

from argos.brain import pipeline as brain_pipeline
from argos.brain.entity_extraction import ExtractedName
from argos.brain.entity_store import canonical_names, storage_key


def test_storage_key_is_the_extraction_canonical_form():
    """정규화 산식을 새로 만들지 않는다 — 추출기가 낸 canonical이 곧 키다."""
    assert storage_key(ExtractedName(canonical="anthropic", surface="Anthropic")) == "anthropic"


def test_canonical_names_deduplicates_and_sorts():
    """같은 문서를 두 번 처리해도 같은 목록이 나와야 한다 (결정성 AC)."""
    names = [
        ExtractedName(canonical="claude sonnet 5", surface="Claude Sonnet 5"),
        ExtractedName(canonical="anthropic", surface="Anthropic"),
        ExtractedName(canonical="anthropic", surface="ANTHROPIC"),
    ]
    assert canonical_names(names) == ["anthropic", "claude sonnet 5"]


def test_canonical_names_of_nothing_is_empty():
    assert canonical_names([]) == []


# ---------------------------------------------------------------------------
# 파이프라인 배선 — _attach_extracted_names가 실제로 state에 이름을 싣는가.
#
# 원래 계획은 BrainState(TypedDict)에 키를 넣고 다시 읽는 테스트였는데, 그건
# 파이썬 dict 동작 자체를 확인할 뿐 이 작업이 지키려는 행동(파이프라인이 유효한
# 문서에만 이름을 얹고, 유효하지 않은 문서는 건드리지 않는다)을 보호하지
# 못한다. 그래서 대신 파이프라인 함수를 직접 부른다.
# ---------------------------------------------------------------------------


def _state(**overrides) -> dict:
    base = {
        "raw_text": "Anthropic released Claude Sonnet 5.",
        "source_url": "https://example.com",
        "is_valid": True,
        "trust_score": 0.7,
        "summary": "s",
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
    }
    base.update(overrides)
    return base


def test_attach_extracted_names_loads_names_onto_valid_states_only(monkeypatch):
    """유효한 state에만 이름을 싣고, 유효하지 않은 state는 손대지 않는다."""
    valid_state = _state()
    invalid_state = _state(is_valid=False, raw_text="junk")

    extracted = [[ExtractedName(canonical="anthropic", surface="Anthropic")]]
    extract_mock = MagicMock(return_value=extracted)
    # 이 모듈의 다른 노드들과 같은 관례 — 모듈에 이름으로 바인딩된 전역을
    # monkeypatch로 갈아 끼운다. 기본 인자로 주입하면 이 관용구가 안 먹는다.
    monkeypatch.setattr(brain_pipeline, "extract_names", extract_mock)

    result = brain_pipeline._attach_extracted_names([valid_state, invalid_state])

    # 배치 계약: 유효한 문서만 한 번의 호출로 넘긴다.
    extract_mock.assert_called_once_with([valid_state["raw_text"]])

    assert result[0]["entity_names"] == ["anthropic"]
    assert result[0]["entity_names_extracted"] == [
        ExtractedName(canonical="anthropic", surface="Anthropic")
    ]
    # 유효하지 않은 state는 이름 키 자체가 생기지 않는다.
    assert "entity_names" not in result[1]
    assert "entity_names_extracted" not in result[1]


def test_attach_extracted_names_swallows_extraction_failure(monkeypatch):
    """추출이 터져도 크롤을 멈추지 않는다 — 빈 목록으로 이어간다."""
    valid_state = _state()
    extract_mock = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(brain_pipeline, "extract_names", extract_mock)

    result = brain_pipeline._attach_extracted_names([valid_state])

    assert result[0]["entity_names"] == []
    assert result[0]["entity_names_extracted"] == []


def test_attach_extracted_names_no_valid_states_skips_extraction(monkeypatch):
    """유효한 문서가 하나도 없으면 추출기를 아예 부르지 않는다."""
    invalid_state = _state(is_valid=False)
    extract_mock = MagicMock()
    monkeypatch.setattr(brain_pipeline, "extract_names", extract_mock)

    result = brain_pipeline._attach_extracted_names([invalid_state])

    extract_mock.assert_not_called()
    assert "entity_names" not in result[0]
