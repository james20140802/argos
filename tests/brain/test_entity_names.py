import pytest

from argos.brain.entity_names import canonical_name


@pytest.mark.parametrize(
    "raw",
    ["Sonnet-5", "sonnet 5", "SONNET  5", "Sonnet_5", "  Sonnet 5  ", "Sonnet—5"],
)
def test_notation_variants_fold_to_one_form(raw):
    assert canonical_name(raw) == "sonnet 5"


def test_version_decimal_point_is_preserved():
    assert canonical_name("GPT-5.2") == "gpt 5.2"
    assert canonical_name("gpt 5.2") == "gpt 5.2"


def test_trailing_and_stray_punctuation_dropped():
    assert canonical_name("Anthropic.") == "anthropic"
    assert canonical_name("Claude's") == "claudes"
    assert canonical_name("(Blackwell)") == "blackwell"


def test_longer_form_is_a_distinct_name():
    # 규칙 기반 정규화의 경계: 더 긴 이름은 다른 이름이다.
    # 'Claude Sonnet 5' -> 'Sonnet 5' 접기는 alias 사전(ARG-240) 소관이라
    # 이 함수가 하지 않는다. 이 테스트는 그 경계를 고정한다.
    assert canonical_name("Claude Sonnet 5") == "claude sonnet 5"
    assert canonical_name("Claude Sonnet 5") != canonical_name("Sonnet 5")


@pytest.mark.parametrize(
    "raw",
    ["Sonnet-5", "GPT-5.2", "Claude Sonnet 5", "Anthropic.", "  ", "NVIDIA Blackwell"],
)
def test_idempotent(raw):
    once = canonical_name(raw)
    assert canonical_name(once) == once


def test_empty_and_whitespace_only():
    assert canonical_name("") == ""
    assert canonical_name("   ") == ""


def test_deterministic_across_calls():
    assert canonical_name("Claude Sonnet 5") == canonical_name("claude   sonnet   5")
