import unicodedata

import pytest

from argos.brain.entity_names import canonical_name


def test_embedded_dot_folds_like_a_separator():
    # 같은 제품을 'Node.js'로도 'Node JS'로도 쓴다. 점을 지우기만 하면 두 표기가
    # 서로 다른 키가 되어 같은 제품의 문서빈도가 둘로 쪼개진다.
    assert canonical_name("Node.js") == canonical_name("Node JS") == "node js"


def test_acronym_dots_still_collapse():
    # 반대 방향 경계. 머리글자 약어의 점까지 구분자로 접으면 훨씬 흔한 표기
    # 둘('U.S.'와 'US')이 서로 다른 키가 된다.
    assert canonical_name("U.S.") == canonical_name("US") == "us"
    assert canonical_name("J.R.R.") == "jrr"


def test_leading_dot_belongs_to_the_name():
    # '.NET'의 점은 이름의 일부다. 지우면 약어 'NET'과 구별되지 않는다.
    assert canonical_name(".NET") == ".net"
    assert canonical_name(".NET") != canonical_name("NET")


def test_combining_marks_survive_canonicalization():
    # NFKC로도 앞 글자에 합성되지 않는 결합 기호가 있다. 잡음으로 지우면 서로
    # 다른 두 이름이 한 키로 합쳐지고, 문서빈도가 뭉쳐 한쪽 표시형이 사라진다.
    with_mark = unicodedata.normalize("NFC", "Ọ́lá")
    without_mark = unicodedata.normalize("NFC", "Ọlá")
    assert canonical_name(with_mark) != canonical_name(without_mark)
    assert canonical_name(with_mark) == unicodedata.normalize("NFKC", with_mark).casefold()


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


def test_ampersand_folds_like_other_separators():
    # 규칙 경로는 'AT&T'를 낱말 둘로 보고, spaCy 경로는 'AT&T' 한 덩어리를
    # 넘긴다. 둘이 다른 키가 되면 같은 회사가 문서빈도에서 둘로 쪼개진다.
    assert canonical_name("AT&T") == "at t"
    assert canonical_name("AT & T") == "at t"
    assert canonical_name("Johnson & Johnson") == canonical_name("Johnson&Johnson")


def test_accented_letters_survive():
    # 라틴 확장 글자를 지우면 이름이 부서진다: François -> franois.
    # 악센트 유무를 접는 건(François == Francois) 하지 않는다 — 일본어
    # 탁점처럼 글자 정체를 바꾸는 결합 기호까지 같이 지워지기 때문이다.
    assert canonical_name("François Chollet") == "françois chollet"
    assert canonical_name("Müller") == "müller"
    assert canonical_name("(Gómez)") == "gómez"


@pytest.mark.parametrize(
    "raw",
    [
        "Sonnet-5",
        "GPT-5.2",
        "Claude Sonnet 5",
        "Anthropic.",
        "  ",
        "NVIDIA Blackwell",
        "François Chollet",
    ],
)
def test_idempotent(raw):
    once = canonical_name(raw)
    assert canonical_name(once) == once


def test_empty_and_whitespace_only():
    assert canonical_name("") == ""
    assert canonical_name("   ") == ""


def test_deterministic_across_calls():
    assert canonical_name("Claude Sonnet 5") == canonical_name("claude   sonnet   5")


def test_symbol_suffixed_names_keep_their_symbol():
    # 'C++'와 'C#'은 서로 다른 기술이다. 기호를 잡음으로 지우면 둘 다 'c'가
    # 되어 하나로 합쳐지고, 정작 두 이름은 어디에도 남지 않는다.
    assert canonical_name("C++") == "c++"
    assert canonical_name("C#") == "c#"
    assert canonical_name("F#") != canonical_name("F")
