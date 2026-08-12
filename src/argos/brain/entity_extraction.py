"""기사 속 고유명사 추출 — ARG-252.

대문자 n-gram으로 후보를 만들고 배치 내 문서빈도로 흔한 말을 걷어낸다.
LLM도 DB도 쓰지 않는다.

주 경로가 규칙 기반인 건 재현율 때문이 아니라 **구조** 때문이다. `Sonnet 5`,
`Blackwell` 같은 신제품명은 어떤 사전학습 NER에도 없어서 spaCy 단독으로는
원리적으로 못 잡는다. spaCy 보강은 ARG-253 소관이다.

문서빈도는 호출 시 넘겨받은 배치 안에서만 센다. 그래서 API가 배치 지향이고,
같은 기사라도 어떤 배치와 함께 넘겼는지에 따라 결과가 달라진다 — 의도된
동작이다. 번들 빈도표나 외부 코퍼스는 만들지 않는다.

알려진 한계 두 가지.

1. **대소문자가 있는 표기에만 적용된다.** 한글에는 대문자가 없어 규칙 경로가
   이름과 보통 명사를 가를 근거 자체가 없다 — 한글 기사에서는 본문에 섞인
   라틴 표기 이름("Claude Sonnet 5")만 잡힌다. 한글 이름까지 잡으려면
   형태소 분석기가 필요하고, 그건 새 필수 의존성이라 이 이슈 범위 밖이다.
   근접중복 판정(`near_duplicate`)은 언어를 가리지 않는다.
2. **대문자가 길게 이어지면 이름 하나로 뽑히지 않는다.** 제목처럼 통째로
   Title Case인 문장은 동사·형용사까지 한 묶음에 들어간다. 폭 단위로 끊어
   버리는 낱말은 없게 했지만, 그 안에서 진짜 이름만 골라내지는 못한다.
   이름 사전으로 거르는 건 ARG-240 소관이다.
3. **숫자로 시작하는 낱말은 이름과 단위를 가르지 못한다.** '4chan'·'500px'를
   잡는 대가로 붙여 쓴 단위('5km' '3pm')도 후보로 남는다. 단위를 목록으로
   막으면 '4k'·'5g'처럼 진짜 이름과 겹쳐 함께 사라진다. 서수('3rd')만은
   어떤 이름과도 겹치지 않아 걸러낸다.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from typing import Sequence

from argos.brain import entity_spacy
from argos.brain.entity_names import (
    JOIN_CLASS,
    MARK_CLASS,
    OPEN_CLASS,
    canonical_name,
    opens_a_position,
)
from argos.config import settings

# 문장 끝: 종결부호 뒤의 공백, 전각 종결부호, 또는 줄바꿈.
# `GPT-5.2`의 점은 뒤에 공백이 없어서 안 걸린다. 전각 종결부호(。！？)는 뒤에
# 공백을 두지 않는 게 보통이라 공백을 요구하지 않는다.
# 크롤한 본문은 소제목·문단이 마침표 없이 줄만 바꾸는 일도 흔하다 — 한 문장으로
# 붙여 두면 문단 첫 단어가 '문장 중간 대문자'로 위장해 아래 문장 첫 단어
# 필터를 통과해 버린다.
# 닫는 기호에 활자 따옴표(”«»)까지 넣는다. 크롤한 인용문은 곧은 따옴표가 아니라
# 활자 따옴표로 닫히는데, 못 알아보면 문장이 안 끊겨 다음 문장 첫 단어가
# 문장 중간으로 위장한다.
# 줄임표(…)도 종결부호에 넣는다. 크롤한 산문에 흔한데 빠뜨리면 그 뒤 문장이
# 앞 문장에 붙어 첫 단어가 문장 중간 대문자로 위장한다.
# 마침표만 뒤에 공백을 요구한다. 'GPT-5.2'의 점과 '.NET'의 점이 문장을 끊으면
# 안 되기 때문이다. 느낌표·물음표·줄임표·고리점은 공백 없이도 문장을 닫는다 —
# CJK 본문은 종결부호 뒤에 공백을 두지 않는다.
_SENTENCE_END = re.compile(
    r"(?<=\.)[\"'’”»)\]]*\s+|(?<=[!?…。])[\"'’”»)\]』」]*\s*|[^\S\n]*\n\s*"
)
# 낱말(내부 하이픈·아포스트로피·버전 점 포함) 또는 숫자. 낱말 끝의 '+'·'#'은
# 이름의 일부다 — 버리면 'C++'와 'C#'이 둘 다 'C' 한 글자로 잘려 서로 다른
# 기술 둘이 사라지고 없는 이름 하나가 남는다.
# 붙임표는 ASCII만이 아니다. HTML 본문의 'GPT‑5'는 줄바꿈 없는 붙임표(U+2011)나
# 반각 줄표(U+2013)로 적히는 일이 흔한데, 거기서 끊으면 버전 숫자가 떨어져 나가
# 'gpt'만 남는다 — 정규형 쪽은 이미 이 부호들을 붙임표와 같게 접는다.
# 전각 줄표(U+2014·U+2015)는 일부러 뺀다. 그건 이름 안이 아니라 절 사이에 쓰여,
# 묶으면 서로 다른 이름 둘이 없는 이름 하나로 붙는다.
# 빗금도 낱말 안에 남긴다 — 'HTTP/2'는 버전 숫자를 잃고 'TCP/IP'는 둘로 쪼개진다.
# 정규형 쪽은 이미 빗금을 붙임표와 같은 구분자로 접는다.
# 어느 부호가 이 목록에 드는지는 `entity_names.JOIN_SYMBOLS`가 정한다 — 근접중복
# 쪽 정규화와 같은 것을 써야 겹이름이 두 경로에서 같게 잘린다.


# 낱말 글자: 글자·숫자(밑줄 제외)에 결합 기호를 더한 것. 결합 기호 정의는
# 정규화 쪽과 공유한다 — 갈라 두면 두 경로가 같은 이름을 다르게 자른다.
_WORD = f"(?:[^\\W_]|[{MARK_CLASS}])"
# 낱말 안에서 이름을 잇는 부호. 정의는 근접중복 쪽과 공유한다 — 갈라 두면
# 'F#/.NET' 같은 겹이름이 두 경로에서 다르게 잘린다. 가운뎃점(U+00B7·U+2027)이
# 들어 있는 이유: 'DALL·E'는 거기서 끊으면 제품 하나가 사라지고 조각 둘이
# 남는다. 목록을 나누는 가운뎃점은 보통 앞뒤에 공백이 있어서 안 걸린다.
_JOIN_SYMBOL = f"[{JOIN_CLASS}]"
# 이음부호 하나, 또는 '이음부호 + 앞점'. 뒤 토막이 점으로 시작하는 이름일 때가
# 있다 — 'F#/.NET'을 빗금에서 끊으면 정규형이 이미 한 덩어리로 접는 이름이
# 문서빈도에서 둘로 쪼개진다.
# 마침표를 이음부호 자리에 따로 두고 '\.?'로 합치지 **않는** 이유: 그러면 점이
# 이어질 때 같은 자리를 가르는 경우의 수가 갈래마다 늘어 역추적이 터진다.
_JOIN = rf"(?:{_JOIN_SYMBOL}\.?|\.)"
# 낱말 뒤에 붙는 이름 기호. 이음부호로 이어진 **토막마다** 붙을 수 있다 —
# 맨 끝에만 두면 'C++/CLI'가 'c++'와 'cli'로 갈라진다.
_SYMBOL_SUFFIX = r"[+#]*"
# 이름 앞에 붙는 점. 이름이 시작할 수 있는 자리(글머리·공백·여는 구두점)에
# 놓인 것만 품는다 — 판정은 정규형·근접중복과 같은 것을 쓴다.
# 뒤보기가 아니라 **앞보기**로 자리를 가리는 이유: 점을 못 품는 자리에서 토큰
# 자체를 버리면 안 되기 때문이다. 'slipped...Next'의 'Next'는 남아야 한다.
_LEAD_DOT = rf"(?:(?<![^\s{OPEN_CLASS}])\.)?"
# 갈래 셋을 이 순서로 본다.
#  1. 글자로 시작하는 낱말. 앞의 점은 이름의 일부라 품는다 — '.NET'에서 떼면
#     표시용 원문이 'NET'이 되어 본 그대로가 아니게 되고, 약어 'NET'과 구별할
#     방법도 사라진다. 뒤에 글자가 바로 붙을 때만이라 문장 끝 마침표는 안 걸린다.
#     줄임표의 꼬리는 앞이 또 점이라 안 걸린다 — 품으면 없는 이름 '.next'가
#     생기고 근접중복 쪽 정규화와도 어긋난다.
#  2. 순수한 숫자 — 뒤에 글자나 '이음부호+글자'가 오면 이름의 앞부분이므로 뺀다.
#     이 앞보기가 없으면 '3M'이 '3'에서 잘려 다음 갈래로 넘어가지 못한다.
#  3. 숫자로 시작하는 이름('3M' '1Password' '7-Zip'). 이 갈래가 없으면 숫자가
#     떨어져 나가고 꼬리('m' 'password' 'zip')만 남아 없는 이름이 생긴다.
_TOKEN = re.compile(
    rf"{_LEAD_DOT}[^\W\d_]{_WORD}*{_SYMBOL_SUFFIX}(?:{_JOIN}{_WORD}+{_SYMBOL_SUFFIX})*"
    rf"|\d+(?:\.\d+)*(?![^\W_]|{_JOIN}[^\W_])"
    rf"|\d+{_WORD}*{_SYMBOL_SUFFIX}(?:{_JOIN}{_WORD}+{_SYMBOL_SUFFIX})*"
)
# 이름 글자가 아닌 자리를 메우는 표시. 공백이 **아니어야** 한다 — 이 자리에서
# 이름 묶음이 끊겨야 하기 때문이다. 토큰 정규식에도 걸리지 않는다.
_MASK = "\x00"
# 소유격 어미. 이름은 소유자에서 끝난다 — "Anthropic's Claude"를 한 묶음으로
# 두면 'anthropics claude'라는 없는 이름이 되고 진짜 이름 둘이 다 사라진다.
_POSSESSIVE = re.compile(r"['’][Ss]$")
# 낱말 사이에 끼어도 이름을 가르지 않는 것. 'AT&T', 'Johnson & Johnson'에서
# 끊으면 회사 하나가 사라지고 한 글자짜리 가짜 이름이 생긴다.
_JOINERS = frozenset({"&", "＆"})
# 항목 번호를 닫는 기호. '1.'은 문장 끝 규칙이 이미 잘라 준다.
_ENUMERATOR_CLOSE = frozenset({")", "]", ":", ".", "-", "–", "—"})
# 괄호로 닫는 항목 번호. 대문자 한 글자는 이것만 번호로 인정한다 —
# 'X: Grok 5 ...'의 X는 항목 번호가 아니라 회사 이름이다.
_BRACKET_CLOSE = frozenset({")", "]"})
# 소문자 로마 숫자 목록 기호 ('(ii)'). 대문자는 일부러 뺀다 — 'MIX:' 같은
# 전부 대문자인 회사명을 목록 기호로 오인해 삼킨다.
_ROMAN = re.compile(r"^[ivxlcdm]+$")
# 항목 번호로 쓰이는 숫자. 계층 번호('2.1)')까지 받는다 — 한 덩어리 숫자만
# 보면 번호가 내용으로 세어져 목록 첫 단어가 문장 첫 단어 필터를 통과한다.
_NUMBERING = re.compile(r"^\d+(?:\.\d+)*$")
# 서수. 숫자머리 이름('4chan')을 받으면서 'the 3rd quarter'는 안 받으려면
# 이것만 빼면 된다 — 어떤 이름과도 겹치지 않는 닫힌 목록이다.
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)")
# 마침표로 끝나도 문장을 끝내지 않는 말. 뒤에 곧바로 이름이 오는 호칭만 둔다 —
# 'etc.'처럼 실제로 문장을 끝내는 말까지 넣으면 반대로 두 문장이 붙어, 다음
# 문장 첫 단어가 문장 중간 대문자로 위장한다.
_ABBREVIATIONS = frozenset(
    {
        "capt",
        "col",
        "dr",
        "gen",
        "gov",
        "hon",
        "jr",
        "lt",
        "mr",
        "mrs",
        "ms",
        "mt",
        "prof",
        "rep",
        "rev",
        "sen",
        "sgt",
        "sr",
        "st",
    }
)
_FULLWIDTH_DOT = re.compile(r"．")
# 문장 경계 바로 앞의 낱말. 'the U.S.'에서는 머리글자 'S'만 잡힌다.
_BOUNDARY_WORD = re.compile(r"([^\W\d_]+)\.$")
# 띄어 쓴 머리글자 연쇄('J. K. Rowling')를 알아보는 자리. 한 글자 앞이나 뒤에
# 또 다른 '한 글자 + 마침표'가 있는지 본다 — 'option A.'와 갈라내는 단서가
# 그것뿐이다. 앞만 보면 'J.'를, 뒤만 보면 'K.'를 놓친다.
_INITIAL_BEFORE = re.compile(r"(?:^|[\s(\[\"'“‘])[^\W\d_]\.[ \t]*$")
_INITIAL_AFTER = re.compile(r"^[^\W\d_]\.")
# 바로 앞에 붙어 있는 낱말. 'George W.'의 'George'를 잡는다.
_WORD_BEFORE = re.compile(r"[^\W_]+$")

# 대문자로 시작해도 이름이 아닌 말들. 문장 첫 단어 규칙이 대부분을 걸러 주므로
# 여기 있는 건 문장 중간에서도 대문자로 나오는 것들이다. 최소한만 둔다 —
# 목록이 길어지면 진짜 이름을 삼킨다.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "but",
        "or",
        "this",
        "that",
        "these",
        "those",
        "available",
        "new",
        "report",
        "today",
        "yesterday",
        "tomorrow",
        "however",
        "meanwhile",
        "according",
        "update",
        "news",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        # 달 이름과 1인칭 대명사도 문장 중간에서 대문자로 쓴다. 문장 첫 단어
        # 규칙이 못 걸러내므로 그냥 두면 없는 이름이 문서빈도 자리를 차지한다.
        # 한 낱말일 때만 걸러지므로 'Theresa May' 같은 이름은 그대로 남는다.
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "i",
    }
)
# 이름 안에 끼는 소문자 낱말. 여기서 끊으면 'Bank of America'가 조각 둘로
# 남고 회사 하나가 사라진다.
# 'and'는 **일부러 뺀다** — 기사에서 'and'는 이름 안의 이음말이 아니라 나열
# 기호다. 이어 붙이면 'Anthropic and Google'이 없는 이름 하나가 되면서 진짜
# 회사 둘이 다 사라진다. 'Procter & Gamble'처럼 '&'로 쓴 이름은 이미 붙는다.
# 관사 쪽('la' 'der' 'den' 'des')도 넣는다 — 이름 안에서는 전치사 뒤에 관사가
# 잇따르는 게 흔하다('de la Cruz' 'van der Waals'). 목록이 길어져도 위험이
# 늘지 않는 건 붙이는 조건이 강해서다: 앞에 이미 이름 묶음이 있고, 뒤에
# (이음말을 몇 개 건너뛰더라도) 대문자 낱말이 이어질 때만 붙는다.
_CONNECTORS = frozenset(
    {
        "of",
        "de",
        "del",
        "della",
        "di",
        "da",
        "du",
        "van",
        "von",
        "la",
        "le",
        "der",
        "den",
        "des",
        "dos",
        "das",
        "ter",
    }
)
# 이름 안에 이음말이 잇따를 수 있는 최대 개수. 'de la' 'van der'가 둘이고,
# 셋을 넘는 이름은 못 봤다. 끝을 두지 않으면 소문자 낱말이 길게 이어지는
# 문장에서 이름이 아닌 데까지 훑는다.
_CONNECTOR_RUN_MAX = 3
# 문장 중간에서 인용을 여는 기호. 두 갈래로 나눠 둔다.
# 여닫는 모양이 다른 기호(“ ‘ « 「 『)는 뒤에 공백이 있어도 여는 것으로 본다 —
# 크롤한 본문은 여는 따옴표와 첫 낱말 사이를 띄우는 일이 흔한데(줄바꿈 없는
# 공백도 마찬가지), 못 알아보면 인용문 첫 단어가 문장 중간 대문자로 위장한다.
_CLAUSE_OPEN_SPACED = frozenset({"“", "‘", "«", "『", "「"})
# 곧은 따옴표는 열고 닫는 모양이 같아서 낱말에 **붙어** 있을 때만 연다. 공백까지
# 허용하면 닫는 자리(`said "hello" Anthropic`)가 열림으로 세어져 뒤의 진짜
# 이름이 문장 첫 단어로 둔갑해 탈락한다.
# 괄호는 일부러 뺀다. 기사에서 문장 중간 괄호 안에 오는 건 문장보다 이름 쪽이
# 훨씬 흔해서('Acme Corp (Globex)'), 첫 자리로 세면 그 이름이 통째로 탈락한다.
# 문장 **머리**의 괄호는 이 목록과 무관하게 이미 첫 자리로 센다.
_CLAUSE_OPEN = _CLAUSE_OPEN_SPACED | frozenset({'"', "'"})
# 여는 기호와 그 짝. 짝을 찾아야 따옴표 **안**을 볼 수 있다. 아무 닫는 기호나
# 찾으면 'Anthropic's'의 어포스트로피가 닫는 자리로 세어진다.
_CLAUSE_CLOSE = {
    "“": "”",
    "‘": "’",
    "«": "»",
    "『": "』",
    "「": "」",
    '"': '"',
    "'": "'",
}
# 따옴표 안이 문장이라는 표시. 낱말 수와 함께 본다.
_SENTENCE_FINAL = (".", "!", "?", "…", "。")


@dataclass(frozen=True, order=True)
class ExtractedName:
    """비교용 정규형과 표시용 원문."""

    canonical: str
    surface: str


@dataclass(frozen=True)
class _Candidate:
    canonical: str
    surface: str
    word_count: int
    sentence_initial: bool
    # 소유격 's를 뗀 정규형. 뗄 게 없으면 빈 문자열.
    stem: str = ""


def _is_cased(character: str) -> bool:
    """대소문자 구분이 있는 글자인가. 한글·한자·가나·아랍 문자는 아니다."""
    if character.isascii():
        return character.isalpha()
    return character.isalpha() and character.lower() != character.upper()


def _opens_a_name(token: str) -> bool:
    """이름을 여는 낱말인가.

    첫 글자가 대문자면 이름 후보다. 안쪽에 대문자가 있는 표기도 마찬가지다 —
    'iOS' 'macOS' 'eBay' 'xAI' 'iPhone'처럼 소문자로 시작하는 상표명은 첫 글자만
    보면 통째로 사라진다. 하필 이 프로젝트가 쫓는 이름들이 그 모양이다.

    대문자가 하나도 없어도 숫자로 시작하고 글자가 섞여 있으면 이름 후보다 —
    '3.js' '4chan' '500px' '6sense'가 그 모양이다. 순수한 숫자는 뺀다. 버전
    숫자와 항목 번호가 홀로 이름이 되면 안 된다.

    서수는 걸러낸다. 'the 3rd quarter'의 '3rd'는 이름이 아니다. 단위까지
    목록으로 막지는 **않는다** — '4k'·'5g'처럼 진짜 이름과 겹쳐서, 막으면
    고치려던 것과 똑같은 누락이 반대편에서 생긴다. 대신 붙여 쓴 단위('5km')는
    후보로 남는다. 서수는 어떤 이름과도 겹치지 않는 닫힌 목록이라 다르다.
    """
    if token[0].isupper() or any(character.isupper() for character in token[1:]):
        return True
    return (
        token[0].isdigit()
        and any(character.isalpha() for character in token)
        and _ORDINAL.fullmatch(token) is None
    )


def _mask_uncased(text: str) -> str:
    """이름 글자가 될 수 없는 글자를 자리표시자로 덮는다. 길이는 그대로 둔다.

    대소문자가 없는 글자는 대문자 규칙이 이름 여부를 판정할 근거 자체가 없다.
    지우지 않고 덮는 이유는 두 가지다. 오프셋이 어긋나지 않아 표시용 원문을
    그대로 잘라 쓸 수 있고, 공백이 아닌 자리표시자가 남아 있어야 "Anthropic이
    Claude"처럼 조사를 사이에 낀 두 이름이 한 묶음으로 붙지 않는다.
    """
    if text.isascii():
        return text
    return "".join(
        _MASK if character.isalpha() and not _is_cased(character) else character
        for character in text
    )


def _only_punctuation(text: str) -> bool:
    """앞에 놓인 게 구두점·기호·공백뿐인가.

    따옴표·괄호·목록 기호 뒤도 여전히 문장의 첫 자리다. 뭐라도 있으면 문장
    중간으로 치면 인용문과 목록에서 보통 명사가 이름 행세를 한다. 반대로
    자리표시자(제어 문자)는 구두점이 아니라서 "앞에 글이 있었다"는 증거로 남는다.
    """
    return all(
        character.isspace() or unicodedata.category(character)[0] in "PS"
        for character in text
    )


def _is_enumerator(token: str, sentence: str, end: int) -> bool:
    """'1)' '[1]' '(a)'처럼 항목을 여는 번호인가.

    번호는 문장 내용이 아니라 여는 표시다. 내용으로 세면 목록 첫 단어가
    문장 첫 단어 필터를 그대로 통과해 보통 명사가 이름 행세를 한다.
    """
    closer = sentence[end:].lstrip(" \t")[:1]
    if _NUMBERING.match(token) is not None or _ROMAN.match(token) is not None:
        return closer in _ENUMERATOR_CLOSE
    if len(token) == 1 and token.isalpha():
        # 대문자 한 글자는 이름일 수 있다 ('X'). 괄호로 닫힐 때만 번호로 본다 —
        # 쌍점·붙임표까지 인정하면 진짜 이름이 목록 기호로 오인돼 사라진다.
        return closer in (_BRACKET_CLOSE if token.isupper() else _ENUMERATOR_CLOSE)
    return False


def _ends_with_abbreviation(text: str, boundary: re.Match[str]) -> bool:
    """문장 끝처럼 보이지만 실은 호칭·머리글자인가.

    'Dr. Smith'에서 끊으면 뒤따르는 진짜 이름이 문장 첫 단어로 둔갑해 탈락하고,
    정작 호칭만 이름 행세를 하며 남는다. 'U.S. Army'도 마찬가지다.

    뒤에 대문자가 올 때만 이어 붙인다 — 'expanded into the U.S. the company
    said'처럼 소문자가 오면 이름이 걸린 자리가 아니라서 붙일 이유가 없다.

    한 글자는 이름의 자리에 있을 때만 머리글자로 본다 — 점으로 이어졌거나
    ('U.S.', 'J.R.R.'), 앞뒤에 또 다른 '한 글자 + 마침표'가 있거나('J. K.'),
    앞에 이름이 붙어 있거나('George W. Bush'), 뒤에 이름이 이어지거나
    ('W. Somerset Maugham'). 한 글자 낱말을 전부 머리글자로 치면 'We selected
    option A. Customers ...'처럼 평범하게 끝난 문장이 다음 문장과 붙어,
    고치려던 것과 똑같은 위장이 반대편에서 생긴다.

    한계는 남는다: 'a Ph.D. Later he joined'처럼 진짜 약어로 끝난 문장, 그리고
    'Option A. Customers Bank sued ...'처럼 한 글자로 끝난 뒤에 대문자 낱말이
    둘 이어지는 문장은 여전히 붙는다. 이걸 가르려면 문장 분리기가 필요한데
    그건 spaCy 몫이고, 주 경로는 spaCy 없이도 돌아야 한다.
    """
    word = _BOUNDARY_WORD.search(text[: boundary.start()])
    if word is None:
        return False
    following = text[boundary.end() : boundary.end() + 1]
    if not (following and _is_cased(following) and following.isupper()):
        return False
    if word.group(1).casefold() in _ABBREVIATIONS:
        return True
    if len(word.group(1)) != 1:
        return False
    start = word.start(1)
    return (
        (start > 0 and text[start - 1] == ".")
        or _INITIAL_BEFORE.search(text[:start]) is not None
        or _INITIAL_AFTER.match(text[boundary.end() :]) is not None
        or _name_precedes(text, start)
        or (word.group(1).isupper() and _names_follow(text, boundary.end()))
    )


def _fullwidth_stop(match: re.Match[str]) -> str:
    """전각 마침표가 문장을 닫는가, 이름에 붙은 점인가.

    닫는 점이면 고리점으로 옮긴다. NFKC가 전각 마침표를 ASCII 마침표로 접는데,
    ASCII 마침표는 뒤에 공백을 요구해서 공백 없이 잇는 CJK 문장 경계를 잃기
    때문이다. 한 글자짜리끼리 바꾸므로 원문 위치는 어긋나지 않는다.

    두 자리는 이름 쪽이라 그대로 둔다.

    * 숫자 사이 — 'ＧＰＴ－５．２'의 점은 버전이다. 옮기면 거기서 끊겨 반각으로
      쓴 같은 제품과 다른 키가 된다.
    * 앞이 이름이 시작할 수 있는 자리이고 뒤에 글자가 바로 붙은 자리 —
      '．ＮＥＴ'의 앞점이다. 옮기면 이름이 통째로 사라진다. 글머리·공백에
      더해 **여는** 구두점도 그 자리다('（．ＮＥＴ）'). 문장을 닫는 점은 앞
      낱말에 붙어 있으므로 '背景です．Customers …'는 여기 걸리지 않고,
      닫는 따옴표 뒤('「引用」．Customers …')도 그대로 문장 끝으로 본다.
    """
    text = match.string
    index = match.start()
    before = text[index - 1 : index]
    after = text[index + 1 : index + 2]
    if before.isdigit() and after.isdigit():
        return match.group()
    if opens_a_position(before) and after.isalpha():
        return match.group()
    return "。"


def _clause_opener(raw_gap: str) -> str:
    """이 틈에서 인용을 여는 기호. 없으면 빈 글자.

    붙어 있으면 어느 따옴표든 연 것으로 본다. 떨어져 있으면 여닫는 모양이 다른
    기호만 인정한다 — 곧은 따옴표까지 받으면 닫는 자리가 열림으로 세어진다.
    """
    if raw_gap.endswith(tuple(_CLAUSE_OPEN)):
        return raw_gap[-1]
    stripped = raw_gap.rstrip()
    if stripped.endswith(tuple(_CLAUSE_OPEN_SPACED)):
        return stripped[-1]
    return ""


def _quotes_a_clause(sentence: str, start: int, opener: str) -> bool:
    """따옴표 안이 문장인가, 감싸서 힘준 이름 하나인가.

    'Reviewers called “Claude” promising'의 Claude는 문장의 첫 단어가 아니다 —
    강조 따옴표다. 첫 자리로 세면 한 낱말짜리라 배치 어딘가에 다른 언급이 없는
    한 통째로 탈락한다. 기사에서 제품·회사 이름을 따옴표로 감싸는 건 흔하다.

    반대로 'Analysts said, "Customers pay monthly."'는 진짜 문장이라 첫 자리로
    세야 한다. 아니면 인용문 첫 단어인 보통 명사가 이름 행세를 한다.

    가르는 근거는 따옴표 안의 내용이다: 낱말이 둘 이상이거나 종결부호로 끝나면
    문장이다. 닫는 짝이 없으면 문장으로 본다 — 열린 인용은 뒤로 이어진다.
    """
    end = sentence.find(_CLAUSE_CLOSE[opener], start)
    if end == -1:
        return True
    inside = sentence[start:end].rstrip()
    return len(_TOKEN.findall(inside)) > 1 or inside.endswith(_SENTENCE_FINAL)


def _closes_an_initial(gap: str, previous: str) -> bool:
    """낱말 사이에 낀 이 마침표가 앞 낱말의 머리글자 부호인가.

    'George W. Bush'에서 끊으면 성이 떨어져 나가 이름 하나가 반쪽 둘이 된다.
    점이 앞 낱말에 **붙어** 있고 그 낱말이 대문자 한 글자일 때만이다 — 'Acme
    Corp. Globex'처럼 여러 글자로 끝나면 평범한 문장 부호로 보고 끊는다.
    """
    return (
        len(previous) == 1
        and _is_cased(previous)
        and previous.isupper()
        and gap.startswith(".")
        and not gap[1:].strip()
    )


def _name_precedes(text: str, start: int) -> bool:
    """한 글자 바로 앞에 이름으로 쓰이는 낱말이 붙어 있는가.

    'George W. Bush'의 W는 가운데 이름이다. 앞의 'George'가 근거다 —
    'We selected option A.'의 'option'은 소문자라 여기 걸리지 않는다.
    """
    head = text[:start]
    if not head.endswith((" ", "\t")):
        return False
    word = _WORD_BEFORE.search(head.rstrip(" \t"))
    return word is not None and _opens_a_name(word.group())


def _names_follow(text: str, index: int) -> bool:
    """이 자리 뒤에 이름으로 쓰이는 낱말이 둘 이어지는가.

    'W. Somerset Maugham'의 W는 이름의 첫머리다. 앞에는 근거가 없다 —
    앞 낱말('discussed')은 소문자다. 남은 근거는 뒤에 오는 말뿐이다.

    **둘**을 요구하는 것이 'We selected option A. Customers pay monthly'와
    갈라내는 단서다: 평범하게 끝난 문장의 다음 문장은 첫 단어만 대문자이고
    그 뒤는 동사라 소문자다. 사이에 쉼표 같은 게 끼면 토큰이 그 자리에서
    안 잡혀 저절로 걸러진다 — 이름 안에 낄 것이 아니기 때문이다.

    값을 치르는 쪽은 이름이 하나뿐인 경우다. 'W. Maugham'은 여전히 끊긴다.
    하나로 낮추면 'option A. Customers pay'가 바로 붙어 버려서, 둘 중
    하나는 포기해야 한다. 사전 없이 더 갈라내지 못한다 (ARG-240).
    """
    position = index
    for _ in range(2):
        rest = text[position:]
        gap = len(rest) - len(rest.lstrip(" \t"))
        following = _TOKEN.match(text, position + gap)
        if following is None or not _opens_a_name(following.group()):
            return False
        position = following.end()
    return True


def _sentences(text: str) -> list[tuple[int, str]]:
    """문장을 (시작 위치, 본문) 쌍으로 끊는다.

    위치를 함께 주는 건 표시용 원문 때문이다. 문장 안 위치만으로는 그 조각이
    원문 어디서 왔는지 되짚을 수 없다.
    """
    parts: list[tuple[int, str]] = []
    start = 0
    for boundary in _SENTENCE_END.finditer(text):
        if _ends_with_abbreviation(text, boundary):
            continue
        parts.append((start, text[start : boundary.start()]))
        start = boundary.end()
    parts.append((start, text[start:]))
    return [
        (offset, part) for offset, part in parts if part.strip(f" \t\r\n{_MASK}")
    ]


def _introduces_a_name(sentence: str, end: int) -> bool:
    """소유격 뒤에 곧바로 다른 이름이 오는가.

    'Anthropic's Claude'의 's는 소유격이지만 'Moody's'·'McDonald's'의 's는 이름의
    일부다. 무조건 끊으면 진짜 회사 이름이 'moody'로 잘리고, spaCy가 통째로 넘긴
    'moodys'와 갈려 한 회사가 두 키로 쪼개진다.

    가릴 근거는 뒤에 오는 말뿐이다. 사전 없이 더 정확히는 못 한다 — 'Moody's
    Analytics'처럼 이름이 이어지면 여전히 소유격으로 읽는다.
    """
    rest = sentence[end:]
    gap = len(rest) - len(rest.lstrip(" \t"))
    if not gap:
        return False
    following = _TOKEN.match(rest, gap)
    return following is not None and _opens_a_name(following.group())


def _leads_to_a_name(sentence: str, end: int) -> bool:
    """이 자리 뒤에 이름이 이어지는가. 이음말은 몇 개까지 건너뛴다.

    이름 안에서는 전치사 뒤에 관사가 잇따른다 — 'José de la Cruz',
    'Ludwig van der Waals'. 바로 다음 낱말만 보면 'de' 뒤의 'la'가 이름이
    아니라서 거기서 끊기고, 사람 하나가 조각 둘로 남는다.

    건너뛰는 개수에 끝을 둔다. 안 두면 소문자 낱말이 길게 이어지는 문장을
    이름이 아닌 데까지 훑는다.
    """
    position = end
    for _ in range(_CONNECTOR_RUN_MAX):
        rest = sentence[position:]
        gap = len(rest) - len(rest.lstrip(" \t"))
        if not gap:
            return False
        following = _TOKEN.match(sentence, position + gap)
        if following is None:
            return False
        if _opens_a_name(following.group()):
            return True
        if following.group().casefold() not in _CONNECTORS:
            return False
        position = following.end()
    return False


def _candidate(
    words: Sequence[str], *, sentence_initial: bool, surface: str | None = None
) -> _Candidate | None:
    """낱말 묶음을 후보로 만든다. 이름이 될 수 없으면 None.

    `surface`를 주면 표시용 원문으로 그대로 쓴다. 낱말만 이어 붙이면 사이에
    낀 이음말이 사라져 'AT&T'가 'AT T'로 망가진다.
    """
    if surface is None:
        surface = " ".join(words)
    canonical = canonical_name(surface)
    if not canonical:
        return None
    # 글자가 하나도 없으면 이름이 아니다. 이름에 붙은 버전 숫자가 n-gram 폭
    # 경계에서 떨어져 나와 홀로 남는 경우를 막는다 ("Claude Sonnet" | "5").
    if not any(character.isalpha() for character in surface):
        return None
    # 한 낱말짜리 흔한 말은 이름이 아니다. 여러 낱말이면 그대로 둔다 —
    # "New York"의 New까지 잘라내면 이름이 부서진다. 호칭도 같이 걸러낸다 —
    # 'Dr. Smith'의 'Dr'는 마침표에서 묶음이 끊겨 혼자 남는데, 두면 사람 이름
    # 행세를 한다.
    if len(words) == 1 and (canonical in _STOPWORDS or canonical in _ABBREVIATIONS):
        return None
    trimmed = _without_possessive(surface)
    stem = canonical_name(trimmed) if trimmed is not None else ""
    return _Candidate(
        canonical=canonical,
        surface=surface,
        word_count=len(words),
        sentence_initial=sentence_initial,
        stem="" if stem == canonical else stem,
    )


def _without_possessive(surface: str) -> str | None:
    """소유격 's를 뗀 원문. 뗄 게 없으면 None.

    판정은 마지막 두 글자를 NFKC로 접어서 한다 — CJK 매체는 전각 어포스트로피로
    싣는데, 그것도 같은 소유격이다. 자르기는 원문 쪽에서 한다: 표시용은 크롤한
    표기 그대로여야 하기 때문이다. 어포스트로피와 s는 호환 형태도 한 글자씩이라
    접어도 자릿수가 어긋나지 않는다.
    """
    if len(surface) < 3:
        return None
    if _POSSESSIVE.search(unicodedata.normalize("NFKC", surface[-2:])) is None:
        return None
    return surface[:-2]


def _clusters(text: str) -> list[tuple[int, str]]:
    """앞 글자와 그 뒤에 붙는 결합 기호를 한 덩이로 끊는다.

    유니코드 합성은 이 덩이 안에서만 일어난다. 그래서 덩이별로 정규화한 결과를
    이어 붙이면 글 전체를 한 번에 정규화한 것과 같고, 대신 **어느 글자가 원문
    어디서 왔는지**를 잃지 않는다. 글자 하나씩 정규화하면 'e'+U+0301이 합성되지
    않아 이름이 다시 조각난다.
    """
    if not text:
        return []
    starts = [0] + [
        index for index in range(1, len(text)) if unicodedata.combining(text[index]) == 0
    ]
    return [
        (start, text[start:end]) for start, end in zip(starts, starts[1:] + [len(text)])
    ]


def _normalize_with_origins(document: str) -> tuple[str, list[int]]:
    """정규화한 글과, 그 글의 각 자리가 원문 어디서 왔는지의 표.

    호환 형태(NFKC)는 길이를 바꾼다('ﬁ' -> 'fi'). 접은 글에서 자른 조각을 그대로
    표시용 원문으로 쓰면 크롤한 표기가 사라지므로, 자리마다 원문 위치를 들고
    다니다가 표시할 때 원문에서 다시 자른다.
    """
    folded: list[str] = []
    origins: list[int] = []
    for start, cluster in _clusters(document):
        piece = unicodedata.normalize("NFKC", cluster)
        folded.append(piece)
        origins.extend([start] * len(piece))
    return "".join(folded), origins


def _candidates(document: str, max_ngram: int) -> list[_Candidate]:
    """한 문서에서 대문자 n-gram 후보를 뽑는다 (필터 적용 전)."""
    found: list[_Candidate] = []

    # 크롤한 글이 합성형이라는 보장이 없다. 결합 기호가 분리된 채로 오면 토큰이
    # 거기서 끊겨 'François'가 'Franc'과 'ois'로 갈라진다.
    # 호환 형태(NFKC)까지 접는 이유: CJK 매체는 라틴 글자를 전각으로 싣는데,
    # 접지 않으면 전각 기호가 토큰화에서 버려져 'Ｃ＋＋'와 'Ｃ＃'이 둘 다 'c'가
    # 되고 'ＧＰＴ－５'는 'gpt'로 잘린다. 정규형 쪽이 이미 NFKC라 갈라 둘 이유도
    # 없다 — 접기는 여기서 해야 이름이 조각나기 **전에** 걸린다.
    # 문장을 닫는 전각 마침표는 접히기 전에 고리점으로 옮긴다. 어느 점이 문장을
    # 닫고 어느 점이 이름에 붙었는지는 `_fullwidth_stop`이 가른다.
    document = _FULLWIDTH_DOT.sub(_fullwidth_stop, document)
    folded, origins = _normalize_with_origins(document)
    normalized = _mask_uncased(folded)

    def source_slice(start: int, end: int) -> str:
        """접은 글의 구간을 원문 구간으로 되짚는다."""
        stop = origins[end] if end < len(origins) else len(document)
        return document[origins[start] : stop]

    for sentence_offset, sentence in _sentences(normalized):
        # (문장 첫 단어인가, 낱말, 문장 안 시작 위치, 끝 위치)
        run: list[tuple[bool, str, int, int]] = []

        def flush() -> None:
            # n-gram 폭을 넘는 묶음은 앞부분만 남기고 버리는 대신 폭 단위로
            # 끊는다. 잘라 버리면 뒤쪽 이름이 결과 어디에도 나타나지 않는다.
            for start in range(0, len(run), max_ngram):
                window = run[start : start + max_ngram]
                candidate = _candidate(
                    [token for _, token, _, _ in window],
                    sentence_initial=window[0][0],
                    # 표시용 원문은 낱말을 이어 붙이지 않고 원문 구간을 그대로
                    # 쓴다. 사이에 낀 이음말('&')이 사라지면 안 되고, 크롤한
                    # 표기(전각 등)도 접힌 채로 보여 주면 안 된다.
                    surface=source_slice(
                        sentence_offset + window[0][2],
                        sentence_offset + window[-1][3],
                    ),
                )
                if candidate is not None:
                    found.append(candidate)
            run.clear()

        previous_end = 0
        seen_content = False
        for match in _TOKEN.finditer(sentence):
            raw_gap = sentence[previous_end : match.start()]
            gap = raw_gap.strip()
            # 낱말 사이에 공백이 아닌 게 끼면(쉼표·괄호·따옴표) 거기서 이름이
            # 끊긴다. "Acme Corp, Globex"를 한 이름으로 붙이면 안 된다.
            # 이음말('&')은 예외다 — 거기서 끊으면 이름이 부서진다.
            # 머리글자 뒤의 마침표도 예외다 — 'George W. Bush'의 점은 낱말에
            # 붙은 머리글자 부호지 구분 기호가 아니다.
            if (
                run
                and gap
                and gap not in _JOINERS
                and not _closes_an_initial(raw_gap, run[-1][1])
            ):
                flush()

            # 문장 첫 단어인지는 토큰 순번이 아니라 앞에 실제로 뭐가 있었는지로
            # 본다. 순번으로 세면 마스킹된 한글이 통째로 없던 일이 되어
            # "연구진은 Anthropic과"의 Anthropic이 문장 첫 단어로 둔갑한다.
            # 문장 중간이라도 인용·괄호가 열리면 그 안은 새 문장의 첫 자리다.
            # 아니라고 보면 'Analysts said, "Customers pay monthly."'의 첫 단어가
            # 문장 중간 대문자로 위장해 보통 명사가 이름이 된다.
            # 인용이 열려도 그 안이 문장일 때만이다 — 이름 하나를 감싼
            # 강조 따옴표까지 첫 자리로 세면 그 이름이 통째로 탈락한다.
            opener = _clause_opener(raw_gap)
            opening = (not seen_content and _only_punctuation(gap)) or (
                bool(opener) and _quotes_a_clause(sentence, match.start(), opener)
            )
            token = match.group()
            if opening and _is_enumerator(token, sentence, match.end()):
                previous_end = match.end()
                continue

            previous_end = match.end()
            initial = opening
            seen_content = True
            possessive = _POSSESSIVE.search(token)
            if (
                possessive
                and _opens_a_name(token)
                and _introduces_a_name(sentence, match.end())
            ):
                owner = token[: possessive.start()]
                run.append((initial, owner, match.start(), match.start() + len(owner)))
                flush()
            elif _opens_a_name(token):
                run.append((initial, token, match.start(), match.end()))
            elif token[0].isdigit() and run:
                # 이름에 붙은 버전 숫자 ("Claude Sonnet 5").
                run.append((initial, token, match.start(), match.end()))
            elif (
                run
                and token.casefold() in _CONNECTORS
                and _leads_to_a_name(sentence, match.end())
            ):
                # 이름 안에 끼는 소문자 낱말 ("Bank of America"). 뒤에 이름이 이어질
                # 때만 이어 붙인다 — 아니면 문장의 보통 전치사라 거기서 끊어야 한다.
                # 이음말이 잇따르는 이름('de la Cruz')이 있어 몇 개는 건너뛰고 본다.
                run.append((initial, token, match.start(), match.end()))
            else:
                flush()
        flush()

    return found


def extract_names(
    documents: Sequence[str], *, max_ngram: int | None = None
) -> list[list[ExtractedName]]:
    """배치에서 문서별 고유명사를 뽑는다. 입력 순서와 길이를 그대로 지킨다."""
    config = settings.user.event_detection
    if max_ngram is None:
        max_ngram = config.entity_max_ngram

    per_document = [_candidates(document, max_ngram) for document in documents]

    # spaCy 보강(ARG-253)은 규칙 경로 **뒤에** 붙는다. 앞에 두면 표시형 우선순위가
    # 뒤집히고, 무엇보다 주 경로가 보조 경로에 가려진다.
    if config.entity_spacy_enabled:
        for document, spans in zip(per_document, entity_spacy.spacy_names(documents)):
            for span in spans:
                # 폭을 넘는 묶음은 주 경로처럼 폭 단위로 끊는다. 앞부분만 남기고
                # 잘라 버리면 뒤쪽 이름이 결과 어디에도 나타나지 않는다.
                words = span.split()
                for start in range(0, len(words), max_ngram):
                    candidate = _candidate(
                        words[start : start + max_ngram], sentence_initial=False
                    )
                    if candidate is not None:
                        document.append(candidate)

    # 소유격은 배치에 맨 이름이 나올 때만 접는다. 무조건 떼면 's가 이름의
    # 일부인 회사("Moody's" "McDonald's")가 망가지고, 그대로 두면 같은 회사가
    # 'anthropic'과 'anthropics' 두 키로 쪼개진다. 한 문장만 보고는 가를 수
    # 없다 — 둘 다 뒤에 소문자가 온다. 배치 어딘가에 맨 이름이 나온다는 것이
    # 그 's가 소유격이었다는 증거다. 문서빈도를 세기 **전에** 접어야 한 회사가
    # 두 키로 세어지지 않는다.
    bare = {
        candidate.canonical
        for document in per_document
        for candidate in document
        if not candidate.stem
    }
    for document in per_document:
        for index, candidate in enumerate(document):
            if candidate.stem and candidate.stem in bare:
                surface = _without_possessive(candidate.surface) or candidate.surface
                document[index] = replace(
                    candidate, canonical=candidate.stem, surface=surface, stem=""
                )

    # 문장 첫 단어라서 대문자인 것과 진짜 이름을 가르는 증거: 같은 이름이
    # 배치 어딘가에서 문장 중간에도 대문자로 나오는가.
    mid_sentence = {
        candidate.canonical
        for document in per_document
        for candidate in document
        if not candidate.sentence_initial
    }

    surviving: list[dict[str, str]] = []
    for document in per_document:
        kept: dict[str, str] = {}
        for candidate in document:
            suspect = candidate.sentence_initial and candidate.word_count == 1
            if suspect and candidate.canonical not in mid_sentence:
                continue
            kept.setdefault(candidate.canonical, candidate.surface)
        surviving.append(kept)

    document_frequency = Counter(key for document in surviving for key in document)
    batch_size = len(documents)
    apply_ratio_cut = batch_size >= config.entity_df_min_batch

    results: list[list[ExtractedName]] = []
    for document in surviving:
        names = [
            ExtractedName(canonical=key, surface=surface)
            for key, surface in document.items()
            if document_frequency[key] >= config.entity_min_doc_count
            and not (
                apply_ratio_cut
                and document_frequency[key] / batch_size > config.entity_max_doc_ratio
            )
        ]
        results.append(sorted(names))

    return results
