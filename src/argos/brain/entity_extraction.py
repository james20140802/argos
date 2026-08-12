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
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from argos.brain import entity_spacy
from argos.brain.entity_names import canonical_name
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
_SENTENCE_END = re.compile(
    r"(?<=[.!?])[\"'’”»)\]]*\s+|(?<=[。．！？])[\"'’”»)\]』」]*\s*|[^\S\n]*\n\s*"
)
# 낱말(내부 하이픈·아포스트로피·버전 점 포함) 또는 숫자. 낱말 끝의 '+'·'#'은
# 이름의 일부다 — 버리면 'C++'와 'C#'이 둘 다 'C' 한 글자로 잘려 서로 다른
# 기술 둘이 사라지고 없는 이름 하나가 남는다.
# 붙임표는 ASCII만이 아니다. HTML 본문의 'GPT‑5'는 줄바꿈 없는 붙임표(U+2011)나
# 반각 줄표(U+2013)로 적히는 일이 흔한데, 거기서 끊으면 버전 숫자가 떨어져 나가
# 'gpt'만 남는다 — 정규형 쪽은 이미 이 부호들을 붙임표와 같게 접는다.
# 전각 줄표(U+2014·U+2015)는 일부러 뺀다. 그건 이름 안이 아니라 절 사이에 쓰여,
# 묶으면 서로 다른 이름 둘이 없는 이름 하나로 붙는다.
# 아래 문자 클래스에 든 부호는 눈으로 구별되지 않는다: U+2010..U+2013 범위와
# U+2212(빼기 부호). 손대기 전에 코드포인트부터 확인할 것.
_TOKEN = re.compile(
    r"[^\W\d_][^\W_]*(?:[-'’.‐-–−][^\W_]+)*[+#]*|\d+(?:\.\d+)*"
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
# 문장 경계 바로 앞의 낱말. 'the U.S.'에서는 머리글자 'S'만 잡힌다.
_BOUNDARY_WORD = re.compile(r"([^\W\d_]+)\.$")

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
    }
)


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


def _is_cased(character: str) -> bool:
    """대소문자 구분이 있는 글자인가. 한글·한자·가나·아랍 문자는 아니다."""
    if character.isascii():
        return character.isalpha()
    return character.isalpha() and character.lower() != character.upper()


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
    if token.isdigit() or _ROMAN.match(token) is not None:
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
    한계는 남는다: 정말로 머리글자로 끝난 문장('...a Ph.D. Later he joined')은
    다음 문장과 붙는다. 이걸 가르려면 문장 분리기가 필요한데 그건 spaCy 몫이고,
    주 경로는 spaCy 없이도 돌아야 한다.
    """
    word = _BOUNDARY_WORD.search(text[: boundary.start()])
    if word is None:
        return False
    following = text[boundary.end() : boundary.end() + 1]
    if not (following and _is_cased(following) and following.isupper()):
        return False
    return len(word.group(1)) == 1 or word.group(1).casefold() in _ABBREVIATIONS


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for boundary in _SENTENCE_END.finditer(text):
        if _ends_with_abbreviation(text, boundary):
            continue
        parts.append(text[start : boundary.start()])
        start = boundary.end()
    parts.append(text[start:])
    return [part for part in parts if part.strip(f" \t\r\n{_MASK}")]


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
    return _Candidate(
        canonical=canonical,
        surface=surface,
        word_count=len(words),
        sentence_initial=sentence_initial,
    )


def _candidates(document: str, max_ngram: int) -> list[_Candidate]:
    """한 문서에서 대문자 n-gram 후보를 뽑는다 (필터 적용 전)."""
    found: list[_Candidate] = []

    # 크롤한 글이 NFC라는 보장이 없다. 결합 기호가 분리된 채로 오면 토큰이
    # 거기서 끊겨 'François'가 'Franc'과 'ois'로 갈라진다.
    normalized = _mask_uncased(unicodedata.normalize("NFC", document))

    for sentence in _sentences(normalized):
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
                    # 표시용 원문은 낱말을 이어 붙이지 않고 원본 구간을 그대로
                    # 쓴다. 사이에 낀 이음말('&')이 사라지면 안 된다.
                    surface=sentence[window[0][2] : window[-1][3]],
                )
                if candidate is not None:
                    found.append(candidate)
            run.clear()

        previous_end = 0
        seen_content = False
        for match in _TOKEN.finditer(sentence):
            gap = sentence[previous_end : match.start()].strip()
            # 낱말 사이에 공백이 아닌 게 끼면(쉼표·괄호·따옴표) 거기서 이름이
            # 끊긴다. "Acme Corp, Globex"를 한 이름으로 붙이면 안 된다.
            # 이음말('&')은 예외다 — 거기서 끊으면 이름이 부서진다.
            if run and gap and gap not in _JOINERS:
                flush()

            # 문장 첫 단어인지는 토큰 순번이 아니라 앞에 실제로 뭐가 있었는지로
            # 본다. 순번으로 세면 마스킹된 한글이 통째로 없던 일이 되어
            # "연구진은 Anthropic과"의 Anthropic이 문장 첫 단어로 둔갑한다.
            opening = not seen_content and _only_punctuation(gap)
            token = match.group()
            if opening and _is_enumerator(token, sentence, match.end()):
                previous_end = match.end()
                continue

            previous_end = match.end()
            initial = opening
            seen_content = True
            possessive = _POSSESSIVE.search(token)
            if possessive and token[0].isupper():
                owner = token[: possessive.start()]
                run.append((initial, owner, match.start(), match.start() + len(owner)))
                flush()
            elif token[0].isupper():
                run.append((initial, token, match.start(), match.end()))
            elif token[0].isdigit() and run:
                # 이름에 붙은 버전 숫자 ("Claude Sonnet 5").
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
