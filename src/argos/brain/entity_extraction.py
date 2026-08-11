"""기사 속 고유명사 추출 — ARG-252.

대문자 n-gram으로 후보를 만들고 배치 내 문서빈도로 흔한 말을 걷어낸다.
LLM도 DB도 쓰지 않는다.

주 경로가 규칙 기반인 건 재현율 때문이 아니라 **구조** 때문이다. `Sonnet 5`,
`Blackwell` 같은 신제품명은 어떤 사전학습 NER에도 없어서 spaCy 단독으로는
원리적으로 못 잡는다. spaCy 보강은 ARG-253 소관이다.

문서빈도는 호출 시 넘겨받은 배치 안에서만 센다. 그래서 API가 배치 지향이고,
같은 기사라도 어떤 배치와 함께 넘겼는지에 따라 결과가 달라진다 — 의도된
동작이다. 번들 빈도표나 외부 코퍼스는 만들지 않는다.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from argos.brain.entity_names import canonical_name
from argos.config import settings

# 문장 끝은 여기서만 인정한다. `GPT-5.2`의 점은 뒤에 공백이 없어서 안 걸린다.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"'’)\]]*\s+")
# 낱말(내부 하이픈·아포스트로피·버전 점 포함) 또는 숫자.
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'’.][A-Za-z0-9]+)*|\d+(?:\.\d+)*")

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


def _sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_END.split(text) if part.strip()]


def _candidates(document: str, max_ngram: int) -> list[_Candidate]:
    """한 문서에서 대문자 n-gram 후보를 뽑는다 (필터 적용 전)."""
    found: list[_Candidate] = []

    for sentence in _sentences(document):
        tokens = _TOKEN.findall(sentence)
        run: list[tuple[int, str]] = []

        def flush() -> None:
            if not run:
                return
            window = run[:max_ngram]
            surface = " ".join(token for _, token in window)
            canonical = canonical_name(surface)
            # 한 낱말짜리 흔한 말은 이름이 아니다. 여러 낱말이면 그대로 둔다 —
            # "New York"의 New까지 잘라내면 이름이 부서진다.
            if canonical and not (len(window) == 1 and canonical in _STOPWORDS):
                found.append(
                    _Candidate(
                        canonical=canonical,
                        surface=surface,
                        word_count=len(window),
                        sentence_initial=window[0][0] == 0,
                    )
                )
            run.clear()

        for index, token in enumerate(tokens):
            if token[0].isupper():
                run.append((index, token))
            elif token[0].isdigit() and run:
                # 이름에 붙은 버전 숫자 ("Claude Sonnet 5").
                run.append((index, token))
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
