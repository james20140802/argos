"""spaCy 보조 경로 — ARG-253.

주 경로는 ARG-252의 대문자 n-gram + 문서빈도 필터다. 순서를 뒤집으면
`Sonnet 5`, `Blackwell` 같은 신제품명을 구조적으로 놓친다 — 사전학습 NER의
어휘에 없는 이름이기 때문이다. 여기서는 규칙 경로가 놓치는 사람·조직 이름만
주워 담는다.

spaCy는 optional extra다. 없으면 예외 없이 빈 결과를 돌려주고 재현율만
낮아진다. 기본 설치와 릴리스 CI는 이 폴백 경로로 돈다.

    uv sync --extra nlp && uv run python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

logger = logging.getLogger(__name__)

MODEL_NAME = "en_core_web_sm"

# 이름으로 취급할 라벨. DATE/PERCENT/CARDINAL 같은 건 이름이 아니다.
_NAME_LABELS = frozenset({"PERSON", "ORG", "PRODUCT", "GPE", "NORP", "FAC", "WORK_OF_ART"})


@lru_cache(maxsize=1)
def load_pipeline():
    """`en_core_web_sm` 파이프라인. 없으면 None — 예외를 올리지 않는다."""
    try:
        import spacy
    except ImportError:
        logger.debug("spaCy가 설치되어 있지 않다 — 규칙 경로만 사용한다")
        return None

    try:
        return spacy.load(MODEL_NAME)
    except OSError:
        logger.debug("spaCy 모델 %s이 없다 — 규칙 경로만 사용한다", MODEL_NAME)
        return None


def spacy_names(documents: Sequence[str]) -> list[list[str]]:
    """문서별 개체명 표면형. 파이프라인이 없으면 전부 빈 목록."""
    pipeline = load_pipeline()
    if pipeline is None:
        return [[] for _ in documents]

    results: list[list[str]] = []
    for document in documents:
        if not document.strip():
            results.append([])
            continue
        parsed = pipeline(document)
        results.append(
            [ent.text for ent in parsed.ents if ent.label_ in _NAME_LABELS]
        )
    return results
