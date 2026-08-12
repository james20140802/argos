"""spaCy 보조 경로 — ARG-253.

주 경로는 ARG-252의 대문자 n-gram + 문서빈도 필터다. 순서를 뒤집으면
`Sonnet 5`, `Blackwell` 같은 신제품명을 구조적으로 놓친다 — 사전학습 NER의
어휘에 없는 이름이기 때문이다. 여기서는 규칙 경로가 놓치는 사람·조직 이름만
주워 담는다.

spaCy는 optional extra다. 없으면 예외 없이 빈 결과를 돌려주고 재현율만
낮아진다. 기본 설치와 릴리스 CI는 이 폴백 경로로 돈다.

    uv sync --extra nlp
    uv run python -m spacy download "$(uv run argos config get event_detection.entity_spacy_model)"

모델 이름은 설정(`event_detection.entity_spacy_model`)이 정한다. 기본값을
문서에 적지 않는 것도 같은 이유다 — 벤치마크에 따라 바뀔 값을 산문에 박아
두면, 설정을 바꾼 운영자가 엉뚱한 모델을 받고 보조 경로는 조용히 꺼진 채로
돈다. 위 명령이 설정에서 이름을 읽어 오는 것도 그래서다.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

from argos.config import settings

logger = logging.getLogger(__name__)

# 이름으로 취급할 라벨. DATE/PERCENT/CARDINAL 같은 건 이름이 아니다.
_NAME_LABELS = frozenset({"PERSON", "ORG", "PRODUCT", "GPE", "NORP", "FAC", "WORK_OF_ART"})


@lru_cache(maxsize=4)
def _load_pipeline(model_name: str):
    """이름으로 파이프라인을 연다. 없으면 None — 예외를 올리지 않는다."""
    try:
        import spacy
    except ImportError:
        logger.debug("spaCy가 설치되어 있지 않다 — 규칙 경로만 사용한다")
        return None

    try:
        return spacy.load(model_name)
    except OSError:
        logger.debug("spaCy 모델 %s이 없다 — 규칙 경로만 사용한다", model_name)
        return None


def load_pipeline():
    """설정이 가리키는 파이프라인. 없으면 None.

    모델 이름을 코드에 박아 두면 다른 호환 모델을 설치해 벤치마크해도 보조
    경로가 조용히 꺼진 채로 돈다. 캐시 키가 이름이라 설정을 바꾸면 그 모델을
    새로 연다.
    """
    return _load_pipeline(settings.user.event_detection.entity_spacy_model)


def spacy_names(documents: Sequence[str]) -> list[list[str]]:
    """문서별 개체명 표면형. 파이프라인이 없으면 전부 빈 목록."""
    pipeline = load_pipeline()
    if pipeline is None:
        return [[] for _ in documents]

    # 배치를 통째로 `.pipe`에 넘긴다. 문서마다 따로 부르면 spaCy 내부 배치가
    # 놀고 호출 오버헤드만 문서 수만큼 쌓인다 — 결과는 똑같다.
    return [
        [ent.text for ent in parsed.ents if ent.label_ in _NAME_LABELS]
        for parsed in pipeline.pipe(documents)
    ]
