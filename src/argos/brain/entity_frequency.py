"""흔한 이름 걸러내기 — ARG-302.

추출기의 문서빈도 컷은 **한 배치 안에서만** 센다. 그래서 크롤 날짜가 다르면
`ai`·`openai`처럼 코퍼스 전반에 흔한 말이 배치마다 컷을 피해 남는다. 저장된
이름 전체의 빈도로 한 번 더 거르는 게 이 모듈이다. 낱말 목록을 손으로 두지
않고 빈도 규칙으로만 가른다 — 코퍼스가 바뀌어도 같은 규칙이 알아서 맞는다.

순수 함수만 둔다. 빈도 집계(DB 읽기)는 `entity_store`가 맡는다.
"""

from __future__ import annotations

from typing import Hashable, Mapping, TypeVar

K = TypeVar("K", bound=Hashable)


def common_names(
    *,
    document_frequency: Mapping[str, int],
    named_documents: int,
    max_ratio: float,
    min_documents: int = 1,
) -> frozenset[str]:
    """흔한 이름의 키 집합. 평가 스크립트와 ``drop_common_names``가 같이 쓴다.

    두 안전장치가 있다. (1) 한 문서에만 붙은 이름은 아무리 비율이 높아도 흔한 말이
    아니다 — 코퍼스가 작을 때 비율만 보면 모든 이름이 걸린다. (2) 이름 있는 문서가
    ``min_documents``에 못 미치면 빈도 통계가 믿을 만하지 않아 아무것도 걸지 않는다.
    분모 ``named_documents``는 이름이 1개 이상 붙은 문서 수다 — 전체 기사 수로
    나누면 이름 링크가 일부 글에만 있는 운영 DB에서 흔한 이름이 안 걸린다.
    경계값(비율 == max_ratio)은 남긴다.
    """
    if named_documents <= 0 or named_documents < min_documents:
        return frozenset()
    return frozenset(
        key
        for key, count in document_frequency.items()
        if count >= 2 and count / named_documents > max_ratio
    )


def drop_common_names(
    names_by_doc: Mapping[K, frozenset[str]],
    *,
    document_frequency: Mapping[str, int],
    named_documents: int,
    max_ratio: float,
    min_documents: int = 1,
) -> dict[K, frozenset[str]]:
    """``common_names``에 든 이름을 모든 문서에서 뺀다. 지울 게 없으면 입력 그대로."""
    common = common_names(
        document_frequency=document_frequency,
        named_documents=named_documents,
        max_ratio=max_ratio,
        min_documents=min_documents,
    )
    if not common:
        return dict(names_by_doc)
    return {doc: names - common for doc, names in names_by_doc.items()}
