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


def drop_common_names(
    names_by_doc: Mapping[K, frozenset[str]],
    *,
    document_frequency: Mapping[str, int],
    named_documents: int,
    max_ratio: float,
) -> dict[K, frozenset[str]]:
    """문서빈도 비율이 ``max_ratio``를 **넘는** 이름을 모든 문서에서 뺀다.

    분모 ``named_documents``는 이름이 1개 이상 붙은 문서 수다. 전체 기사 수로
    나누면 이름 링크가 일부 글에만 있는 운영 DB에서 흔한 이름이 하나도 안
    걸린다. 이름이 붙은 문서가 없으면(0) 지울 근거가 없으니 입력을 그대로 둔다.
    경계값(비율 == max_ratio)은 남긴다.
    """
    if named_documents <= 0:
        return dict(names_by_doc)
    common = frozenset(
        key
        for key, count in document_frequency.items()
        if count / named_documents > max_ratio
    )
    if not common:
        return dict(names_by_doc)
    return {doc: names - common for doc, names in names_by_doc.items()}
