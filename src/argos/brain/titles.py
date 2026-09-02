"""원문에서 제목을 파생하는 공용 헬퍼 — ARG-269.

``save_node``가 문서 제목을 만드는 규칙(첫 비어있지 않은 줄, 500자 절단)과
사건 명명이 LLM 실패 시 쓰는 폴백 규칙은 **같아야 한다.** 다르면 같은 원문이
문서 쪽과 사건 쪽에서 다른 제목을 갖게 된다. 그래서 규칙을 여기 한 곳에
두고 양쪽이 이 함수를 부른다.

LLM도 DB도 모르는 순수 함수다 — ``save_node``가 LLM 클라이언트 임포트를
끌고 오지 않도록 ``event_naming``이 아니라 별도 모듈에 둔다.
"""

from __future__ import annotations

MAX_TITLE_CHARS = 500
"""``tech_items.title`` / ``tech_events.title`` 컬럼 폭(String(500))과 같다."""

FALLBACK_TITLE = "Untitled"


def derive_title(raw_text: str | None) -> str:
    """*raw_text*의 첫 비어있지 않은 줄을 500자로 잘라 반환한다.

    비어 있거나 공백뿐이면 ``"Untitled"``. 예외를 던지지 않는다 — 제목
    파생이 저장 경로를 막으면 안 된다.
    """
    if not raw_text:
        return FALLBACK_TITLE
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:MAX_TITLE_CHARS]
    return FALLBACK_TITLE
