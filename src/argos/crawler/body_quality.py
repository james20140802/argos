"""본문 길이·문자 깨짐 판정 (ARG-309).

크롤러(본문 보강)·CLI(body-stats / refetch-bodies)·웹(본문 없음 표시)이
모두 이 모듈 하나로 "본문이 있는가"를 판정한다. 별도 상태 컬럼 없이
저장된 ``raw_content`` 길이만으로 가른다 (마이그레이션 없음).
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from argos.crawler._html_utils import _HTML_TAG_RE

__all__ = ["BODY_MIN_CHARS", "body_text", "has_body", "is_mojibake"]

#: 저장된 순수 본문이 이보다 짧으면 "본문 없음"으로 본다.
BODY_MIN_CHARS = 500

_WHITESPACE = re.compile(r"\s+")
# UTF-8을 Latin-1/cp1252로 잘못 읽은 흔적: C1 제어문자, "â€", "Ã" + 0x80~0xBF,
# "Â" + 0xA0~0xBF. 정상 악센트(é, ï, Å)는 걸리지 않는다.
_MOJIBAKE = re.compile("[\u0080-\u009f]|â€|Ã[\u0080-¿]|Â[ -¿]")


def body_text(raw_content: str | None, title: str | None = None) -> str:
    """``title\\n\\n`` 접두와 HTML 태그를 걷어낸 순수 본문.

    ``title``을 모르면(큐 항목 등) 첫 문단이 한 줄이고 빈 줄로 본문과
    구분되어 있을 때 그것을 제목으로 보고 뗀다. 제목 접두가 없는
    본문(add_url 형태)은 그대로 둔다.
    """
    text = (raw_content or "").strip()
    if not text:
        return ""

    head, sep, rest = text.partition("\n\n")
    head = head.strip()
    if title is not None:
        wanted = title.strip()
        if wanted and head == wanted:
            text = rest
        elif wanted and text.splitlines()[0].strip() == wanted:
            text = "\n".join(text.splitlines()[1:])
    elif sep and "\n" not in head and len(head) <= 500:
        text = rest

    if _HTML_TAG_RE.search(text):
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    return _WHITESPACE.sub(" ", text).strip()


def has_body(
    raw_content: str | None,
    title: str | None = None,
    min_chars: int = BODY_MIN_CHARS,
) -> bool:
    return len(body_text(raw_content, title)) >= min_chars


def is_mojibake(text: str | None) -> bool:
    return bool(text) and _MOJIBAKE.search(text) is not None
