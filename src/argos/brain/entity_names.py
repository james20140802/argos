"""이름 정규형(canonical name) 규칙 — ARG-250.

고유명사 표기 차이를 하나의 비교 키로 접는다. 규칙 기반만 쓰고 alias
사전/가제티어는 만들지 않는다 (그건 ARG-240 소관). 따라서 'Claude Sonnet 5'
와 'Sonnet 5'는 **다른** 이름으로 남는다 — 더 짧은 별칭으로 접는 건 사전이
있어야 결정적으로 할 수 있다.

정규형은 표시용 원문을 대체하지 않는다. 비교/중복제거 키 전용이다.
"""

from __future__ import annotations

import re
import unicodedata

# 하이픈/언더스코어/슬래시/유니코드 대시류 -> 공백
_SEPARATORS = re.compile(r"[-_/‐-―−]+")
# 글자·숫자·공백·점 이외 전부 제거 (괄호, 따옴표, 쉼표 …). 글자는 ASCII로
# 한정하지 않는다 — 'François'에서 ç를 지우면 'franois'라는 없는 이름이 된다.
# 악센트 유무를 접지는(François == Francois) 않는다: 결합 기호를 일괄로
# 지우면 일본어 탁점('が' -> 'か')처럼 글자 정체가 바뀌는 표기까지 망가진다.
_NOISE = re.compile(r"[^\w\s.]+")
# 숫자 사이가 아닌 점은 제거 (문장 끝 마침표는 버리고 5.2의 점은 남긴다)
_LONE_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
_SPACES = re.compile(r"\s+")


def canonical_name(name: str) -> str:
    """표기가 다른 같은 이름을 하나의 키로 접는다. 멱등이다."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", name).casefold()
    text = _SEPARATORS.sub(" ", text)
    text = _NOISE.sub("", text)
    text = _LONE_DOT.sub("", text)
    return _SPACES.sub(" ", text).strip()
