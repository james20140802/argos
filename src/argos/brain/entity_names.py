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


def mark_class() -> str:
    """결합 기호(유니코드 M 계열)를 정규식 문자 클래스 조각으로 만든다.

    `\\w`는 결합 기호를 잡지 않는다. 그런데 NFC/NFKC로도 앞 글자에 합성되지
    않는 부호가 있어서('Ọ' + U+0301), 빼 두면 두 군데가 동시에 망가진다.
    정규화에서 지우면 서로 다른 두 이름이 한 키로 합쳐지고, 토큰화에서 빠뜨리면
    낱말이 거기서 끊겨 이름이 두 동강 난다.

    블록을 손으로 열거하지 않고 유니코드 속성에서 뽑는다. 열거는 빠뜨린 블록만큼
    조용히 틀린다. 훑는 구간은 U+0300..U+2FFFF — 살아 있는 문자의 결합 기호는
    전부 이 안에 있고, 그 위의 변형 선택자(U+E0100..)는 글자가 아니라 표시
    지시라 이름에 들어갈 일이 없다. 가져오는 데 15ms쯤 든다.
    """
    ranges: list[list[int]] = []
    for code in range(0x300, 0x30000):
        if unicodedata.category(chr(code))[0] != "M":
            continue
        if ranges and ranges[-1][1] == code - 1:
            ranges[-1][1] = code
        else:
            ranges.append([code, code])
    return "".join(
        chr(low) if low == high else f"{chr(low)}-{chr(high)}" for low, high in ranges
    )


# 정규화와 토큰화가 같은 정의를 써야 두 경로가 같은 키로 모인다.
MARK_CLASS = mark_class()

# 하이픈/언더스코어/슬래시/유니코드 대시류/앰퍼샌드 -> 공백. 앰퍼샌드까지
# 여기 넣는 이유는 'AT&T'와 'AT & T'가 같은 키로 모여야 하기 때문이다 —
# 규칙 경로는 낱말 둘로 보고 spaCy 경로는 한 덩어리로 넘겨서, 안 접으면
# 같은 회사가 문서빈도에서 둘로 쪼개진다.
_SEPARATORS = re.compile(r"[-_/&＆‐-―−]+")
# 글자·숫자·공백·점 이외 전부 제거 (괄호, 따옴표, 쉼표 …). 글자는 ASCII로
# 한정하지 않는다 — 'François'에서 ç를 지우면 'franois'라는 없는 이름이 된다.
# 악센트 유무를 접지는(François == Francois) 않는다: 결합 기호를 일괄로
# 지우면 일본어 탁점('が' -> 'か')처럼 글자 정체가 바뀌는 표기까지 망가진다.
# '+'와 '#'은 예외로 남긴다 — 'C++'와 'C#'은 기호가 곧 이름의 일부여서,
# 지우면 둘 다 'c'가 되어 서로 다른 기술 둘이 한 이름으로 합쳐진다.
# 결합 기호도 남긴다: `\w`가 안 잡는데 NFKC가 합성해 주지도 않는 부호가 있어서,
# 지우면 'Ọ́lá'와 'Ọlá'가 한 키로 합쳐져 서로 다른 두 이름이 하나가 된다.
_NOISE = re.compile(rf"[^\w\s.+#{MARK_CLASS}]+")
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
