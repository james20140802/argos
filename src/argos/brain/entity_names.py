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


def open_punctuation() -> str:
    """여는 구두점(유니코드 Ps·Pi)을 모은다.

    `mark_class()`와 같은 이유로 손으로 열거하지 않는다 — 빠뜨린 기호만큼
    조용히 틀린다. 훑는 구간도 같게 둔다(U+0300 위는 U+FF62가 마지막이고 그
    위에는 없다).

    닫는 쪽(Pe·Pf)은 일부러 뺀다. 여는 자리만 이름이 시작할 수 있는 자리다.
    곧은 따옴표는 Po라 애초에 안 들어온다 — 여는 모양과 닫는 모양이 같아
    가릴 근거가 없다.
    """
    return "".join(
        chr(code)
        for code in range(0x30000)
        if unicodedata.category(chr(code)) in {"Ps", "Pi"}
    )


OPEN_PUNCTUATION = open_punctuation()
# 정규식 문자 클래스에 그대로 끼워 넣을 수 있는 꼴. 안에 '[' ']' '\'이 들어 있어
# escape 없이 넣으면 클래스가 깨진다.
OPEN_CLASS = "".join(re.escape(character) for character in OPEN_PUNCTUATION)


# 낱말 **안에서** 이름을 잇는 부호. 토큰화(고유명사)와 정규화(근접중복)가 같은
# 정의를 써야 'F#/.NET' 같은 겹이름이 두 경로에서 같게 잘린다.
# 전각 줄표(U+2014·U+2015)는 일부러 뺀다. 그건 이름 안이 아니라 절 사이에 쓰여,
# 이름 안 부호로 치면 서로 다른 이름 둘이 없는 이름 하나로 붙는다. 아래 목록에
# 든 부호는 눈으로 구별되지 않는다: U+2010..U+2013과 U+2212(빼기 부호).
# 손대기 전에 코드포인트부터 확인할 것.
JOIN_SYMBOLS = frozenset("-'’/·‧‐‑‒–−")
JOIN_CLASS = "".join(re.escape(character) for character in sorted(JOIN_SYMBOLS))


def opens_a_position(character: str) -> bool:
    """이 글자 뒤가 이름이 시작할 수 있는 자리인가.

    글머리(빈 글자)와 공백, 그리고 여는 구두점이다. 이름 앞에 붙는 점('.NET')을
    문장 부호와 가르는 자리 판정이라, 정규형·토큰화·근접중복 세 경로가 같은
    정의를 써야 한다. 갈라 두면 같은 글이 경로마다 다르게 잘린다.
    """
    if not character.strip():
        return True
    return character in OPEN_PUNCTUATION

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
_DOT = re.compile(r"\.")
_SPACES = re.compile(r"\s+")


def _segment_length(text: str, start: int, step: int) -> int:
    """`start`에서 `step` 방향으로 이어지는 글자·숫자의 개수."""
    length = 0
    index = start
    while 0 <= index < len(text) and text[index].isalnum():
        length += 1
        index += step
    return length


def _resolve_dot(match: re.Match[str]) -> str:
    """점 하나를 어떻게 할지 정한다: 남길지, 구분자로 벌릴지, 지울지.

    점은 자리마다 뜻이 다르다.

    - 숫자 사이(`5.2`)는 버전이다. 그대로 둔다.
    - 이름 앞(`.NET`)은 이름의 일부다. 지우면 약어 `NET`과 구별되지 않는다.
    - 낱말 사이(`Node.js`)는 구분자다. 지우기만 하면 `nodejs`가 되어 `Node JS`
      라고 쓴 같은 제품과 다른 키가 되고 문서빈도가 둘로 쪼개진다.
    - 머리글자 사이(`U.S.`)는 붙여야 한다. 여기까지 벌리면 훨씬 흔한 표기인
      `US`와 갈라져, 고치려던 것과 똑같은 쪼개짐이 반대편에서 생긴다. 그래서
      양옆 토막이 **둘 다 두 글자 이상일 때만** 벌린다.
    - 그 밖(낱말 끝 `Corp.`)은 버린다.
    """
    text = match.string
    index = match.start()
    before = text[index - 1] if index else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    if before.isdigit() and after.isdigit():
        return "."
    if not before.isalnum():
        return "." if after.isalpha() else ""
    if not after.isalnum():
        return ""
    left = _segment_length(text, index - 1, -1)
    right = _segment_length(text, index + 1, 1)
    return " " if left > 1 and right > 1 else ""


def canonical_name(name: str) -> str:
    """표기가 다른 같은 이름을 하나의 키로 접는다. 멱등이다."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", name).casefold()
    text = _SEPARATORS.sub(" ", text)
    text = _NOISE.sub("", text)
    text = _DOT.sub(_resolve_dot, text)
    return _SPACES.sub(" ", text).strip()
