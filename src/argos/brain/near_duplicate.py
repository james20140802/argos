"""기사 근접중복(재배포) 판정 — ARG-251.

같은 기사가 제목만 갈아끼워지거나 문단 순서만 바뀐 채 다시 들어오는 걸
잡는다. SimHash 64bit + 해밍 거리 컷이고, LLM도 DB도 쓰지 않는다.

피처는 **문자 n-gram + 등장 빈도 가중**이다. 단어 n-gram으로는 확정
임계값(해밍 <= 3)을 구조적으로 만족할 수 없다 — 제목 한 줄(6단어)이
바뀌면 단어 shingle 집합의 5~6%가 통째로 갈리고, SimHash 해밍 거리는
피처 벡터 사잇각에 비례하므로 거리 7 근처가 나온다. 문자 n-gram은
바뀐 제목의 n-gram 대부분이 이미 본문에 등장하는 것들이라 벡터 방향이
거의 움직이지 않는다(실측: 제목 교체 1, 문단 재배열 1, 무관한 기사 22).

공개 API는 해시 계산과 두 기사 쌍 비교까지다. 리스트에서 근접중복 무리를
찾는 건 clustering이고 뒤 이슈 소관이다.

알려진 한계: 판정은 본문 분량에 비례해 정확해진다. 제목 한 줄이 차지하는
비중이 클수록 재배포본도 멀어 보이기 때문이다. 한두 문장짜리 블러브는
제목이 바뀌면 놓칠 수 있다 — 글자당 정보량이 큰 한글은 특히 그렇다.
기사 전문에서는 문제가 되지 않는다.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

from argos.brain.entity_names import (
    JOIN_SYMBOLS,
    MARK_CLASS,
    STRAIGHT_QUOTES,
    opens_a_position,
)
from argos.config import settings

_HASH_BITS = 64
# 글자·숫자·결합 기호만 남기고 나머지(구두점, 공백)는 하나의 공백으로. 스크립트를
# 가리지 않는다 — 라틴 문자만 남기면 한글 기사가 통째로 빈 글이 되어 전부 SimHash
# 0으로 뭉개지고, 서로 무관한 기사끼리 같은 기사로 판정된다.
# 결합 기호(유니코드 M 계열)를 남기는 이유: 데바나가리·타이 문자는 모음이 자음에
# 붙는 부호로 적히는데 NFC는 이걸 자음에 합성해 주지 않는다. 지워 버리면 모음이
# 통째로 사라져 'कि'와 'कु'가 똑같이 'क'가 되고, 서로 다른 낱말로 쓰인 두 기사가
# 같은 기사로 판정된다.
_NON_TEXT = re.compile(rf"(?:[^\w{MARK_CLASS}]|_)+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _keep_name_symbols(match: re.Match[str]) -> str:
    """구두점 덩어리를 공백으로 바꾸되, 낱말에 붙은 '+'·'#'은 남긴다.

    이 둘은 구두점이 아니라 이름의 일부다. 지우면 'C++' 기사와 'C#' 기사가
    똑같이 'c'가 되어, 나머지 문장이 같기만 하면 거리 0이 나온다 — 서로 다른
    기술을 다룬 두 기사가 무조건 재배포본으로 판정된다.

    낱말 **뒤**에 붙은 것만 남긴다. 앞에 붙는 '#해시태그'나 마크다운 제목의
    '###'까지 남기면 구두점 차이를 지운다는 원칙이 깨진다.

    이름 **앞**에 붙은 점도 같은 이유로 남긴다 — 지우면 '.NET' 기사와 'NET'
    기사가 똑같아진다. 이름이 시작할 수 있는 자리(글머리·공백·여는 구두점),
    낱말 안의 이음부호 뒤('F#/.NET'), 그리고 곧은 따옴표 뒤('".NET"')가 그
    자리다. 판정은 고유명사 쪽 토큰화와 같은 것을 쓴다 — 갈라 두면 같은 글이
    두 경로에서 다르게 잘린다.

    그래서 줄임표의 꼬리는 안 걸린다. 앞이 또 점이라 이름이 시작할 자리가
    아니다 — 붙잡으면 'slipped...Next'와 'slipped... Next'가, 띄어쓰기만 다른
    같은 기사인데도 서로 멀어진다.
    """
    run = match.group()
    text = match.string
    before = text[match.start() - 1] if match.start() else ""
    after = text[match.end() : match.end() + 1]
    # 점 바로 앞의 글자. 덩어리 안에 있으면 거기서, 없으면 덩어리 앞에서 본다.
    ahead = run[-2] if len(run) > 1 else before
    starts_a_name = (
        opens_a_position(ahead)
        or ahead in JOIN_SYMBOLS
        or ahead in STRAIGHT_QUOTES
    )
    lead = "." if run.endswith(".") and after.isalpha() and starts_a_name else ""
    if not before.isalnum():
        return f" {lead}"
    kept = run[: len(run) - len(run.lstrip("+#"))]
    return f"{kept} {lead}"


def _normalize(text: str) -> str:
    """대소문자·구두점·공백 차이를 지운다. 재배포본은 이런 게 흔히 다르다."""
    return _NON_TEXT.sub(_keep_name_symbols, text.casefold()).strip()


def _feature_hash(feature: str) -> int:
    """프로세스 간 결정적인 64bit 해시.

    내장 `hash()`는 PYTHONHASHSEED에 따라 값이 달라져서 쓸 수 없다 —
    같은 기사가 실행마다 다른 SimHash를 갖게 된다.
    """
    return int.from_bytes(
        hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
    )


def _shingles(text: str, size: int) -> Counter[str]:
    # 같은 글을 NFC로 저장한 곳과 NFD로 저장한 곳에서 각각 긁어 오는 일이 있다.
    # 합성해 두지 않으면 바이트 표현만 다른 같은 글이 완전히 다른 피처를 낸다.
    # 호환 형태(NFKC)까지 접는다 — CJK 매체는 같은 기사를 전각 라틴으로 싣는데,
    # 접지 않으면 인코딩만 다른 같은 글이 해밍 32로 벌어져 재배포본을 놓친다.
    # ASCII 본문에는 NFKC가 아무 일도 하지 않아서 기존 해시는 그대로다.
    text = unicodedata.normalize("NFKC", text)
    normalized = _normalize(text)
    if not normalized:
        # 글자·숫자가 하나도 없는 글(렌더 전에 긁힌 자리표시자 페이지 등)은
        # 정규화하면 전부 빈 글이 된다. 그대로 두면 SimHash 0으로 뭉개져
        # 서로 무관한 것끼리 같은 기사로 판정되므로, 이럴 때만 원문을 쓴다.
        normalized = _WHITESPACE.sub(" ", text.casefold()).strip()
    if not normalized:
        return Counter()
    if len(normalized) <= size:
        return Counter([normalized])
    return Counter(normalized[i : i + size] for i in range(len(normalized) - size + 1))


def simhash(text: str, *, shingle_size: int | None = None) -> int:
    """본문의 64bit SimHash. 빈 글은 0."""
    if shingle_size is None:
        shingle_size = settings.user.event_detection.simhash_shingle_size

    features = _shingles(text, shingle_size)
    if not features:
        return 0

    columns = [0] * _HASH_BITS
    for feature, weight in features.items():
        digest = _feature_hash(feature)
        for bit in range(_HASH_BITS):
            columns[bit] += weight if (digest >> bit) & 1 else -weight

    value = 0
    for bit, column in enumerate(columns):
        if column > 0:
            value |= 1 << bit
    return value


def hamming_distance(left: int, right: int) -> int:
    """두 해시가 다른 비트 수."""
    return (left ^ right).bit_count()


def is_near_duplicate(
    left: str,
    right: str,
    *,
    max_distance: int | None = None,
    shingle_size: int | None = None,
) -> bool:
    """두 기사가 사실상 같은 기사인가.

    거리 컷을 넘기지 않으면 `[event_detection] simhash_hamming_max`를 쓴다.
    엄격함은 코드가 아니라 설정에서 조절한다.
    """
    if max_distance is None:
        max_distance = settings.user.event_detection.simhash_hamming_max

    distance = hamming_distance(
        simhash(left, shingle_size=shingle_size),
        simhash(right, shingle_size=shingle_size),
    )
    return distance <= max_distance
