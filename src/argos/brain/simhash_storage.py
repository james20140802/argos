"""64bit SimHash를 Postgres BIGINT 자리에 담는 변환 — ARG-262.

``near_duplicate.simhash()``는 0..2^64-1의 unsigned 값을 낸다. Postgres
``BIGINT``는 signed라 2^63 이상이 그대로 들어가지 않는다. 더 넓은
``NUMERIC``으로 가는 대신 부호만 접는다 — 해밍 거리는 비트 패턴만 보므로
부호 해석은 판정에 아무 영향이 없고, ``BIGINT``가 좁고 비교가 빠르다.

되찾을 때 반드시 ``from_storage``를 거쳐야 한다. 음수인 채로
``hamming_distance``에 넣으면 파이썬 int는 무한 비트 2의 보수라
``bit_count()``가 상위 비트를 무한히 세어 거리가 터무니없이 커진다.
"""

from __future__ import annotations

_MASK = (1 << 64) - 1
_SIGN_BIT = 1 << 63


def to_storage(value: int) -> int:
    """unsigned 64bit → signed BIGINT."""
    if not 0 <= value <= _MASK:
        raise ValueError(f"SimHash must be an unsigned 64-bit value, got {value}")
    return value - (1 << 64) if value & _SIGN_BIT else value


def from_storage(value: int) -> int:
    """signed BIGINT → unsigned 64bit."""
    return value & _MASK
