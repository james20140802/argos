"""64bit unsigned SimHash를 signed BIGINT 자리에 담았다 되찾는다 — ARG-262."""
from __future__ import annotations

import pytest

from argos.brain.near_duplicate import simhash
from argos.brain.simhash_storage import from_storage, to_storage


@pytest.mark.parametrize(
    "value",
    [0, 1, 2**62, 2**63 - 1, 2**63, 2**63 + 1, 2**64 - 1],
)
def test_round_trip_preserves_every_64bit_value(value):
    assert from_storage(to_storage(value)) == value


@pytest.mark.parametrize("value", [2**63, 2**64 - 1])
def test_high_values_fit_in_signed_bigint(value):
    """BIGINT 범위를 넘으면 asyncpg가 쓰기 자체를 거부한다."""
    stored = to_storage(value)
    assert -(2**63) <= stored <= 2**63 - 1


def test_a_real_article_hash_round_trips():
    value = simhash("Anthropic released Claude Sonnet 5 today. " * 20)
    assert from_storage(to_storage(value)) == value


def test_rejects_values_outside_64_bits():
    with pytest.raises(ValueError):
        to_storage(2**64)
    with pytest.raises(ValueError):
        to_storage(-1)
