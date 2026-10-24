from __future__ import annotations

from argos.brain.event_summary_input import (
    BLOCK_SEPARATOR,
    budget_chars,
    joined_length,
    pack_chunks,
)


def test_budget_subtracts_reserve_and_template():
    b = budget_chars(
        num_ctx=16384, output_reserve_tok=1024, chars_per_tok=2.5, template_chars=1000
    )
    assert b == int((16384 - 1024) * 2.5) - 1000


def test_budget_has_a_floor():
    b = budget_chars(
        num_ctx=2048, output_reserve_tok=1024, chars_per_tok=1.0, template_chars=10_000
    )
    assert b > 0


def test_within_budget_is_one_chunk():
    blocks = ["a" * 100, "b" * 100]
    assert pack_chunks(blocks, 1000) == [blocks]


def test_over_budget_splits_and_preserves_order():
    blocks = [str(i) * 400 for i in range(5)]
    chunks = pack_chunks(blocks, 900)
    assert [b for c in chunks for b in c] == blocks
    assert len(chunks) > 1
    assert all(joined_length(c) <= 900 for c in chunks)


def test_separator_counts_toward_the_budget():
    # two 450-char blocks + separator exceed 900
    chunks = pack_chunks(["x" * 450, "y" * 450], 900)
    assert len(BLOCK_SEPARATOR) > 0
    assert len(chunks) == 2


def test_oversized_block_is_truncated_into_its_own_chunk():
    chunks = pack_chunks(["z" * 5000, "small"], 1000)
    assert len(chunks[0][0]) == 1000
    assert chunks[1] == ["small"]


def test_empty_blocks_are_dropped():
    assert pack_chunks(["", "a", ""], 100) == [["a"]]
    assert pack_chunks([], 100) == []
