"""근거 수 집계 — ARG-267. DB를 쓰지 않는다."""
from __future__ import annotations

from argos.brain.event_evidence import fold_near_duplicates


def test_identical_hashes_count_once():
    assert fold_near_duplicates([7, 7, 7], max_distance=3) == 1


def test_distant_hashes_count_separately():
    assert fold_near_duplicates([0b0000, 0b1111_1111], max_distance=3) == 2


def test_hashes_within_the_cut_fold_together():
    # 1비트 차이 → 한 묶음
    assert fold_near_duplicates([0b0000, 0b0001], max_distance=3) == 1


def test_missing_hashes_each_count_as_their_own():
    """백필하지 않은 기존 문서는 서로 접지 않는다."""
    assert fold_near_duplicates([None, None], max_distance=3) == 2


def test_missing_and_present_mix():
    assert fold_near_duplicates([None, 7, 7], max_distance=3) == 2


def test_the_order_of_the_input_does_not_change_the_count():
    hashes = [0b0000, 0b0001, 0b1111_1111]
    assert fold_near_duplicates(hashes, max_distance=3) == fold_near_duplicates(
        list(reversed(hashes)), max_distance=3
    )


def test_nothing_counts_as_zero():
    assert fold_near_duplicates([], max_distance=3) == 0
