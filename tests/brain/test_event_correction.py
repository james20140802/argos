"""교정 반영의 판단 로직 단위 테스트 — DB 없이 돈다 (ARG-245)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from argos.brain.event_correction import (
    choose_largest_fragment,
    choose_survivor,
    find_conflicts,
    group_merge_candidates,
)
from argos.brain.recluster_candidates import MergeCandidate, SplitCandidate

_T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
_T1 = datetime(2026, 9, 2, tzinfo=timezone.utc)


def _ids(n: int) -> list[uuid.UUID]:
    return sorted(uuid.uuid4() for _ in range(n))


def _pair(a, b) -> MergeCandidate:
    return MergeCandidate(event_ids=tuple(sorted((a, b))), evidence_document_ids=())


def test_chained_pairs_form_one_group_and_disjoint_pairs_stay_apart():
    a, b, c, d, e = _ids(5)
    groups = group_merge_candidates([_pair(d, e), _pair(b, c), _pair(a, b)])
    assert [g.event_ids for g in groups] == [(a, b, c), (d, e)]
    assert len(groups[0].candidates) == 2


def test_no_candidates_no_groups():
    assert group_merge_candidates([]) == []


def test_survivor_is_the_event_with_more_documents():
    a, b = _ids(2)
    assert choose_survivor({a: (2, _T0), b: (5, _T1)}) == b


def test_survivor_tie_goes_to_earlier_occurred_at_then_smaller_id():
    a, b, c = _ids(3)
    assert choose_survivor({a: (3, _T1), b: (3, _T0)}) == b
    assert choose_survivor({c: (3, _T0), a: (3, _T0)}) == a


def test_largest_fragment_by_size_then_earliest_time_then_smallest_doc_id():
    d = _ids(6)
    times = {doc: _T1 for doc in d}
    assert choose_largest_fragment([(d[0],), (d[1], d[2])], times) == 1
    times_tie = {**times, d[4]: _T0}
    assert choose_largest_fragment([(d[3],), (d[4],)], times_tie) == 1  # 이른 시각
    assert choose_largest_fragment([(d[5],), (d[3],)], times) == 1      # 작은 문서 id


def test_split_event_in_a_merge_group_blocks_that_whole_group():
    a, b, c, x, y = _ids(5)
    groups = group_merge_candidates([_pair(a, b), _pair(b, c), _pair(x, y)])
    splits = [SplitCandidate(event_id=c, groups=((uuid.uuid4(),), (uuid.uuid4(),)))]
    clean, conflicted, blocked = find_conflicts(groups, splits)
    assert [g.event_ids for g in clean] == [(x, y)]
    assert [g.event_ids for g in conflicted] == [(a, b, c)]
    assert blocked == frozenset({c})
