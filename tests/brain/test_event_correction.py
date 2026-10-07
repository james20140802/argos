"""교정 반영의 판단 로직 단위 테스트 — DB 없이 돈다 (ARG-245)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from argos.brain.event_correction import choose_survivor, group_merge_candidates
from argos.brain.recluster_candidates import MergeCandidate

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
