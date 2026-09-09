"""recluster_candidates — 커뮤니티 ↔ 사건 대조 (ARG-280). DB를 쓰지 않는다."""
from __future__ import annotations

import uuid

from argos.brain.recluster_candidates import derive_candidates
from argos.brain.recluster_core import Community


def _id(index: int) -> uuid.UUID:
    return uuid.UUID(int=index)


# 문서 id는 1~99, 사건 id는 101~199로 나눠 읽기 쉽게 둔다.
D1, D2, D3, D4 = _id(1), _id(2), _id(3), _id(4)
E1, E2, E3 = _id(101), _id(102), _id(103)


def test_two_events_inside_one_community_become_a_merge_candidate():
    communities = [Community(members=(D1, D2))]
    links = {D1: [E1], D2: [E2]}
    result = derive_candidates(communities, links)
    assert len(result.merges) == 1
    assert result.merges[0].event_ids == (E1, E2)


def test_a_merge_candidate_carries_its_evidence_documents():
    # AC: 사람이 표본을 열어 맞는지 판단할 수 있어야 한다.
    communities = [Community(members=(D1, D2))]
    links = {D1: [E1], D2: [E2]}
    result = derive_candidates(communities, links)
    assert result.merges[0].evidence_document_ids == (D1, D2)


def test_three_events_in_one_community_produce_every_pair():
    communities = [Community(members=(D1, D2, D3))]
    links = {D1: [E1], D2: [E2], D3: [E3]}
    result = derive_candidates(communities, links)
    assert [m.event_ids for m in result.merges] == [(E1, E2), (E1, E3), (E2, E3)]


def test_one_event_split_across_communities_becomes_a_split_candidate():
    communities = [Community(members=(D1,)), Community(members=(D2,))]
    links = {D1: [E1], D2: [E1]}
    result = derive_candidates(communities, links)
    assert len(result.splits) == 1
    assert result.splits[0].event_id == E1
    assert result.splits[0].groups == ((D1,), (D2,))


def test_an_event_wholly_inside_one_community_is_not_a_split_candidate():
    communities = [Community(members=(D1, D2))]
    links = {D1: [E1], D2: [E1]}
    result = derive_candidates(communities, links)
    assert result.splits == ()
    assert result.merges == ()


def test_unlinked_documents_never_produce_candidates():
    communities = [Community(members=(D1, D2))]
    result = derive_candidates(communities, {})
    assert result.merges == ()
    assert result.splits == ()
    assert result.is_empty() is True


def test_a_document_on_two_events_participates_for_both():
    # 1단계가 이미 한 문서가 여러 사건에 걸치는 경우를 허용한다.
    communities = [Community(members=(D1, D2))]
    links = {D1: [E1, E2], D2: [E3]}
    result = derive_candidates(communities, links)
    assert [m.event_ids for m in result.merges] == [(E1, E2), (E1, E3), (E2, E3)]


def test_output_is_ordered_deterministically():
    # AC: 같은 입력에 대해 내용과 순서가 항상 같다.
    communities = [Community(members=(D3, D4)), Community(members=(D1, D2))]
    links = {D1: [E2], D2: [E1], D3: [E1], D4: [E3]}
    first = derive_candidates(communities, links)
    second = derive_candidates(list(reversed(communities)), dict(reversed(list(links.items()))))
    assert first == second
    assert [m.event_ids for m in first.merges] == sorted(m.event_ids for m in first.merges)
    assert [s.event_id for s in first.splits] == sorted(s.event_id for s in first.splits)


def test_split_groups_are_sorted_and_documents_inside_them_too():
    communities = [Community(members=(D2, D4)), Community(members=(D1, D3))]
    links = {D1: [E1], D2: [E1], D3: [E1], D4: [E1]}
    result = derive_candidates(communities, links)
    assert result.splits[0].groups == ((D1, D3), (D2, D4))


def test_the_same_event_pair_seen_twice_is_reported_once_with_all_evidence():
    communities = [Community(members=(D1, D2)), Community(members=(D3, D4))]
    links = {D1: [E1], D2: [E2], D3: [E1], D4: [E2]}
    result = derive_candidates(communities, links)
    assert len(result.merges) == 1
    assert result.merges[0].event_ids == (E1, E2)
    assert result.merges[0].evidence_document_ids == (D1, D2, D3, D4)


def test_is_empty_is_false_when_anything_was_found():
    communities = [Community(members=(D1, D2))]
    result = derive_candidates(communities, {D1: [E1], D2: [E2]})
    assert result.is_empty() is False
