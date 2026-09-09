"""커뮤니티 ↔ 사건 대조로 합칠·가를 후보를 뽑는다 — ARG-280. DB를 쓰지 않는다.

재군집이 그린 새 경계(커뮤니티)와 지금 DB에 있는 경계(사건 링크)가 어긋나는
자리가 곧 교정 후보다. 어긋남은 두 방향뿐이다:

- **합칠 후보**: 한 커뮤니티가 서로 다른 둘 이상 사건의 문서를 품었다 → 그
  사건들은 사실 한 사건이었을 수 있다.
- **가를 후보**: 한 사건의 문서가 둘 이상 커뮤니티로 흩어졌다 → 그 사건은
  사실 여러 사건이었을 수 있다.

**최소 겹침 게이트를 두지 않는다.** 걸치기만 하면 보고한다 — 이 이슈는 계산만
하고 반영은 ARG-245가 한다. 여기서 조용히 걸러 버리면 사람이 볼 기회 자체가
사라진다. 랭킹·점수화도 범위 밖이다.

**입력의 사건 id는 이미 생존 사건으로 해석돼 있어야 한다.** 해석은 조회
계층(`recluster_input.fetch_period_input`)의 몫이다. 이 함수를 순수하게 두려면
DB를 볼 수 없고, 툼스톤 해석에는 DB가 필요하기 때문이다. 그 계약 덕분에
ARG-245는 툼스톤을 병합 대상으로 받는 일이 없다.

**근거 문서를 함께 싣는 이유:** 후보만 나열하면 사람이 맞는지 틀린지 판단할
방법이 없다. 그 판정을 만든 문서 id를 같이 줘야 표본을 열어 볼 수 있다.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from argos.brain.recluster_core import Community


@dataclass(frozen=True)
class MergeCandidate:
    """합칠 후보 사건 쌍. `event_ids`는 오름차순 2-튜플."""

    event_ids: tuple[uuid.UUID, uuid.UUID]
    evidence_document_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True)
class SplitCandidate:
    """가를 후보 사건. `groups`는 그 사건 문서들이 흩어진 커뮤니티별 묶음."""

    event_id: uuid.UUID
    groups: tuple[tuple[uuid.UUID, ...], ...]


@dataclass(frozen=True)
class ReclusterCandidates:
    """한 기간의 교정 후보 전체."""

    merges: tuple[MergeCandidate, ...]
    splits: tuple[SplitCandidate, ...]

    def is_empty(self) -> bool:
        return not self.merges and not self.splits


def derive_candidates(
    communities: Sequence["Community"],
    event_links: Mapping[uuid.UUID, Sequence[uuid.UUID]],
) -> ReclusterCandidates:
    """커뮤니티와 현재 사건 링크를 대조해 후보를 만든다.

    Args:
        communities: 재군집이 그린 커뮤니티들.
        event_links: 문서 id → 그 문서가 걸린 **생존** 사건 id들. 링크가 없는
            문서는 키가 없거나 빈 시퀀스다.

    Returns:
        내용과 순서가 입력 순서에 무관한 후보 목록. 정렬 키는 사건 id다.
    """
    # 사건 쌍 → 근거 문서, 사건 → 커뮤니티별 문서 묶음. 커뮤니티 인덱스가
    # 아니라 커뮤니티의 멤버 튜플을 키로 쓰면 입력 순서에 흔들리지 않는다.
    merge_evidence: dict[tuple[uuid.UUID, uuid.UUID], set[uuid.UUID]] = {}
    split_groups: dict[uuid.UUID, dict[tuple[uuid.UUID, ...], set[uuid.UUID]]] = {}

    for community in communities:
        community_key = tuple(sorted(community.members))
        events_here: dict[uuid.UUID, set[uuid.UUID]] = {}
        for document_id in community.members:
            for event_id in event_links.get(document_id, ()):
                events_here.setdefault(event_id, set()).add(document_id)
                split_groups.setdefault(event_id, {}).setdefault(
                    community_key, set()
                ).add(document_id)

        # 이 커뮤니티가 품은 사건이 둘 이상이면 그 조합 전부가 합칠 후보다.
        for left, right in combinations(sorted(events_here), 2):
            bucket = merge_evidence.setdefault((left, right), set())
            bucket |= events_here[left] | events_here[right]

    merges = tuple(
        MergeCandidate(
            event_ids=pair,
            evidence_document_ids=tuple(sorted(documents)),
        )
        for pair, documents in sorted(merge_evidence.items())
    )

    splits = tuple(
        SplitCandidate(
            event_id=event_id,
            groups=tuple(
                tuple(sorted(documents))
                for _, documents in sorted(
                    groups.items(), key=lambda item: sorted(item[1])
                )
            ),
        )
        for event_id, groups in sorted(split_groups.items())
        if len(groups) > 1  # 한 커뮤니티에 다 있으면 가를 이유가 없다
    )

    return ReclusterCandidates(merges=merges, splits=splits)
