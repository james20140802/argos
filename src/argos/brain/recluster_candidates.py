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
from typing import TYPE_CHECKING, AbstractSet, Mapping, Sequence

if TYPE_CHECKING:
    from argos.brain.recluster_core import Community


@dataclass(frozen=True)
class MergeCandidate:
    """합칠 후보 사건 쌍. `event_ids`는 오름차순 2-튜플."""

    event_ids: tuple[uuid.UUID, uuid.UUID]
    evidence_document_ids: tuple[uuid.UUID, ...]


@dataclass(frozen=True)
class SplitCandidate:
    """가를 후보 사건. `groups`는 그 사건 문서들이 흩어진 커뮤니티별 묶음.

    **판정에 쓰인 문서는 재군집을 돌린 기간 안 문서뿐이다.** 사건에 문서가
    10건 걸려 있어도 그중 2건만 기간 안이면 그 2건만 보고 "2조각"이라고 말한다
    — 나머지 8건은 아예 보지 않았다. 그래서 `groups`는 "이 사건을 이렇게 가르면
    된다"는 완성된 분할이 아니라 "기간 안에서 이만큼 어긋나 보인다"는 신호다.
    반영을 맡을 ARG-245는 사건의 전체 문서를 다시 봐야 한다.
    """

    event_id: uuid.UUID
    groups: tuple[tuple[uuid.UUID, ...], ...]
    capped: bool = False
    """그 사건 문서 중 하나라도 이웃이 `candidate_k`개로 꽉 찼다(ARG-283).

    참이면 이 갈라짐은 내용이 아니라 이웃 상한 탓일 수 있다 — CPM은 안 보인
    쌍에도 대가를 물린다. 그래도 후보에서 빼지는 않는다: 이 모듈은 최소 겹침
    게이트를 두지 않는 것과 같은 이유로, 사람이 볼 기회를 조용히 없애지 않는다.
    합칠 후보에는 이 표시가 없다 — 상한은 간선을 지우기만 하므로 거짓 합침을
    만들 수 없다.
    """


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
    *,
    capped_document_ids: AbstractSet[uuid.UUID] = frozenset(),
) -> ReclusterCandidates:
    """커뮤니티와 현재 사건 링크를 대조해 후보를 만든다.

    **전제: `communities`는 진짜 파티션이어야 한다** — 한 문서가 두 커뮤니티에
    동시에 들어 있으면 안 된다. `recluster_core.detect_communities`는 이를
    보장하지만(Leiden 파티션), 이 함수를 직접 부르는 쪽(예: ARG-245)이 겹치는
    묶음을 넘기면 그 문서가 여러 커뮤니티에서 중복 집계돼 가를 후보의 조각 수가
    실제보다 부풀어 오른다.

    Args:
        communities: 재군집이 그린 커뮤니티들. 서로 겹치지 않아야 한다.
        event_links: 문서 id → 그 문서가 걸린 **생존** 사건 id들. 링크가 없는
            문서는 키가 없거나 빈 시퀀스다.
        capped_document_ids: 이웃이 상한으로 꽉 찬 문서 id
            (`ReclusterInput.capped_document_ids`). 가를 후보의 `capped`를 정한다.

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
            capped=any(
                document_id in capped_document_ids
                for documents in groups.values()
                for document_id in documents
            ),
        )
        for event_id, groups in sorted(split_groups.items())
        if len(groups) > 1  # 한 커뮤니티에 다 있으면 가를 이유가 없다
    )

    return ReclusterCandidates(merges=merges, splits=splits)
