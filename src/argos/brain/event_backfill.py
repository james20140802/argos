"""기존 문서를 사건에 소급 배정한다 — ARG-271 / ARG-272 / ARG-274.

**순회 엔진 하나를 미리보기와 실행이 공유한다.** 두 벌로 쓰면 미리보기가
보고한 사건 수와 실제 실행 결과가 어긋나고, 그러면 "이 설정이면 몇 개
사건이 되는지"를 미리 보는 일 자체가 의미를 잃는다.

**미리보기가 DB를 한 글자도 건드리지 않으면서 실행과 같은 판정을 내는 방법:**
판정 함수(``decide_event``)가 이미 DB를 모르는 순수 함수라서(ARG-270),
후보 목록만 만들어 주면 된다. 실행 모드는 링크를 flush해 다음 문서의 DB
조회가 그것을 보게 하고, 미리보기 모드는 같은 정보를 **인메모리 오버레이**로
쌓는다 — 첫 문서가 만든 (아직 존재하지 않는) 사건이 둘째 문서의 후보로
보여야 두 문서가 같은 사건에 묶인다. 오버레이가 없으면 미리보기는 항상
"문서 수 = 사건 수"라는 거짓 결과를 낸다.

**근접중복(SimHash) 접기는 하지 않는다.** 온라인 배정 경로에도 접기 단계가
없기 때문이다 — SimHash는 근거 "수"를 셀 때만 접는다(``event_evidence``).
여기서 접으면 소급 판정이 온라인과 달라진다.

**순회 순서는 발행 시각 오름차순**이다. 온라인 경로가 시간순으로 들어오는
기사를 하나씩 배정하는 것을 그대로 재생한다. 동시각은 ``id`` 오름차순으로
깨서 같은 입력이 항상 같은 결과를 내게 한다.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_store import names_for_documents
from argos.brain.event_assignment import db_candidate_source, decide_event
from argos.brain.event_candidates import CandidateNeighbor, as_vector, keywords_of
from argos.brain.event_scoring import DocumentFeatures

if TYPE_CHECKING:
    from argos.config import EventDetectionConfig

logger = logging.getLogger(__name__)

_UNASSIGNED_SQL = text(
    """
    SELECT id, embedding, summary, digest, title,
           COALESCE(published_at, created_at) AS occurred_at
    FROM tech_items
    WHERE embedding IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM event_documents ed
          WHERE ed.tech_item_id = tech_items.id
      )
    ORDER BY COALESCE(published_at, created_at) ASC, id ASC
    """
)


@dataclass(frozen=True)
class BackfillDoc:
    """소급 배정 대상 문서 한 건 — 판정에 필요한 피처와 명명 근거."""

    tech_item_id: uuid.UUID
    features: DocumentFeatures
    title: str | None
    summary: str | None


@dataclass(frozen=True)
class Assignment:
    """문서 한 건의 배정 결과."""

    doc: BackfillDoc
    event_id: uuid.UUID
    created: bool


@dataclass(frozen=True)
class BackfillPlan:
    """순회 결과 — 미리보기 리포트와 실행 모드가 함께 쓰는 자료구조."""

    assignments: list[Assignment] = field(default_factory=list)

    @property
    def new_event_count(self) -> int:
        return sum(1 for assignment in self.assignments if assignment.created)

    @property
    def size_distribution(self) -> dict[int, int]:
        """{사건 크기: 그 크기인 사건 수}. 크기 내림차순으로 정렬해 돌려준다."""
        per_event: dict[uuid.UUID, int] = {}
        for assignment in self.assignments:
            per_event[assignment.event_id] = per_event.get(assignment.event_id, 0) + 1
        histogram: dict[int, int] = {}
        for size in per_event.values():
            histogram[size] = histogram.get(size, 0) + 1
        return dict(sorted(histogram.items(), key=lambda kv: -kv[0]))

    def samples(self, count: int = 3) -> list[tuple[int, list[str]]]:
        """가장 큰 사건부터 ``count``개, (크기, 문서 제목 목록)."""
        grouped: dict[uuid.UUID, list[str]] = {}
        for assignment in self.assignments:
            grouped.setdefault(assignment.event_id, []).append(
                assignment.doc.title or "(제목 없음)"
            )
        ordered = sorted(grouped.values(), key=lambda titles: (-len(titles), titles[0]))
        return [(len(titles), titles) for titles in ordered[:count]]


async def fetch_unassigned_documents(
    session: AsyncSession, *, limit: int | None = None
) -> list[BackfillDoc]:
    """임베딩이 있고 사건 링크가 없는 문서를 발행 시각 오름차순으로 읽는다.

    "링크가 없다"는 조건 자체가 진행 표식이다 — 별도 체크포인트 테이블 없이,
    이미 배정된 문서는 다음 실행의 대상 집합에서 자동으로 빠진다. 그래서
    중간에 죽어도 재실행이 남은 문서만 집는다.
    """
    statement = _UNASSIGNED_SQL
    params: dict[str, object] = {}
    if limit is not None:
        statement = text(str(_UNASSIGNED_SQL) + "\n    LIMIT :limit")
        params["limit"] = limit

    rows = (await session.execute(statement, params)).all()
    if not rows:
        return []

    ids = [row.id for row in rows]
    names_by_item = await names_for_documents(session, ids)

    docs: list[BackfillDoc] = []
    for row in rows:
        keyword_source = row.summary or row.digest
        docs.append(
            BackfillDoc(
                tech_item_id=row.id,
                features=DocumentFeatures(
                    embedding=as_vector(row.embedding),
                    names=names_by_item.get(row.id, frozenset()),
                    at=row.occurred_at,
                    keywords=keywords_of(keyword_source),
                ),
                title=row.title,
                summary=row.summary,
            )
        )
    return docs


class _PendingOverlay:
    """아직 커밋되지 않은 배정을 후보로 되돌려 주는 인메모리 계층.

    미리보기 전용이다. 실행 모드는 링크를 flush해 DB 조회가 직접 보게 하므로
    이 계층이 필요 없다 — 그래서 두 모드의 판정이 같은 값으로 수렴한다.
    """

    def __init__(self) -> None:
        self._by_event: dict[uuid.UUID, list[BackfillDoc]] = {}

    def add(self, doc: BackfillDoc, event_id: uuid.UUID) -> None:
        self._by_event.setdefault(event_id, []).append(doc)

    def candidates(self, at: datetime, *, window_days: float) -> list[CandidateNeighbor]:
        """*at* 기준 시간 창 안의 pending 이웃들.

        창 계산은 ``fetch_candidates``와 같다 — *at* 앞뒤로 각각
        ``window_days``. 다르게 두면 미리보기가 DB 경로보다 넓거나 좁게 봐서
        판정이 어긋난다.
        """
        window = timedelta(days=window_days)
        neighbours: list[CandidateNeighbor] = []
        for event_id, docs in self._by_event.items():
            for doc in docs:
                doc_at = doc.features.at
                if doc_at is None or abs(doc_at - at) > window:
                    continue
                neighbours.append(
                    CandidateNeighbor(
                        tech_item_id=doc.tech_item_id,
                        features=doc.features,
                        event_ids=(event_id,),
                    )
                )
        # 결정성: 같은 입력이 항상 같은 순서를 내도록 id로 정렬한다.
        return sorted(neighbours, key=lambda neighbour: str(neighbour.tech_item_id))


async def plan_backfill(
    session: AsyncSession,
    docs: Sequence[BackfillDoc],
    *,
    config: "EventDetectionConfig",
) -> BackfillPlan:
    """DB에 아무것도 쓰지 않고 배정 결과만 계산한다 (미리보기).

    가상 사건 id는 ``uuid.uuid4()``로 만든다 — 리포트 안에서 사건을 구분하는
    용도일 뿐, 실행 모드가 만들 실제 id와는 무관하다.
    """
    overlay = _PendingOverlay()
    plan = BackfillPlan()
    for doc in docs:
        at = doc.features.at
        db_neighbours: list[CandidateNeighbor] = []
        if doc.features.embedding and at is not None:
            db_neighbours = await db_candidate_source(
                session, embedding=doc.features.embedding, at=at
            )
        pending = overlay.candidates(at, window_days=config.window_days) if at else []
        event_id = decide_event(doc.features, [*db_neighbours, *pending], config=config)
        created = event_id is None
        if event_id is None:
            event_id = uuid.uuid4()
        overlay.add(doc, event_id)
        plan.assignments.append(Assignment(doc=doc, event_id=event_id, created=created))
    return plan
