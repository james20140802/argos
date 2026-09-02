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
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from argos.brain.entity_store import names_for_documents
from argos.brain.event_assignment import (
    LinkResult,
    db_candidate_source,
    decide_event,
    link_document_to_event,
)
from argos.brain.event_candidates import CandidateNeighbor, as_vector, keywords_of
from argos.brain.event_naming import EvidenceDoc, apply_event_naming
from argos.brain.event_scoring import DocumentFeatures, cosine_similarity
from argos.brain.llm_client import OllamaClient

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


def _cap_candidates(
    subject: DocumentFeatures,
    candidates: list[CandidateNeighbor],
    *,
    k: int,
) -> list[CandidateNeighbor]:
    """합친 후보 목록을 ``k``개로 자른다 — DB 경로가 미리 자르는 것과 같은 자리.

    ``fetch_candidates``는 ``LIMIT :limit``으로 상위 K만 반환한 뒤 판정에
    넘긴다. 미리보기가 DB 후보와 pending 오버레이 후보를 합친 목록을 그대로
    넘기면, 밀집한 시간 창에서 실제 실행보다 **더 많은** 이웃을 보게 되어
    실행이라면 갈랐을 두 문서를 미리보기가 묶어 버릴 수 있다. 그래서 합친
    뒤에도 DB 쿼리와 같은 규칙(코사인 상위 K, 동점은 id 오름차순)으로 다시
    한 번 잘라야 두 경로의 판정이 수렴한다.
    """
    if len(candidates) <= k:
        return candidates
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -cosine_similarity(subject.embedding, candidate.features.embedding),
            str(candidate.tech_item_id),
        ),
    )
    return ranked[:k]


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
                session,
                embedding=doc.features.embedding,
                at=at,
                config=config,
                exclude_id=doc.tech_item_id,
            )
        pending = overlay.candidates(at, window_days=config.window_days) if at else []
        candidates = _cap_candidates(
            doc.features, [*db_neighbours, *pending], k=config.candidate_k
        )
        event_id = decide_event(doc.features, candidates, config=config)
        created = event_id is None
        if event_id is None:
            event_id = uuid.uuid4()
        overlay.add(doc, event_id)
        plan.assignments.append(Assignment(doc=doc, event_id=event_id, created=created))
    return plan


@dataclass(frozen=True)
class ExecuteResult:
    """실행 모드의 종료 요약."""

    assigned: int
    created_events: int
    skipped: int


async def execute_backfill(
    session: AsyncSession,
    docs: Sequence[BackfillDoc],
    *,
    config: "EventDetectionConfig",
    batch_size: int = 50,
    on_progress: "Callable[[int, int], None] | None" = None,
) -> ExecuteResult:
    """문서를 실제로 사건에 배정한다. ``batch_size``마다 커밋한다.

    **왜 오버레이가 없는가:** 문서마다 링크를 flush하므로, 다음 문서의
    ``db_candidate_source`` 조회가 같은 트랜잭션 안에서 그 링크를 본다. 즉
    미리보기의 인메모리 오버레이가 하던 일을 여기서는 DB가 한다 — 그래서
    두 모드가 같은 판정에 수렴한다.

    **왜 배치마다 커밋하는가:** 전체를 한 트랜잭션으로 묶으면 556건째에서
    실패했을 때 앞선 555건이 통째로 사라진다. 배치 하나가 커밋되면 그 배치의
    배정은 확정이고, 재실행은 "링크가 없는 문서"만 다시 집으므로 이어서
    진행된다.

    **문서 한 건의 실패는 그 건만 건너뛴다.** 세이브포인트 안에서 쓰고 실패를
    삼키므로, 실패가 중단시킨 트랜잭션이 뒤따르는 문서까지 오염시키지 않는다.
    """
    assigned = 0
    created_events = 0
    skipped = 0
    pending_in_batch = 0

    for index, doc in enumerate(docs, start=1):
        at = doc.features.at
        try:
            candidates: list[CandidateNeighbor] = []
            if doc.features.embedding and at is not None:
                candidates = await db_candidate_source(
                    session,
                    embedding=doc.features.embedding,
                    at=at,
                    config=config,
                    exclude_id=doc.tech_item_id,
                )
            event_id = decide_event(doc.features, candidates, config=config)
            async with session.begin_nested():
                link: LinkResult = await link_document_to_event(
                    session,
                    tech_item_id=doc.tech_item_id,
                    event_id=event_id,
                    occurred_at=at or datetime.now(timezone.utc),
                )
                # 다음 문서의 후보 조회가 이 링크를 보게 한다.
                await session.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "backfill: assigning %s failed: %r — skipping", doc.tech_item_id, exc
            )
            skipped += 1
            continue

        assigned += 1
        if link.created:
            created_events += 1
        pending_in_batch += 1
        if on_progress is not None:
            on_progress(index, len(docs))

        if pending_in_batch >= batch_size:
            await session.commit()
            pending_in_batch = 0

    if pending_in_batch:
        await session.commit()

    return ExecuteResult(
        assigned=assigned, created_events=created_events, skipped=skipped
    )


_STALE_EVENTS_SQL = text(
    """
    SELECT e.id AS event_id, e.updated_at AS event_updated_at,
           i.title AS doc_title, i.summary AS doc_summary
    FROM tech_events e
    JOIN event_documents ed ON ed.event_id = e.id
    JOIN tech_items i ON i.id = ed.tech_item_id
    WHERE e.merged_into_id IS NULL
      AND (e.naming_stale IS TRUE OR e.title IS NULL)
      AND e.id IN (
          SELECT id FROM tech_events
          WHERE merged_into_id IS NULL
            AND (naming_stale IS TRUE OR title IS NULL)
          ORDER BY occurred_at ASC, id ASC
          LIMIT :limit
      )
    ORDER BY e.occurred_at ASC, e.id ASC, i.id ASC
    """
)
"""재명명 대상과 그 근거 문서를 한 번에 읽는다.

툼스톤(``merged_into_id``가 채워진 사건)은 제외한다 — 흡수돼 더 이상
표시되지 않는 사건에 LLM을 태울 이유가 없다. 대상 조건이 ``naming_stale``
**또는** ``title IS NULL``인 이유는, 플래그를 세우는 코드(ARG-270)가 생기기
전에 만들어진 무명 사건까지 이 경로로 줍기 위해서다.

``LIMIT``이 서브쿼리에 걸린 것은 의도적이다 — 바깥에 걸면 조인으로 늘어난
행 수를 자르게 되어 사건 하나의 근거가 잘려 나간다.

``updated_at``을 같이 읽는 것은 ARG-274의 낙관적 가드용이다 — 근거 스냅샷을
뜬 시점의 행 버전을 들고 있어야, LLM이 도는 동안 사건이 바뀌었는지
``apply_event_naming``이 판정할 수 있다.
"""

_NO_LIMIT = 2_000_000_000
"""``--limit`` 미지정 시 쓰는 사실상 무한대. SQL을 두 벌로 나누지 않으려는 것."""


@dataclass(frozen=True)
class StaleEvent:
    """재명명 대상 사건 하나와 그 근거 문서들.

    ``updated_at``은 근거를 읽은 시점의 행 버전이다 — ``apply_event_naming``이
    "그 사이 사건이 안 바뀌었을 때만 쓴다"를 판정하는 데 쓴다(ARG-274).
    스냅샷 없이 만든 ``StaleEvent``는 ``None``이라 가드가 걸리지 않는다.
    """

    event_id: uuid.UUID
    docs: list[EvidenceDoc]
    updated_at: datetime | None = None


@dataclass(frozen=True)
class RenameResult:
    """재명명 종료 요약."""

    renamed: int
    skipped: int


async def fetch_stale_events(
    session: AsyncSession, *, limit: int | None = None
) -> list[StaleEvent]:
    """``naming_stale``이 섰거나 아직 무명인 (툼스톤 아닌) 사건들을 읽는다."""
    rows = (
        await session.execute(
            _STALE_EVENTS_SQL, {"limit": limit if limit is not None else _NO_LIMIT}
        )
    ).all()

    grouped: dict[uuid.UUID, list[EvidenceDoc]] = {}
    versions: dict[uuid.UUID, datetime | None] = {}
    order: list[uuid.UUID] = []
    for row in rows:
        if row.event_id not in grouped:
            grouped[row.event_id] = []
            versions[row.event_id] = row.event_updated_at
            order.append(row.event_id)
        grouped[row.event_id].append(
            EvidenceDoc(title=row.doc_title, summary=row.doc_summary)
        )
    return [
        StaleEvent(
            event_id=event_id,
            docs=grouped[event_id],
            updated_at=versions[event_id],
        )
        for event_id in order
    ]


async def rename_stale_events(
    session: AsyncSession,
    events: Sequence[StaleEvent],
    *,
    batch_size: int = 50,
    client: "OllamaClient | None" = None,
) -> RenameResult:
    """대상 사건들의 이름·요약을 다시 짓는다. ``batch_size``마다 커밋한다.

    사건 구성 자체는 건드리지 않는다 — 쪼개기/합치기는 이 단계의 비목표다.
    실패한 사건은 ``naming_stale``이 선 채로 남아 다음 실행이 다시 집는다.

    ARG-274: 근거를 읽은 뒤 LLM이 도는 동안 그 사건이 바뀌었으면(온라인
    파이프라인이 문서를 하나 더 매달았으면) ``apply_event_naming``이 아무것도
    쓰지 않고 ``False``를 돌린다. 그 사건은 여기서 ``skipped``로 세고, 새 근거를
    포함한 이름은 다음 실행이 짓는다 — 위 실패 처리와 같은 규약이다. 스냅샷은
    ``fetch_stale_events``가 한 번만 뜨므로, 배치가 길수록 이 가드가 실제로
    필요해진다.
    """
    renamed = 0
    skipped = 0
    pending = 0

    for event in events:
        try:
            async with session.begin_nested():
                applied = await apply_event_naming(
                    session,
                    event.event_id,
                    event.docs,
                    client=client,
                    expected_updated_at=event.updated_at,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("backfill: renaming %s failed: %r", event.event_id, exc)
            applied = False

        if applied:
            renamed += 1
            pending += 1
        else:
            skipped += 1

        if pending >= batch_size:
            await session.commit()
            pending = 0

    if pending:
        await session.commit()

    return RenameResult(renamed=renamed, skipped=skipped)
