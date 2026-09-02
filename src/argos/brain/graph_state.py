from __future__ import annotations

import uuid
from datetime import datetime
from typing import NotRequired, TypedDict

from argos.brain.entity_extraction import ExtractedName
from argos.models.tech_item import CategoryType


class BrainState(TypedDict):
    raw_text: str
    source_url: str
    is_valid: bool
    trust_score: float | None
    summary: str | None
    extracted_info: dict | None
    related_tech_ids: list[str]
    succession_result: dict | None
    saved: bool
    genealogy_skipped: bool
    genealogy_skip_reason: str | None
    # Hint from the fetcher (RSS, arXiv, etc.) indicating which category the
    # source leans towards. GitHub/HN fetchers leave this as None.
    source_category: CategoryType | None
    # Decided by triage_node via LLM; falls back to ALPHA if LLM omits the
    # field or returns an unrecognised value.
    category: CategoryType | None
    # Populated by save_node when a new TechItem row is inserted (ARG-103).
    # Downstream consumers (succession alert post-processing) use this to
    # collect the freshly-saved item IDs and call check_succession.
    # None when the item already existed (duplicate URL) or save was skipped.
    saved_item_id: NotRequired[uuid.UUID | None]
    # Publication date extracted by the fetcher (HN Unix epoch, RSS published_parsed,
    # arXiv published_parsed, GitHub API created_at, OpenGraph article:published_time).
    # None when the source did not provide a date or extraction failed.
    published_at: NotRequired[datetime | None]
    # og:image URL extracted by the fetcher (ARG-135). None when the source
    # had no og:image / twitter:image meta or the value failed validation.
    image_url: NotRequired[str | None]
    # Longform digest produced by digest_node (ARG-173). NotRequired so existing
    # state initializers need not set it; save_node reads via state.get("digest").
    # None when the node skipped (thin content), failed, or output was rejected.
    digest: NotRequired[str | None]
    # Set by _triage_one ONLY when the Ollama call fails for infrastructure
    # reasons (OllamaInfraError: connection/timeout/OOM). None/absent on success
    # and on genuine is_valid=False rejections. Consumers (pipeline Stage 6,
    # CLI _run) key retention/exit-code off this, never off is_valid. (ARG-190)
    triage_error: NotRequired[str | None]
    # ARG-206: 5-field evidence rubric extracted by triage_node (temperature 0).
    # Feeds argos.brain.trust.score_rubric() for the deterministic trust
    # synthesis; None on parse failure / infra error / relevance-gate demotion.
    trust_rubric: NotRequired[dict | None]
    # ARG-263: 이 문서에서 뽑은 고유명사. 배정 단계(ARG-266)가 새 문서 쪽
    # 이름 항 입력으로 쓰고, save_node가 document_entities 링크로 옮긴다.
    # 정규형 목록은 entity_names, 원문까지 실은 추출 결과는 별도로 둔다 —
    # 링크를 쓸 때 표시용 원문(Entity.name)이 필요하기 때문이다.
    entity_names: NotRequired[list[str] | None]
    entity_names_extracted: NotRequired[list[ExtractedName] | None]
    # ARG-266: assign_event_node이 채운다. 임계값을 넘는 기존 사건이 있으면
    # 그 id, 없으면 None(=새 사건이 필요하다는 뜻 — event_assigned=True일
    # 때만). 배정 노드는 사건 row를 만들지 않는다 — 새 사건 생성은
    # save_node가 문서 저장과 같은 자리에서 한다(배정 뒤 저장이 실패하면
    # 문서 없는 빈 사건이 남는 것을 막기 위해). 배정 자체가 실패해도 None으로
    # 두고 예외를 밖으로 내보내지 않는다.
    event_id: NotRequired[uuid.UUID | None]
    # ARG-266: event_id=None의 두 뜻(판정 끝냈지만 못 찾음 / 판정 자체가
    # 실패함)을 가른다. True면 배정이 끝까지 돌았다는 뜻 — save_node가
    # event_id가 None이어도 새 사건을 만들어도 된다. False(또는 이 키가
    # 아예 없음)면 배정이 시도조차 안 됐거나 도중에 실패한 것 — save_node는
    # 이때 사건도 링크도 만들지 않는다. 그래야 배정 실패가 잘못된 새 사건을
    # 영구히 남기지 않고, "링크 없음"이 그대로 나중 백필의 대상 표시로
    # 남는다(부모 AC: 배정에 "성공한" 문서만 무소속 없음을 보장한다).
    event_assigned: NotRequired[bool]
    # ARG-267: _attach_extracted_names(pipeline.py)가 이름 추출과 같은 자리에서
    # 계산해 싣는다 — raw_text를 이미 손에 든 자리라 두 번째 패스를 만들지
    # 않는다. save_node가 to_storage를 거쳐 tech_items.simhash에 저장하고,
    # 그 저장값을 event_evidence.evidence_count가 근접중복 접기에 쓴다.
    # 유효하지 않은 state는 채워지지 않는다(entity_names와 같은 계약).
    simhash: NotRequired[int | None]
    # ARG-273: save_node가 채운다. 이 저장이 새로 만든 사건의 id — 문서가
    # 기존 사건에 붙었거나 사건이 아예 관여하지 않았으면 None. 다른 값과
    # 달리 assign_event_node가 아니라 save_node가 채운다: 새 사건 row 자체를
    # save_node가 문서 저장과 같은 자리에서 만들기 때문이다(event_id 주석
    # 참고). ``_assign_then_save``가 이 값을 읽어 방금 생긴 사건에만 명명
    # 훅(``apply_event_naming``)을 붙일지 정한다 — 기존 사건에 합류한
    # 경우는 이미 naming_stale이 서 있으므로 여기서 LLM을 또 부르지 않는다.
    created_event_id: NotRequired[uuid.UUID | None]
