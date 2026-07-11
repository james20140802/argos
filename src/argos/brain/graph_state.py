from __future__ import annotations

import uuid
from datetime import datetime
from typing import NotRequired, TypedDict

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
