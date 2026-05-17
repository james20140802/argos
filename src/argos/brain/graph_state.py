from __future__ import annotations

import uuid
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
