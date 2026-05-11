from __future__ import annotations
from typing import TypedDict


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
