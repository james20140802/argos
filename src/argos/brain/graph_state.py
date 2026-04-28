from __future__ import annotations
from typing import TypedDict


class BrainState(TypedDict):
    raw_text: str
    source_url: str
    is_valid: bool
    extracted_info: dict | None
    related_tech_ids: list[str]
    succession_result: dict | None
