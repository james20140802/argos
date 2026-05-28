from __future__ import annotations

from datetime import datetime, timezone
from typing import get_type_hints

from argos.brain.graph_state import BrainState


def test_brain_state_has_published_at_field():
    """BrainState TypedDict must include published_at as a NotRequired field."""
    hints = get_type_hints(BrainState, include_extras=True)
    assert "published_at" in hints


def test_brain_state_accepts_published_at():
    """BrainState can be constructed with a published_at datetime value."""
    state: BrainState = {
        "raw_text": "test content",
        "source_url": "https://example.com",
        "is_valid": False,
        "trust_score": None,
        "summary": None,
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
        "published_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
    }
    assert state["published_at"] == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_brain_state_accepts_none_published_at():
    """published_at=None is valid (NotRequired field)."""
    state: BrainState = {
        "raw_text": "test",
        "source_url": "https://example.com",
        "is_valid": False,
        "trust_score": None,
        "summary": None,
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
        "published_at": None,
    }
    assert state.get("published_at") is None
