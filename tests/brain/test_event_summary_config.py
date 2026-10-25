from __future__ import annotations

import pytest
from pydantic import ValidationError

from argos.config import EventSummaryConfig, UserConfig


def test_defaults():
    cfg = EventSummaryConfig()
    assert cfg.model == "gemma4:26b"
    assert cfg.num_ctx == 16384
    assert cfg.doc_max_chars == 6000
    assert cfg.max_docs == 30
    assert cfg.chars_per_tok == 2.5
    assert cfg.output_reserve_tok == 2048
    assert cfg.timeout == 600
    assert cfg.temperature == 0.3


def test_user_config_has_event_summary():
    assert isinstance(UserConfig().event_summary, EventSummaryConfig)


def test_model_is_overridable():
    cfg = UserConfig(event_summary={"model": "qwen3:14b", "num_ctx": 8192})
    assert cfg.event_summary.model == "qwen3:14b"
    assert cfg.event_summary.num_ctx == 8192


def test_num_ctx_has_a_floor():
    with pytest.raises(ValidationError):
        EventSummaryConfig(num_ctx=1024)
