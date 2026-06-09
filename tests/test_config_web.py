"""Tests for the [web] UserConfig section (ARG-133)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from argos.config import UserConfig, WebConfig


def test_web_config_defaults():
    """Default WebConfig binds to localhost:8765."""
    cfg = WebConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8765


def test_user_config_includes_web_section_by_default():
    """UserConfig() exposes a default WebConfig under .web."""
    cfg = UserConfig()
    assert isinstance(cfg.web, WebConfig)
    assert cfg.web.host == "127.0.0.1"
    assert cfg.web.port == 8765


def test_web_config_accepts_custom_host_and_port():
    """Operator can override host/port via TOML."""
    cfg = UserConfig.model_validate(
        {"web": {"host": "0.0.0.0", "port": 9000}}
    )
    assert cfg.web.host == "0.0.0.0"
    assert cfg.web.port == 9000


def test_web_config_rejects_invalid_port():
    """Port must be a positive int in TCP range."""
    with pytest.raises(ValidationError):
        UserConfig.model_validate({"web": {"port": 0}})
    with pytest.raises(ValidationError):
        UserConfig.model_validate({"web": {"port": 70000}})
