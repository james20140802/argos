from __future__ import annotations

from argos.config import DigestConfig, UserConfig


def test_digest_config_defaults():
    cfg = DigestConfig()
    assert cfg.model == "qwen3:14b"
    assert cfg.num_ctx == 4096
    assert cfg.input_max_chars == 6000
    assert cfg.min_content_chars == 1000
    assert cfg.min_output_chars == 150


def test_user_config_has_digest():
    assert isinstance(UserConfig().digest, DigestConfig)
