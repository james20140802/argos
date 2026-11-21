from __future__ import annotations

from unittest.mock import MagicMock

from argos.config import UserConfig


def test_defaults_follow_event_summary():
    resolved = UserConfig().deep_analysis_resolved()
    assert resolved.model == "gemma4:26b"
    assert resolved.num_ctx == 16384
    assert resolved.timeout == 600
    assert resolved.output_reserve_tok == 2048


def test_follows_customized_event_summary():
    cfg = UserConfig(event_summary={"model": "x:1", "num_ctx": 8192})
    resolved = cfg.deep_analysis_resolved()
    assert resolved.model == "x:1"
    assert resolved.num_ctx == 8192


def test_explicit_values_override():
    cfg = UserConfig(deep_analysis={"model": "y:2", "timeout": 900})
    resolved = cfg.deep_analysis_resolved()
    assert resolved.model == "y:2"
    assert resolved.timeout == 900
    assert resolved.num_ctx == cfg.event_summary.num_ctx
    assert resolved.output_reserve_tok == cfg.event_summary.output_reserve_tok


def test_existing_config_file_without_section_loads(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[event_summary]\nmodel = "z:3"\n', encoding="utf-8")
    cfg = UserConfig.load_strict(path=path)
    assert cfg.deep_analysis.model is None
    assert cfg.deep_analysis_resolved().model == "z:3"


def test_get_deep_analysis_llm_client_resolves_model():
    from argos.brain import llm_client

    settings = MagicMock()
    settings.user = UserConfig(deep_analysis={"model": "deep:9"})
    settings.user.llm.backend = "ollama"
    client = llm_client.OllamaClient(settings=settings, large_model="deep:9")
    assert client._resolve("large") == "deep:9"

    original = llm_client._get_settings
    llm_client._get_settings = lambda: settings
    try:
        built = llm_client.get_deep_analysis_llm_client()
    finally:
        llm_client._get_settings = original
    assert built._resolve("large") == "deep:9"
