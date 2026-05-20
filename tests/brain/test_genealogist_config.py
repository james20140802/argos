"""Tests for ARG-126: genealogist.model config field and prewarm/unload propagation.

Covers:
- GenealogistConfig defaults (model + num_ctx)
- GenealogistConfig overrides
- OllamaClient reads genealogist.model for the large role
- OllamaClient.prewarm propagates num_ctx from genealogist config
- genealogist_node uses the configured model
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import respx
import httpx

from argos.brain.ollama_client import OLLAMA_BASE_URL


# ---------------------------------------------------------------------------
# Config: GenealogistConfig.model field
# ---------------------------------------------------------------------------


def test_genealogist_config_has_model_field_with_default():
    """GenealogistConfig must expose a `model` field defaulting to 'qwen3:32b'."""
    from argos.config import GenealogistConfig

    cfg = GenealogistConfig()
    assert hasattr(cfg, "model")
    assert cfg.model == "qwen3:32b"


def test_genealogist_config_model_can_be_overridden():
    """GenealogistConfig.model must accept the quantized variant."""
    from argos.config import GenealogistConfig

    cfg = GenealogistConfig(model="qwen3:32b-q4_K_M")
    assert cfg.model == "qwen3:32b-q4_K_M"


def test_genealogist_config_num_ctx_default_is_3072():
    """GenealogistConfig.num_ctx default must remain 3072 to preserve existing behavior."""
    from argos.config import GenealogistConfig

    cfg = GenealogistConfig()
    assert cfg.num_ctx == 3072


def test_genealogist_config_num_ctx_can_be_set_to_6144():
    """GenealogistConfig.num_ctx must accept 6144 for the quantized model use case."""
    from argos.config import GenealogistConfig

    cfg = GenealogistConfig(num_ctx=6144)
    assert cfg.num_ctx == 6144


# ---------------------------------------------------------------------------
# OllamaClient: large role resolves to genealogist.model, not ollama.model_deepdive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_client_large_role_uses_genealogist_model(monkeypatch):
    """OllamaClient._resolve('large') must return genealogist.model, not ollama.model_deepdive."""
    from argos.brain.llm_client import OllamaClient

    # Patch genealogist.model to the quantized variant
    mock_settings = MagicMock()
    mock_settings.user.genealogist.model = "qwen3:32b-q4_K_M"
    mock_settings.user.genealogist.num_ctx = 6144
    mock_settings.user.ollama.model_triage = "qwen3:8b"
    mock_settings.user.ollama.model_deepdive = "qwen3:32b"  # old value, should not be used

    client = OllamaClient(settings=mock_settings)
    resolved = client._resolve("large")
    assert resolved == "qwen3:32b-q4_K_M"


@pytest.mark.asyncio
async def test_ollama_client_small_role_still_uses_triage_model(monkeypatch):
    """OllamaClient._resolve('small') must still come from ollama.model_triage."""
    from argos.brain.llm_client import OllamaClient

    mock_settings = MagicMock()
    mock_settings.user.genealogist.model = "qwen3:32b-q4_K_M"
    mock_settings.user.genealogist.num_ctx = 3072
    mock_settings.user.ollama.model_triage = "qwen3:8b"
    mock_settings.user.ollama.model_deepdive = "qwen3:32b"

    client = OllamaClient(settings=mock_settings)
    resolved = client._resolve("small")
    assert resolved == "qwen3:8b"


# ---------------------------------------------------------------------------
# OllamaClient.prewarm propagates genealogist num_ctx
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_client_prewarm_propagates_genealogist_num_ctx(monkeypatch):
    """OllamaClient.prewarm('large') must pass genealogist.num_ctx to prewarm_model.

    This fixes the prewarm waste: if prewarm uses DEFAULT_NUM_CTX=4096 but
    the real call uses genealogist.num_ctx=3072, Ollama reloads the model.
    """
    from argos.brain.llm_client import OllamaClient
    import argos.brain.llm_client as llm_module

    mock_settings = MagicMock()
    mock_settings.user.genealogist.model = "qwen3:32b"
    mock_settings.user.genealogist.num_ctx = 3072
    mock_settings.user.ollama.model_triage = "qwen3:8b"

    captured: dict = {}

    async def _fake_prewarm(model, keep_alive="5m", num_ctx=4096):
        captured["model"] = model
        captured["num_ctx"] = num_ctx
        captured["keep_alive"] = keep_alive

    monkeypatch.setattr(llm_module, "prewarm_model", _fake_prewarm)

    client = OllamaClient(settings=mock_settings)
    await client.prewarm("large")

    assert captured["model"] == "qwen3:32b"
    assert captured["num_ctx"] == 3072  # must match genealogist.num_ctx, not DEFAULT_NUM_CTX


@pytest.mark.asyncio
async def test_ollama_client_prewarm_uses_configured_num_ctx_6144(monkeypatch):
    """When genealogist.num_ctx=6144, prewarm must allocate the 6144 context."""
    from argos.brain.llm_client import OllamaClient
    import argos.brain.llm_client as llm_module

    mock_settings = MagicMock()
    mock_settings.user.genealogist.model = "qwen3:32b-q4_K_M"
    mock_settings.user.genealogist.num_ctx = 6144
    mock_settings.user.ollama.model_triage = "qwen3:8b"

    captured: dict = {}

    async def _fake_prewarm(model, keep_alive="5m", num_ctx=4096):
        captured["model"] = model
        captured["num_ctx"] = num_ctx

    monkeypatch.setattr(llm_module, "prewarm_model", _fake_prewarm)

    client = OllamaClient(settings=mock_settings)
    await client.prewarm("large")

    assert captured["model"] == "qwen3:32b-q4_K_M"
    assert captured["num_ctx"] == 6144


# ---------------------------------------------------------------------------
# genealogist_node: uses the model from genealogist.model config (via LLMClient.query)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_genealogist_node_query_uses_configured_model(monkeypatch):
    """genealogist_node must call client.query('large', ...) — model resolution
    happens inside OllamaClient, not in the node itself. Verify that the 'large'
    role is passed (not a hard-coded model string)."""
    from argos.brain.nodes import genealogist as gen_module

    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured["model_role"] = model_role
            return '{"replace_target_id": null, "relation_type": null, "reason": "no relation"}'

    monkeypatch.setattr(gen_module, "get_llm_client", lambda: _FakeClient())

    from argos.brain.graph_state import BrainState

    state: BrainState = {
        "raw_text": "New AI inference runtime",
        "source_url": "https://example.com",
        "is_valid": True,
        "trust_score": 0.8,
        "summary": None,
        "extracted_info": {
            "similar_items": [
                {"id": "abc-123", "title": "Old Tech", "raw_content": "legacy content"}
            ],
            "embedding": [0.1] * 10,
        },
        "related_tech_ids": ["abc-123"],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
    }

    result = await gen_module.genealogist_node(state)

    assert captured["model_role"] == "large"
    assert result["succession_result"] is not None


@pytest.mark.asyncio
async def test_genealogist_node_passes_configured_num_ctx(monkeypatch):
    """genealogist_node must pass settings.user.genealogist.num_ctx to client.query."""
    from argos.brain.nodes import genealogist as gen_module
    import argos.config as config_module

    # Patch genealogist.num_ctx on the live settings singleton
    monkeypatch.setattr(config_module.settings.user.genealogist, "num_ctx", 6144)

    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured.update(kwargs)
            return '{"replace_target_id": null, "relation_type": null, "reason": "ok"}'

    monkeypatch.setattr(gen_module, "get_llm_client", lambda: _FakeClient())

    from argos.brain.graph_state import BrainState

    state: BrainState = {
        "raw_text": "New tech",
        "source_url": "https://example.com",
        "is_valid": True,
        "trust_score": 0.8,
        "summary": None,
        "extracted_info": {
            "similar_items": [
                {"id": "abc-123", "title": "Old Tech", "raw_content": "legacy"}
            ],
            "embedding": [0.1] * 10,
        },
        "related_tech_ids": ["abc-123"],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
    }

    await gen_module.genealogist_node(state)

    assert captured.get("num_ctx") == 6144


# ---------------------------------------------------------------------------
# End-to-end: OllamaClient query sends the configured model name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_client_query_large_sends_configured_model_name(monkeypatch):
    """OllamaClient.query('large', ...) must send genealogist.model to /api/generate."""
    from argos.brain.llm_client import OllamaClient

    mock_settings = MagicMock()
    mock_settings.user.genealogist.model = "qwen3:32b-q4_K_M"
    mock_settings.user.genealogist.num_ctx = 6144
    mock_settings.user.ollama.model_triage = "qwen3:8b"

    client = OllamaClient(settings=mock_settings)

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "answer"})
        )
        result = await client.query("large", "test prompt", num_ctx=6144)

    assert result == "answer"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:32b-q4_K_M"


@pytest.mark.asyncio
async def test_ollama_client_prewarm_http_sends_configured_model(monkeypatch):
    """OllamaClient.prewarm('large') must send genealogist.model name in HTTP request."""
    from argos.brain.llm_client import OllamaClient

    mock_settings = MagicMock()
    mock_settings.user.genealogist.model = "qwen3:32b-q4_K_M"
    mock_settings.user.genealogist.num_ctx = 6144
    mock_settings.user.ollama.model_triage = "qwen3:8b"

    client = OllamaClient(settings=mock_settings)

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": ""})
        )
        await client.prewarm("large")

    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:32b-q4_K_M"
    assert body["options"]["num_ctx"] == 6144
