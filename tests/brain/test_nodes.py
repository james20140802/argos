from __future__ import annotations
import pytest
import respx
import httpx
from unittest.mock import AsyncMock
from argos.brain.graph_state import BrainState
from argos.brain.nodes.triage import triage_node
from argos.brain.nodes.embed import embed_and_search_node
from argos.brain.nodes.save import save_node
from argos.brain.ollama_client import OLLAMA_BASE_URL

def _state(**kwargs) -> BrainState:
    base: BrainState = {
        "raw_text": "sample text",
        "source_url": "https://example.com",
        "is_valid": False,
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
    }
    return {**base, **kwargs}

@pytest.mark.asyncio
async def test_triage_node_valid():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": '{"is_valid": true, "reason": "real tool"}'})
        )
        result = await triage_node(_state())
    assert result["is_valid"] is True

@pytest.mark.asyncio
async def test_triage_node_invalid():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": '{"is_valid": false, "reason": "marketing"}'})
        )
        result = await triage_node(_state())
    assert result["is_valid"] is False

@pytest.mark.asyncio
async def test_triage_node_parse_error():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "not json at all"})
        )
        result = await triage_node(_state())
    assert result["is_valid"] is False

@pytest.mark.asyncio
async def test_embed_node_skips_if_invalid():
    session = AsyncMock()
    result = await embed_and_search_node(_state(is_valid=False), session=session)
    assert result["related_tech_ids"] == []
    session.execute.assert_not_called()

@pytest.mark.asyncio
async def test_save_node_skips_if_invalid():
    session = AsyncMock()
    result = await save_node(_state(is_valid=False), session=session)
    session.add.assert_not_called()
