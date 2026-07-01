from __future__ import annotations

import httpx
import pytest
import respx

from argos.brain.graph_state import BrainState
from argos.brain.nodes.digest import digest_node, generate_digest
from argos.brain.ollama_client import OLLAMA_BASE_URL


def _state(**kw) -> BrainState:
    base: BrainState = {
        "raw_text": "x" * 2000,
        "source_url": "https://example.com",
        "is_valid": True,
        "trust_score": 0.8,
        "summary": "짧은 한 줄 요약.",
        "extracted_info": None,
        "related_tech_ids": [],
        "succession_result": None,
        "saved": False,
        "genealogy_skipped": False,
        "genealogy_skip_reason": None,
        "source_category": None,
        "category": None,
    }
    return {**base, **kw}


def _mock_generate(text: str):
    return respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": text})
    )


@pytest.mark.asyncio
async def test_digest_node_generates_longform():
    longform = "문단1 " * 60  # >150 chars, summary와 다름
    with respx.mock:
        _mock_generate(longform)
        result = await digest_node(_state())
    assert result["digest"] is not None
    assert len(result["digest"]) >= 150


@pytest.mark.asyncio
async def test_digest_node_skips_thin_content():
    # raw_text < min_content_chars(1000) → LLM 호출 없이 None
    with respx.mock:
        route = _mock_generate("무시됨")
        result = await digest_node(_state(raw_text="너무 짧음"))
    assert result["digest"] is None
    assert route.called is False


@pytest.mark.asyncio
async def test_digest_node_rejects_short_output():
    with respx.mock:
        _mock_generate("짧아")  # < min_output_chars
        result = await digest_node(_state())
    assert result["digest"] is None


@pytest.mark.asyncio
async def test_digest_node_rejects_duplicate_of_summary():
    summary = "이것은 요약입니다. " * 12  # >150 chars
    with respx.mock:
        _mock_generate(summary)  # digest ≈ summary
        result = await digest_node(_state(summary=summary))
    assert result["digest"] is None


@pytest.mark.asyncio
async def test_digest_node_invalid_state_noop():
    with respx.mock:
        route = _mock_generate("무시됨")
        result = await digest_node(_state(is_valid=False))
    assert result.get("digest") is None
    assert route.called is False


@pytest.mark.asyncio
async def test_digest_node_llm_error_returns_none():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(500)
        )
        result = await digest_node(_state())
    assert result["digest"] is None


@pytest.mark.asyncio
async def test_generate_digest_strips_think_block():
    with respx.mock:
        _mock_generate("<think>계획</think>" + ("실제 다이제스트 본문. " * 20))
        out = await generate_digest("y" * 2000)
    assert out is not None
    assert "<think>" not in out
