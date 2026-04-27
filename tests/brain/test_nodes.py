from __future__ import annotations
import uuid
import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock
from argos.brain.graph_state import BrainState
from argos.brain.nodes.triage import triage_node
from argos.brain.nodes.embed import embed_and_search_node
from argos.brain.nodes.genealogist import genealogist_node
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


# ---------------------------------------------------------------------------
# Helpers for save / embed happy-path tests
# ---------------------------------------------------------------------------

def _mock_session_no_existing() -> MagicMock:
    session = MagicMock()
    session.execute = AsyncMock()
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = None
    session.execute.return_value = execute_result
    session.flush = AsyncMock()
    return session


def _mock_session_with_existing() -> MagicMock:
    session = MagicMock()
    session.execute = AsyncMock()
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = uuid.uuid4()
    session.execute.return_value = execute_result
    session.flush = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# save_node — happy-path and branch coverage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_node_creates_item_when_valid():
    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True), session=session)
    session.add.assert_called_once()
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_node_skips_duplicate_url():
    session = _mock_session_with_existing()
    await save_node(_state(is_valid=True), session=session)
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_save_node_attaches_embedding():
    session = _mock_session_no_existing()
    embedding = [0.1, 0.2, 0.3]
    await save_node(
        _state(is_valid=True, extracted_info={"embedding": embedding}),
        session=session,
    )
    added_item = session.add.call_args[0][0]
    assert added_item.embedding == embedding


@pytest.mark.asyncio
async def test_save_node_creates_succession():
    session = _mock_session_no_existing()
    predecessor_id = str(uuid.uuid4())
    await save_node(
        _state(
            is_valid=True,
            succession_result={
                "replace_target_id": predecessor_id,
                "relation_type": "Replace",
                "reason": "superseded",
            },
        ),
        session=session,
    )
    # add called twice: TechItem + TechSuccession
    assert session.add.call_count == 2


@pytest.mark.asyncio
async def test_save_node_skips_succession_on_unknown_relation_type():
    session = _mock_session_no_existing()
    await save_node(
        _state(
            is_valid=True,
            succession_result={
                "replace_target_id": str(uuid.uuid4()),
                "relation_type": "Unknown",
                "reason": "unrecognised type",
            },
        ),
        session=session,
    )
    assert session.add.call_count == 1  # only TechItem, no TechSuccession


@pytest.mark.asyncio
async def test_save_node_skips_succession_when_replace_target_is_none():
    session = _mock_session_no_existing()
    await save_node(
        _state(
            is_valid=True,
            succession_result={
                "replace_target_id": None,
                "relation_type": "Replace",
                "reason": "no predecessor",
            },
        ),
        session=session,
    )
    assert session.add.call_count == 1


@pytest.mark.asyncio
async def test_save_node_uses_untitled_when_raw_text_empty():
    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True, raw_text=""), session=session)
    added_item = session.add.call_args[0][0]
    assert added_item.title == "Untitled"


# ---------------------------------------------------------------------------
# embed_and_search_node — happy-path and error branches
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_node_happy_path():
    embedding = [0.5] * 10
    mock_row = MagicMock()
    mock_row.id = uuid.uuid4()
    mock_row.title = "Similar Tech"
    mock_row.raw_content = "some content"

    session = AsyncMock()
    db_result = MagicMock()
    db_result.fetchall.return_value = [mock_row]
    session.execute.return_value = db_result

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": embedding})
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)

    assert result["related_tech_ids"] == [str(mock_row.id)]
    assert result["extracted_info"]["embedding"] == embedding
    assert len(result["extracted_info"]["similar_items"]) == 1


@pytest.mark.asyncio
async def test_embed_node_handles_empty_db_rows():
    embedding = [0.1, 0.2, 0.3]
    session = AsyncMock()
    db_result = MagicMock()
    db_result.fetchall.return_value = []
    session.execute.return_value = db_result

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": embedding})
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)

    assert result["related_tech_ids"] == []
    assert result["extracted_info"]["similar_items"] == []
    assert result["extracted_info"]["embedding"] == embedding


@pytest.mark.asyncio
async def test_embed_node_handles_http_error():
    session = AsyncMock()
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(500)
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)

    assert result["related_tech_ids"] == []
    assert result["extracted_info"] is None
    session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# genealogist_node — all branches
# ---------------------------------------------------------------------------

def _genealogist_state(**kwargs):
    return _state(
        is_valid=True,
        related_tech_ids=["abc-123"],
        extracted_info={
            "similar_items": [
                {"id": "abc-123", "title": "Old Tech", "raw_content": "legacy content"}
            ]
        },
        **kwargs,
    )


@pytest.mark.asyncio
async def test_genealogist_node_skips_if_invalid():
    result = await genealogist_node(_state(is_valid=False, related_tech_ids=["abc"]))
    assert result["succession_result"] is None


@pytest.mark.asyncio
async def test_genealogist_node_skips_if_no_related_ids():
    result = await genealogist_node(_state(is_valid=True, related_tech_ids=[]))
    assert result["succession_result"] is None


@pytest.mark.asyncio
async def test_genealogist_node_skips_if_no_similar_items():
    result = await genealogist_node(
        _state(
            is_valid=True,
            related_tech_ids=["abc-123"],
            extracted_info={"similar_items": []},
        )
    )
    assert result["succession_result"] is None


@pytest.mark.asyncio
async def test_genealogist_node_happy_path():
    payload = '{"replace_target_id": "abc-123", "relation_type": "Replace", "reason": "superseded"}'
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": payload})
        )
        result = await genealogist_node(_genealogist_state())

    assert result["succession_result"]["replace_target_id"] == "abc-123"
    assert result["succession_result"]["relation_type"] == "Replace"
    assert result["succession_result"]["reason"] == "superseded"


@pytest.mark.asyncio
async def test_genealogist_node_null_replace_target():
    payload = '{"replace_target_id": null, "relation_type": null, "reason": "no relation"}'
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": payload})
        )
        result = await genealogist_node(_genealogist_state())

    assert result["succession_result"]["replace_target_id"] is None
    assert result["succession_result"]["relation_type"] is None


@pytest.mark.asyncio
async def test_genealogist_node_parse_error():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "not valid json"})
        )
        result = await genealogist_node(_genealogist_state())

    assert result["succession_result"] is None


# ---------------------------------------------------------------------------
# save_node — edge cases from security review
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_node_skips_succession_on_invalid_uuid():
    session = _mock_session_no_existing()
    await save_node(
        _state(
            is_valid=True,
            succession_result={
                "replace_target_id": "not-a-uuid",
                "relation_type": "Replace",
                "reason": "llm hallucinated an id",
            },
        ),
        session=session,
    )
    assert session.add.call_count == 1  # TechItem only, TechSuccession skipped
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_node_skips_on_empty_source_url():
    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True, source_url=""), session=session)
    session.add.assert_not_called()
    session.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_node_uses_untitled_when_raw_text_is_whitespace_only():
    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True, raw_text="   \n\t\n  "), session=session)
    added_item = session.add.call_args[0][0]
    assert added_item.title == "Untitled"


# ---------------------------------------------------------------------------
# triage_node / genealogist_node — Pydantic validation failure (valid JSON, wrong schema)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_triage_node_valid_json_fails_pydantic_schema():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": '{"is_valid": "yes", "reason": 42}'})
        )
        result = await triage_node(_state())
    assert result["is_valid"] is False


@pytest.mark.asyncio
async def test_genealogist_node_valid_json_fails_pydantic_schema():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": '{"replace_target_id": 999, "relation_type": null}'})
        )
        result = await genealogist_node(_genealogist_state())
    assert result["succession_result"] is None


# ---------------------------------------------------------------------------
# embed_and_search_node — DB error after successful embedding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_node_handles_db_error_after_successful_embedding():
    embedding = [0.1, 0.2, 0.3]
    session = AsyncMock()
    session.execute.side_effect = Exception("pgvector not available")
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": embedding})
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)
    assert result["related_tech_ids"] == []
    assert result["extracted_info"] is None
