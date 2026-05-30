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
    }
    return {**base, **kwargs}

@pytest.mark.asyncio
async def test_triage_node_valid():
    payload = '{"is_valid": true, "reason": "real tool", "trust_score": 0.82}'
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": payload})
        )
        result = await triage_node(_state())
    assert result["is_valid"] is True
    assert result["trust_score"] == 0.82

@pytest.mark.asyncio
async def test_triage_node_invalid():
    payload = '{"is_valid": false, "reason": "marketing", "trust_score": 0.1}'
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": payload})
        )
        result = await triage_node(_state())
    assert result["is_valid"] is False
    assert result["trust_score"] == 0.1

@pytest.mark.asyncio
async def test_triage_node_parse_error():
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "not json at all"})
        )
        result = await triage_node(_state())
    assert result["is_valid"] is False
    assert result["trust_score"] is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.5", 1.0),
        ("-0.2", 0.0),
        ('"high"', None),
        ("null", None),
        ("true", None),
        ("false", None),
    ],
)
def test_triage_result_clamps_out_of_range_trust_score(raw, expected):
    from argos.brain.nodes.triage import _TriageResult

    payload = (
        '{"is_valid": true, "reason": "x", "trust_score": ' + raw + '}'
    )
    result = _TriageResult.model_validate_json(payload)
    assert result.trust_score == expected


@pytest.mark.asyncio
async def test_triage_node_extracts_summary():
    payload = (
        '{"is_valid": true, "reason": "real", "trust_score": 0.7,'
        ' "summary": "A short factual blurb."}'
    )
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": payload})
        )
        result = await triage_node(_state())
    assert result["summary"] == "A short factual blurb."


@pytest.mark.asyncio
async def test_triage_node_drops_summary_when_invalid():
    payload = (
        '{"is_valid": false, "reason": "marketing", "trust_score": 0.1,'
        ' "summary": "ignored when invalid"}'
    )
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": payload})
        )
        result = await triage_node(_state())
    assert result["is_valid"] is False
    assert result["summary"] is None


@pytest.mark.asyncio
async def test_triage_prompt_uses_configured_summary_language(monkeypatch):
    from argos.brain.nodes import triage as triage_module

    monkeypatch.setattr(triage_module.settings.user.slack, "summary_language", "English")
    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured["prompt"] = prompt
            return (
                '{"is_valid": true, "reason": "x", "trust_score": 0.5,'
                ' "summary": "An English blurb."}'
            )

        async def unload(self, model_role):
            return None

    monkeypatch.setattr(triage_module, "get_llm_client", lambda: _FakeClient())

    result = await triage_node(_state())
    assert "English" in captured["prompt"]
    assert result["summary"] == "An English blurb."


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"a normal summary"', "a normal summary"),
        ('"  padded  "', "padded"),
        ("null", None),
        ('"   "', None),
        ('"null"', None),
        ("false", None),
        ("42", None),
    ],
)
def test_triage_result_normalizes_summary(raw, expected):
    from argos.brain.nodes.triage import _TriageResult

    payload = (
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": ' + raw + "}"
    )
    result = _TriageResult.model_validate_json(payload)
    assert result.summary == expected


def test_triage_result_truncates_long_summary():
    from argos.brain.nodes.triage import _SUMMARY_MAX_CHARS, _TriageResult

    long_text = "x" * (_SUMMARY_MAX_CHARS + 200)
    payload = (
        '{"is_valid": true, "reason": "y", "trust_score": 0.5, "summary": "'
        + long_text
        + '"}'
    )
    result = _TriageResult.model_validate_json(payload)
    assert len(result.summary) == _SUMMARY_MAX_CHARS

# ---------------------------------------------------------------------------
# _TriageResult — category field validator (ARG-54)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"Mainstream"', "Mainstream"),
        ('"mainstream"', "Mainstream"),
        ('"MAINSTREAM"', "Mainstream"),
        ('"Alpha"', "Alpha"),
        ('"alpha"', "Alpha"),
        ('"ALPHA"', "Alpha"),
        # fallback cases
        ("null", "Alpha"),
        ('"null"', "Alpha"),
        ('"none"', "Alpha"),
        ('"garbage"', "Alpha"),
        ('"  "', "Alpha"),
    ],
)
def test_triage_result_category_validator(raw, expected):
    from argos.brain.nodes.triage import _TriageResult

    payload = (
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "category": ' + raw + "}"
    )
    result = _TriageResult.model_validate_json(payload)
    assert result.category.value == expected


def test_triage_result_category_defaults_to_alpha_when_field_absent():
    from argos.brain.nodes.triage import _TriageResult
    from argos.models.tech_item import CategoryType

    payload = '{"is_valid": true, "reason": "x", "trust_score": 0.5}'
    result = _TriageResult.model_validate_json(payload)
    assert result.category is CategoryType.ALPHA


# ---------------------------------------------------------------------------
# triage_node — category propagation (ARG-54)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_node_propagates_mainstream_category(monkeypatch):
    """When LLM returns category=Mainstream on a valid item, state carries MAINSTREAM."""
    from argos.brain.nodes import triage as triage_module
    from argos.models.tech_item import CategoryType

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.7, "summary": "ok", "category": "Mainstream"}',
        captured,
    )

    result = await triage_node(_state(raw_text="React 19 stable release."))
    assert result["category"] is CategoryType.MAINSTREAM
    assert result["is_valid"] is True


@pytest.mark.asyncio
async def test_triage_node_propagates_alpha_category(monkeypatch):
    """When LLM returns category=Alpha on a valid item, state carries ALPHA."""
    from argos.brain.nodes import triage as triage_module
    from argos.models.tech_item import CategoryType

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.6, "summary": "ok", "category": "Alpha"}',
        captured,
    )

    result = await triage_node(_state(raw_text="Experimental LLM inference runtime."))
    assert result["category"] is CategoryType.ALPHA
    assert result["is_valid"] is True


@pytest.mark.asyncio
async def test_triage_node_category_fallback_to_alpha_when_field_missing(monkeypatch):
    """When LLM omits category field, validator defaults to ALPHA and it is propagated."""
    from argos.brain.nodes import triage as triage_module
    from argos.models.tech_item import CategoryType

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "ok"}',
        captured,
    )

    result = await triage_node(_state(raw_text="A new tool."))
    assert result["category"] is CategoryType.ALPHA


@pytest.mark.asyncio
async def test_triage_node_category_fallback_to_alpha_when_garbage(monkeypatch):
    """When LLM returns garbage for category, validator falls back to ALPHA."""
    from argos.brain.nodes import triage as triage_module
    from argos.models.tech_item import CategoryType

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "ok", "category": "banana"}',
        captured,
    )

    result = await triage_node(_state(raw_text="A new tool."))
    assert result["category"] is CategoryType.ALPHA


@pytest.mark.asyncio
async def test_triage_node_category_none_when_item_invalid(monkeypatch):
    """When item is invalid, category must be None regardless of LLM value."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": false, "reason": "marketing", "trust_score": 0.1, "category": "Mainstream"}',
        captured,
    )

    result = await triage_node(_state(raw_text="Pure marketing copy."))
    assert result["is_valid"] is False
    assert result["category"] is None


@pytest.mark.asyncio
async def test_triage_node_category_none_on_parse_error(monkeypatch):
    """On LLM parse failure, category must be None."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])

    class _BrokenClient:
        async def query(self, model_role, prompt, **kwargs):
            return "not json at all"

        async def unload(self, model_role):
            return None

    monkeypatch.setattr(triage_module, "get_llm_client", lambda: _BrokenClient())

    result = await triage_node(_state(raw_text="Test."))
    assert result["is_valid"] is False
    assert result["category"] is None


@pytest.mark.asyncio
async def test_triage_node_source_hint_in_prompt_when_present(monkeypatch):
    """When state has source_category, the prompt must include the source hint."""
    from argos.brain.nodes import triage as triage_module
    from argos.models.tech_item import CategoryType

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.6, "summary": "ok", "category": "Mainstream"}',
        captured,
    )

    await triage_node(_state(raw_text="React release.", source_category=CategoryType.MAINSTREAM))
    assert "Source hint" in captured["prompt"]
    assert "Mainstream" in captured["prompt"]


@pytest.mark.asyncio
async def test_triage_node_no_source_hint_in_prompt_when_absent(monkeypatch):
    """When state has source_category=None, no source hint line in prompt."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.6, "summary": "ok"}',
        captured,
    )

    await triage_node(_state(raw_text="React release.", source_category=None))
    assert "Source hint" not in captured["prompt"]


@pytest.mark.asyncio
async def test_triage_node_source_hint_accepts_plain_string(monkeypatch):
    """source_category supplied as a plain string must not raise AttributeError."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.6, "summary": "ok", "category": "Mainstream"}',
        captured,
    )

    # Pass source_category as a raw string, as fetcher-provided item dicts would.
    await triage_node(_state(raw_text="React release.", source_category="Mainstream"))
    assert "Source hint" in captured["prompt"]
    assert "Mainstream" in captured["prompt"]


@pytest.mark.asyncio
async def test_triage_node_source_hint_unrecognised_string_produces_no_hint(monkeypatch):
    """An unrecognised source_category string must produce no hint, not an error."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.6, "summary": "ok"}',
        captured,
    )

    await triage_node(_state(raw_text="React release.", source_category="garbage_value"))
    assert "Source hint" not in captured["prompt"]


@pytest.mark.asyncio
async def test_embed_node_skips_if_invalid():
    session = AsyncMock()
    result = await embed_and_search_node(_state(is_valid=False), session=session)
    assert result["related_tech_ids"] == []
    session.execute.assert_not_called()

@pytest.mark.asyncio
async def test_save_node_skips_if_invalid():
    session = AsyncMock()
    await save_node(_state(is_valid=False), session=session)
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
async def test_save_node_persists_trust_score():
    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True, trust_score=0.73), session=session)
    added_item = session.add.call_args[0][0]
    assert added_item.trust_score == 0.73


@pytest.mark.asyncio
async def test_save_node_persists_summary():
    session = _mock_session_no_existing()
    await save_node(
        _state(is_valid=True, summary="A short blurb."), session=session
    )
    added_item = session.add.call_args[0][0]
    assert added_item.summary == "A short blurb."


@pytest.mark.asyncio
async def test_save_node_persists_null_summary_by_default():
    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True), session=session)
    added_item = session.add.call_args[0][0]
    assert added_item.summary is None


@pytest.mark.asyncio
async def test_save_node_persists_published_at():
    from datetime import datetime, timezone
    session = _mock_session_no_existing()
    pub = datetime(2024, 7, 4, 15, 0, 0, tzinfo=timezone.utc)
    await save_node(_state(is_valid=True, published_at=pub), session=session)
    added_item = session.add.call_args[0][0]
    assert added_item.published_at == pub


@pytest.mark.asyncio
async def test_save_node_persists_null_published_at_by_default():
    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True), session=session)
    added_item = session.add.call_args[0][0]
    assert added_item.published_at is None


@pytest.mark.asyncio
async def test_save_node_creates_succession():
    # execute is called twice: source_url duplicate check (None) then predecessor existence check (found)
    no_existing = MagicMock()
    no_existing.scalar_one_or_none.return_value = None
    predecessor_found = MagicMock()
    predecessor_found.scalar_one_or_none.return_value = uuid.uuid4()
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[no_existing, predecessor_found])
    session.flush = AsyncMock()

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

def _mock_embed_session(*, count: int, rows: list | None = None) -> AsyncMock:
    """Build an AsyncMock session that satisfies embed_and_search_node's
    two-query contract: a count() probe followed (optionally) by a Top-5 query.
    """
    session = AsyncMock()
    count_result = MagicMock()
    count_result.scalar.return_value = count
    rows_result = MagicMock()
    rows_result.fetchall.return_value = rows or []
    session.execute = AsyncMock(side_effect=[count_result, rows_result])
    return session


@pytest.mark.asyncio
async def test_embed_node_happy_path(monkeypatch):
    from argos.brain.nodes import embed as embed_module

    monkeypatch.setattr(embed_module.settings.user.genealogist, "min_db_items", 0)
    embedding = [0.5] * 10
    mock_row = MagicMock()
    mock_row.id = uuid.uuid4()
    mock_row.title = "Similar Tech"
    mock_row.raw_content = "some content"

    session = _mock_embed_session(count=100, rows=[mock_row])

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": embedding})
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)

    assert result["related_tech_ids"] == [str(mock_row.id)]
    assert result["extracted_info"]["embedding"] == embedding
    assert len(result["extracted_info"]["similar_items"]) == 1
    assert result.get("genealogy_skipped", False) is False


@pytest.mark.asyncio
async def test_embed_node_handles_empty_db_rows(monkeypatch):
    """When the count is above threshold but Top-5 returns no rows, the run
    still gets flagged as cold-start so genealogist is skipped."""
    from argos.brain.nodes import embed as embed_module

    monkeypatch.setattr(embed_module.settings.user.genealogist, "min_db_items", 0)
    embedding = [0.1, 0.2, 0.3]
    session = _mock_embed_session(count=100, rows=[])

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": embedding})
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)

    assert result["related_tech_ids"] == []
    assert result["extracted_info"]["similar_items"] == []
    assert result["extracted_info"]["embedding"] == embedding
    assert result["genealogy_skipped"] is True
    assert result["genealogy_skip_reason"] == "cold_start"


@pytest.mark.asyncio
async def test_embed_node_handles_http_error(monkeypatch):
    from argos.brain.nodes import embed as embed_module

    monkeypatch.setattr(embed_module.settings.user.genealogist, "min_db_items", 0)
    # The count query succeeds; the embedding HTTP call then fails.
    session = AsyncMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 100
    session.execute = AsyncMock(return_value=count_result)
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(500)
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)

    assert result["related_tech_ids"] == []
    assert result["extracted_info"] is None


# ---------------------------------------------------------------------------
# embed_and_search_node — cold-start branch (ARG-39)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_node_skips_genealogy_when_below_threshold(monkeypatch):
    from argos.brain.nodes import embed as embed_module

    monkeypatch.setattr(
        embed_module.settings.user.genealogist, "min_db_items", 50
    )
    embedding = [0.0] * 5
    session = AsyncMock()
    count_result = MagicMock()
    count_result.scalar.return_value = 3  # well below threshold
    session.execute = AsyncMock(return_value=count_result)

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": embedding})
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)

    assert result["genealogy_skipped"] is True
    assert result["genealogy_skip_reason"] == "cold_start"
    assert result["related_tech_ids"] == []
    # Similarity query must NOT be executed on cold start.
    assert session.execute.await_count == 1


@pytest.mark.asyncio
async def test_embed_node_runs_genealogy_when_at_or_above_threshold(monkeypatch):
    from argos.brain.nodes import embed as embed_module

    monkeypatch.setattr(
        embed_module.settings.user.genealogist, "min_db_items", 50
    )
    embedding = [0.0] * 5
    mock_row = MagicMock()
    mock_row.id = uuid.uuid4()
    mock_row.title = "Tech"
    mock_row.raw_content = "content"
    session = _mock_embed_session(count=50, rows=[mock_row])

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": embedding})
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)

    assert result.get("genealogy_skipped", False) is False
    assert result["related_tech_ids"] == [str(mock_row.id)]


@pytest.mark.asyncio
async def test_embed_node_threshold_zero_never_skips(monkeypatch):
    from argos.brain.nodes import embed as embed_module

    monkeypatch.setattr(embed_module.settings.user.genealogist, "min_db_items", 0)
    embedding = [0.0] * 3
    mock_row = MagicMock()
    mock_row.id = uuid.uuid4()
    mock_row.title = "T"
    mock_row.raw_content = "c"
    session = _mock_embed_session(count=0, rows=[mock_row])

    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/embeddings").mock(
            return_value=httpx.Response(200, json={"embedding": embedding})
        )
        result = await embed_and_search_node(_state(is_valid=True), session=session)

    # threshold=0 means even an empty DB clears the bar; cold-start only fires
    # if the Top-5 search also returns no rows (covered by another test).
    assert result.get("genealogy_skipped", False) is False


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
async def test_genealogist_node_respects_skip_flag(monkeypatch):
    """When BrainState carries genealogy_skipped=True, the node must not
    invoke the LLM and must leave succession_result alone."""
    from argos.brain.nodes import genealogist as gen_module

    called = {"count": 0}

    class _BoomClient:
        async def query(self, *args, **kwargs):  # pragma: no cover - must not run
            called["count"] += 1
            raise AssertionError("LLM must not be called when skip flag is set")

    monkeypatch.setattr(gen_module, "get_genealogist_llm_client", lambda: _BoomClient())

    state = _genealogist_state(
        genealogy_skipped=True, genealogy_skip_reason="cold_start"
    )
    result = await genealogist_node(state)
    assert called["count"] == 0
    assert result["succession_result"] is None
    assert result["genealogy_skipped"] is True


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


def test_succession_result_normalizes_string_null():
    from argos.brain.nodes.genealogist import _SuccessionResult

    for raw in ['"null"', '"NULL"', '" None "', '""']:
        payload = (
            '{"replace_target_id": ' + raw + ', '
            '"relation_type": null, "reason": "no relation"}'
        )
        result = _SuccessionResult.model_validate_json(payload)
        assert result.replace_target_id is None


@pytest.mark.asyncio
async def test_genealogist_disables_qwen_thinking_via_api(monkeypatch):
    # qwen3 emits a long `<think>...</think>` trace by default which eats the output
    # budget and slows generation by ~14x. The genealogist MUST disable thinking via the
    # Ollama API `think` field — the textual `/no_think` marker is not honored by the
    # qwen3:32b chat template in Ollama 0.21.
    from argos.brain.nodes import genealogist as gen_module

    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured.update(kwargs)
            return '{"replace_target_id": null, "relation_type": null, "reason": "x"}'

    monkeypatch.setattr(gen_module, "get_genealogist_llm_client", lambda: _FakeClient())
    await genealogist_node(_genealogist_state())

    assert captured.get("think") is False


@pytest.mark.asyncio
async def test_genealogist_node_awaits_prewarm_task():
    import asyncio

    awaited = asyncio.Event()

    async def _prewarm():
        awaited.set()

    task = asyncio.create_task(_prewarm())
    payload = '{"replace_target_id": null, "relation_type": null, "reason": "x"}'
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": payload})
        )
        await genealogist_node(_genealogist_state(), prewarm_task=task)

    assert awaited.is_set()
    assert task.done()


@pytest.mark.asyncio
async def test_genealogist_node_uses_large_model_timeout(monkeypatch):
    from argos.brain.nodes import genealogist as gen_module
    from argos.brain.ollama_client import LARGE_MODEL_TIMEOUT

    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, keep_alive="5m", timeout=120, **_kwargs):
            captured["model_role"] = model_role
            captured["timeout"] = timeout
            captured["keep_alive"] = keep_alive
            return '{"replace_target_id": null, "relation_type": null, "reason": "x"}'

    monkeypatch.setattr(gen_module, "get_genealogist_llm_client", lambda: _FakeClient())
    await genealogist_node(_genealogist_state())

    assert captured["model_role"] == "large"
    assert captured["timeout"] == LARGE_MODEL_TIMEOUT
    assert captured["keep_alive"] == "5m"


@pytest.mark.asyncio
async def test_genealogist_node_swallows_prewarm_failure():
    import asyncio

    async def _broken_prewarm():
        raise RuntimeError("ollama unreachable")

    task = asyncio.create_task(_broken_prewarm())
    payload = '{"replace_target_id": null, "relation_type": null, "reason": "x"}'
    with respx.mock:
        respx.post(f"{OLLAMA_BASE_URL}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": payload})
        )
        result = await genealogist_node(_genealogist_state(), prewarm_task=task)

    # Prewarm errors must not stop the genealogist from running its own query.
    assert result["succession_result"] is not None
    assert result["succession_result"]["reason"] == "x"


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


# ---------------------------------------------------------------------------
# save_node — saved flag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_node_sets_saved_true_on_insert():
    session = _mock_session_no_existing()
    result = await save_node(_state(is_valid=True), session=session)
    assert result["saved"] is True


@pytest.mark.asyncio
async def test_save_node_does_not_set_saved_on_duplicate():
    session = _mock_session_with_existing()
    result = await save_node(_state(is_valid=True), session=session)
    assert result["saved"] is False


@pytest.mark.asyncio
async def test_save_node_does_not_set_saved_when_invalid():
    session = AsyncMock()
    result = await save_node(_state(is_valid=False), session=session)
    assert result["saved"] is False


@pytest.mark.asyncio
async def test_save_node_does_not_set_saved_on_empty_source_url():
    session = _mock_session_no_existing()
    result = await save_node(_state(is_valid=True, source_url=""), session=session)
    assert result["saved"] is False


# ---------------------------------------------------------------------------
# save_node — category persistence (ARG-54)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_node_persists_triage_decided_mainstream_category():
    from argos.models.tech_item import CategoryType

    session = _mock_session_no_existing()
    await save_node(
        _state(is_valid=True, category=CategoryType.MAINSTREAM), session=session
    )
    added_item = session.add.call_args[0][0]
    assert added_item.category is CategoryType.MAINSTREAM


@pytest.mark.asyncio
async def test_save_node_persists_triage_decided_alpha_category():
    from argos.models.tech_item import CategoryType

    session = _mock_session_no_existing()
    await save_node(
        _state(is_valid=True, category=CategoryType.ALPHA), session=session
    )
    added_item = session.add.call_args[0][0]
    assert added_item.category is CategoryType.ALPHA


@pytest.mark.asyncio
async def test_save_node_falls_back_to_alpha_when_category_is_none():
    """When state.category is None (e.g. older code path), save_node must default to ALPHA."""
    from argos.models.tech_item import CategoryType

    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True, category=None), session=session)
    added_item = session.add.call_args[0][0]
    assert added_item.category is CategoryType.ALPHA


# ---------------------------------------------------------------------------
# save_node — flush parameter (ARG-90)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_node_flushes_by_default():
    """save_node(state) with default flush=True must call session.flush() exactly once."""
    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True), session=session)
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_node_flush_false_skips_flush():
    """save_node(state, flush=False) must NOT call session.flush() at all."""
    session = _mock_session_no_existing()
    await save_node(_state(is_valid=True), session=session, flush=False)
    session.add.assert_called_once()  # item was still added
    session.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_save_node_flush_false_does_not_set_saved_flag():
    """With flush=False, saved_item_id is populated (PK pre-assigned) but saved stays False.

    The caller is responsible for setting saved=True only after its own flush succeeds,
    so a failed flush cannot leave the state with a misleading saved=True.
    """
    session = _mock_session_no_existing()
    result = await save_node(_state(is_valid=True), session=session, flush=False)
    assert result["saved"] is False
    assert result["saved_item_id"] is not None


@pytest.mark.asyncio
async def test_save_node_flush_false_skips_invalid():
    """flush=False on an invalid state must be a no-op (same as default)."""
    session = AsyncMock()
    result = await save_node(_state(is_valid=False), session=session, flush=False)
    session.add.assert_not_called()
    session.flush.assert_not_awaited()
    assert result["saved"] is False


# ---------------------------------------------------------------------------
# triage_node — Interests injection (ARG-50)
# ---------------------------------------------------------------------------


def _patch_interests(monkeypatch, triage_module, *, topics, exclusions):
    monkeypatch.setattr(triage_module.settings.user.interests, "topics", topics)
    monkeypatch.setattr(
        triage_module.settings.user.interests, "exclusions", exclusions
    )


def _install_fake_client(monkeypatch, triage_module, response: str, captured: dict):
    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured["prompt"] = prompt
            return response

        async def unload(self, model_role):
            return None

    monkeypatch.setattr(triage_module, "get_llm_client", lambda: _FakeClient())


@pytest.mark.asyncio
async def test_triage_empty_interests_preserves_llm_result(monkeypatch):
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "ok"}',
        captured,
    )

    state = _state(raw_text="This is about crypto wallets.")
    result = await triage_node(state)

    assert result["is_valid"] is True
    assert result["trust_score"] == 0.5
    assert result["summary"] == "ok"


@pytest.mark.asyncio
async def test_triage_topic_match_bumps_trust_score(monkeypatch):
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=["RAG"], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.6, "summary": "s"}',
        captured,
    )

    state = _state(raw_text="Retrieval-augmented generation (RAG) pipeline notes.")
    result = await triage_node(state)

    assert result["is_valid"] is True
    assert result["trust_score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_triage_topic_bump_caps_at_one(monkeypatch):
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=["RAG"], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.95, "summary": "s"}',
        captured,
    )

    state = _state(raw_text="A RAG benchmark paper.")
    result = await triage_node(state)

    assert result["trust_score"] == 1.0


@pytest.mark.asyncio
async def test_triage_topic_match_is_case_insensitive(monkeypatch):
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=["rag"], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.6, "summary": "s"}',
        captured,
    )

    state = _state(raw_text="An article describing RAG architecture.")
    result = await triage_node(state)

    assert result["trust_score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_triage_exclusion_match_forces_pass(monkeypatch):
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=["crypto"])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.9, "summary": "shiny"}',
        captured,
    )

    state = _state(raw_text="Crypto wallet integration walkthrough.")
    result = await triage_node(state)

    assert result["is_valid"] is False
    assert result["trust_score"] == 0.0
    assert result["summary"] is None


@pytest.mark.asyncio
async def test_triage_exclusion_beats_topic_when_both_match(monkeypatch):
    from argos.brain.nodes import triage as triage_module

    _patch_interests(
        monkeypatch, triage_module, topics=["RAG"], exclusions=["crypto"]
    )
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.8, "summary": "ok"}',
        captured,
    )

    state = _state(raw_text="A RAG pipeline indexing crypto tweets.")
    result = await triage_node(state)

    assert result["is_valid"] is False
    assert result["trust_score"] == 0.0
    assert result["summary"] is None


@pytest.mark.asyncio
async def test_triage_prompt_includes_interest_topics(monkeypatch):
    from argos.brain.nodes import triage as triage_module

    _patch_interests(
        monkeypatch, triage_module, topics=["RAG", "LLM"], exclusions=[]
    )
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "s"}',
        captured,
    )

    await triage_node(_state(raw_text="An overview of vector search."))

    prompt = captured["prompt"]
    assert "User interests" in prompt
    assert "RAG" in prompt
    assert "LLM" in prompt
    assert "Exclusions" not in prompt


@pytest.mark.asyncio
async def test_triage_prompt_caps_topics_at_max(monkeypatch, caplog):
    import logging as _logging

    from argos.brain.nodes import triage as triage_module

    topics = [f"topic{i}" for i in range(15)]
    _patch_interests(monkeypatch, triage_module, topics=topics, exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "s"}',
        captured,
    )

    with caplog.at_level(_logging.WARNING, logger="argos.brain.nodes.triage"):
        await triage_node(_state(raw_text="Unrelated body."))

    prompt = captured["prompt"]
    for i in range(10):
        assert f"topic{i}" in prompt
    for i in range(10, 15):
        assert f"topic{i}" not in prompt
    assert any("truncated" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_triage_exclusion_past_window_does_not_force_pass(monkeypatch):
    """Exclusion terms beyond the LLM truncation window must not flip the verdict.

    Regression for ARG-50 review: deterministic rules must scan the same
    truncated text passed to the model.
    """
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=["crypto"])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.7, "summary": "shiny"}',
        captured,
    )

    # Push the exclusion term past the 2000-char triage window.
    head = "A clean RAG pipeline article. " * 100
    assert len(head) > 2000
    raw_text = head + " (footer mentions crypto wallets)"

    state = _state(raw_text=raw_text)
    result = await triage_node(state)

    # Exclusion is outside the LLM-visible window, so the verdict must stand.
    assert result["is_valid"] is True
    assert result["trust_score"] == pytest.approx(0.7)
    assert result["summary"] == "shiny"


@pytest.mark.asyncio
async def test_triage_topic_past_window_does_not_bump_trust(monkeypatch):
    """Topic terms beyond the LLM truncation window must not bump trust_score."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=["RAG"], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "ok"}',
        captured,
    )

    head = "Unrelated boilerplate filler sentence. " * 100
    assert len(head) > 2000
    raw_text = head + " RAG appears only here in the trailing section."

    state = _state(raw_text=raw_text)
    result = await triage_node(state)

    # Topic match is outside the visible window: no trust bump should fire.
    assert result["is_valid"] is True
    assert result["trust_score"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_triage_normalizes_blank_and_non_string_terms(monkeypatch):
    from argos.brain.nodes import triage as triage_module

    _patch_interests(
        monkeypatch,
        triage_module,
        topics=["", " ", None, "RAG"],  # type: ignore[list-item]
        exclusions=[],
    )
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "s"}',
        captured,
    )

    await triage_node(_state(raw_text="Some RAG content."))

    prompt = captured["prompt"]
    assert "RAG" in prompt
    # No empty-term artifacts like ", ," in the rendered list
    assert ", ," not in prompt
    assert ":  ," not in prompt


# ---------------------------------------------------------------------------
# triage_node — is_relevant gating (ARG-86)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_is_relevant_false_demotes_to_invalid(monkeypatch):
    """When LLM emits is_relevant=false with non-empty topics, item must be demoted."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=["AI", "LLM"], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "real tech", "trust_score": 0.8, "summary": "ok", "is_relevant": false}',
        captured,
    )

    state = _state(raw_text="A tool for managing npm supply-chain vulnerabilities.")
    result = await triage_node(state)

    assert result["is_valid"] is False
    assert result["trust_score"] is None
    assert result["summary"] is None


@pytest.mark.asyncio
async def test_triage_is_relevant_true_with_topics_passes_through_rules(monkeypatch):
    """When LLM emits is_relevant=true with non-empty topics, _apply_interest_rules runs."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=["LLM"], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.6, "summary": "s", "is_relevant": true}',
        captured,
    )

    # text contains the topic term so trust-score bump fires
    state = _state(raw_text="LLM benchmarking framework released.")
    result = await triage_node(state)

    assert result["is_valid"] is True
    # trust bump of 0.1 applied via _apply_interest_rules
    assert result["trust_score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_triage_empty_topics_no_is_relevant_key_regression(monkeypatch):
    """When topics is empty, a response without is_relevant still validates (fail-open)."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "ok"}',
        captured,
    )

    state = _state(raw_text="A new database indexing library.")
    result = await triage_node(state)

    # Behavior identical to pre-ARG-86: is_relevant key absent, no gating
    assert result["is_valid"] is True
    assert result["trust_score"] == pytest.approx(0.5)
    assert result["summary"] == "ok"
    # Prompt must not mention is_relevant when topics empty
    assert "is_relevant" not in captured["prompt"]


@pytest.mark.asyncio
async def test_triage_npm_supply_chain_irrelevant_to_ai_topics(monkeypatch):
    """Concrete scenario: npm supply-chain text with AI/LLM/Agent topics -> not saved."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(
        monkeypatch, triage_module, topics=["AI", "LLM", "Agent"], exclusions=[]
    )
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "real tool", "trust_score": 0.75, "summary": "npm audit tool", "is_relevant": false}',
        captured,
    )

    state = _state(
        raw_text=(
            "socket.dev launches supply-chain scanner for npm packages, "
            "detecting malicious dependencies before they ship to production."
        )
    )
    result = await triage_node(state)

    # LLM judged not relevant to AI/LLM/Agent topics → demoted to invalid → not saved
    assert result["is_valid"] is False
    assert result["trust_score"] is None
    assert result["summary"] is None


@pytest.mark.asyncio
async def test_triage_prompt_includes_is_relevant_schema_when_topics_present(monkeypatch):
    """When topics is non-empty, the prompt must include is_relevant in the JSON schema example."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=["AI"], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "s", "is_relevant": true}',
        captured,
    )

    await triage_node(_state(raw_text="An AI inference runtime."))

    assert "is_relevant" in captured["prompt"]


@pytest.mark.asyncio
async def test_triage_prompt_omits_is_relevant_schema_when_topics_empty(monkeypatch):
    """When topics is empty, the prompt must NOT include is_relevant in the JSON schema."""
    from argos.brain.nodes import triage as triage_module

    _patch_interests(monkeypatch, triage_module, topics=[], exclusions=[])
    captured: dict = {}
    _install_fake_client(
        monkeypatch,
        triage_module,
        '{"is_valid": true, "reason": "x", "trust_score": 0.5, "summary": "s"}',
        captured,
    )

    await triage_node(_state(raw_text="A new database library."))

    assert "is_relevant" not in captured["prompt"]


# ---------------------------------------------------------------------------
# ARG-127: language config respected in brain LLM prompts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_prompt_reason_uses_configured_language(monkeypatch):
    """The triage prompt must instruct the LLM to write reason in the configured language."""
    from argos.brain.nodes import triage as triage_module

    monkeypatch.setattr(triage_module.settings.user.slack, "summary_language", "Korean")
    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured["prompt"] = prompt
            return (
                '{"is_valid": true, "reason": "이유", "trust_score": 0.5,'
                ' "summary": "요약."}'
            )

        async def unload(self, model_role):
            return None

    monkeypatch.setattr(triage_module, "get_llm_client", lambda: _FakeClient())

    await triage_node(_state())
    assert "Korean" in captured["prompt"]


@pytest.mark.asyncio
async def test_triage_prompt_language_fallback_when_empty(monkeypatch):
    """When summary_language is empty, the prompt must fall back to 'English'."""
    from argos.brain.nodes import triage as triage_module

    monkeypatch.setattr(triage_module.settings.user.slack, "summary_language", "")
    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured["prompt"] = prompt
            return (
                '{"is_valid": true, "reason": "reason", "trust_score": 0.5,'
                ' "summary": "blurb."}'
            )

        async def unload(self, model_role):
            return None

    monkeypatch.setattr(triage_module, "get_llm_client", lambda: _FakeClient())

    await triage_node(_state())
    assert "English" in captured["prompt"]


@pytest.mark.asyncio
async def test_genealogist_prompt_uses_configured_language(monkeypatch):
    """The genealogist prompt must instruct the LLM to write reason in the configured language."""
    from argos.brain.nodes import genealogist as gen_module

    monkeypatch.setattr(gen_module.settings.user.slack, "summary_language", "Korean")
    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"replace_target_id": null, "relation_type": null, "reason": "이유"}'

    monkeypatch.setattr(gen_module, "get_genealogist_llm_client", lambda: _FakeClient())
    await genealogist_node(_genealogist_state())

    assert "Korean" in captured["prompt"]


@pytest.mark.asyncio
async def test_genealogist_prompt_language_fallback_when_empty(monkeypatch):
    """When summary_language is empty, genealogist prompt must fall back to 'English'."""
    from argos.brain.nodes import genealogist as gen_module

    monkeypatch.setattr(gen_module.settings.user.slack, "summary_language", "")
    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"replace_target_id": null, "relation_type": null, "reason": "reason"}'

    monkeypatch.setattr(gen_module, "get_genealogist_llm_client", lambda: _FakeClient())
    await genealogist_node(_genealogist_state())

    assert "English" in captured["prompt"]


@pytest.mark.asyncio
async def test_genealogist_prompt_does_not_contain_language_in_json_keys(monkeypatch):
    """Language directive must NOT appear inside the JSON schema example to avoid
    the model translating enum values like relation_type."""
    from argos.brain.nodes import genealogist as gen_module

    monkeypatch.setattr(gen_module.settings.user.slack, "summary_language", "Korean")
    captured: dict = {}

    class _FakeClient:
        async def query(self, model_role, prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"replace_target_id": null, "relation_type": null, "reason": "이유"}'

    monkeypatch.setattr(gen_module, "get_genealogist_llm_client", lambda: _FakeClient())
    await genealogist_node(_genealogist_state())

    # JSON schema line must not be language-modified — these keys stay English
    assert '"Replace or Enhance or Fork or null"' in captured["prompt"]
