from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from argos.models.tech_item import CategoryType
from argos.models.user_asset import AssetStatus
from argos.slack.services.briefing_query import fetch_today_briefing, fetch_user_portfolio, KST


def _make_tech_item(category: CategoryType, trust_score: float | None, created_at: datetime):
    item = MagicMock()
    item.id = uuid.uuid4()
    item.category = category
    item.trust_score = trust_score
    item.created_at = created_at
    item.source_url = f"https://example-{uuid.uuid4()}.com/article"
    item.embedding = None
    return item


# ---------------------------------------------------------------------------
# ARG-132: published_at lookback filter tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_today_briefing_uses_published_at_lookback_window():
    """fetch_today_briefing with lookback_days must filter by published_at >= cutoff."""
    now_utc = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    captured_stmts = []

    async def fake_execute(stmt):
        captured_stmts.append(stmt)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    await fetch_today_briefing(mock_session, now_utc=now_utc, lookback_days=7)

    # Filter to only the per-category queries (they reference tech_items.category)
    category_stmts = [
        stmt for stmt in captured_stmts
        if "tech_items.category" in str(stmt.compile(compile_kwargs={"literal_binds": True}))
    ]
    assert len(category_stmts) == 2  # one per category
    for stmt in category_stmts:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "published_at" in compiled
        # Must NOT use old created_at day-window logic
        assert "created_at >=" not in compiled


@pytest.mark.asyncio
async def test_fetch_today_briefing_lookback_days_default_is_seven():
    """fetch_today_briefing without explicit lookback_days defaults to 7-day window."""
    now_utc = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    captured_stmts = []

    async def fake_execute(stmt):
        captured_stmts.append(stmt)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    await fetch_today_briefing(mock_session, now_utc=now_utc)

    category_stmts = [
        stmt for stmt in captured_stmts
        if "tech_items.category" in str(stmt.compile(compile_kwargs={"literal_binds": True}))
    ]
    assert len(category_stmts) == 2
    for stmt in category_stmts:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "published_at" in compiled


@pytest.mark.asyncio
async def test_fetch_today_briefing_null_published_at_falls_back_to_created_at():
    """Items with NULL published_at must still appear when created_at is within the lookback window.

    This covers the 'argos add' / Slack add path: run_brain_pipeline sets published_at=None,
    so those items must use created_at as the effective date instead of being silently dropped.
    """
    now_utc = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    captured_stmts = []

    async def fake_execute(stmt):
        captured_stmts.append(stmt)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    await fetch_today_briefing(mock_session, now_utc=now_utc, lookback_days=7)

    category_stmts = [
        stmt for stmt in captured_stmts
        if "tech_items.category" in str(stmt.compile(compile_kwargs={"literal_binds": True}))
    ]
    assert len(category_stmts) == 2
    for stmt in category_stmts:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        # COALESCE(published_at, created_at) must be used so NULL published_at falls back to created_at
        assert "coalesce" in compiled.lower()
        assert "published_at" in compiled
        assert "created_at" in compiled
        # Must NOT have a bare published_at IS NOT NULL exclusion
        assert "published_at IS NOT NULL" not in compiled.upper()


@pytest.mark.asyncio
async def test_briefing_config_has_lookback_days_field():
    """BriefingConfig must expose a lookback_days field defaulting to 7."""
    from argos.config import BriefingConfig
    cfg = BriefingConfig(weekdays=["Mon"])
    assert hasattr(cfg, "lookback_days")
    assert cfg.lookback_days == 7


@pytest.mark.asyncio
async def test_briefing_config_lookback_days_is_configurable():
    """BriefingConfig.lookback_days must accept positive integer values."""
    from argos.config import BriefingConfig
    cfg = BriefingConfig(weekdays=["Mon"], lookback_days=14)
    assert cfg.lookback_days == 14


@pytest.mark.asyncio
async def test_kst_window_filters_today_items(now_utc):
    now_kst = now_utc.astimezone(KST)
    start_kst = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
    within_window = start_kst.astimezone(timezone.utc) + timedelta(hours=5)
    outside_window = start_kst.astimezone(timezone.utc) - timedelta(hours=1)

    today_item = _make_tech_item(CategoryType.MAINSTREAM, 0.8, within_window)
    _make_tech_item(CategoryType.MAINSTREAM, 0.9, outside_window)

    captured_queries = []

    async def fake_execute(stmt):
        captured_queries.append(stmt)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [today_item]
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session, now_utc=now_utc)

    assert CategoryType.MAINSTREAM in result
    assert CategoryType.ALPHA in result


@pytest.mark.asyncio
async def test_returns_dict_with_both_categories(now_utc):
    ms_item = _make_tech_item(CategoryType.MAINSTREAM, 0.9, now_utc)
    alpha_item = _make_tech_item(CategoryType.ALPHA, 0.5, now_utc)

    call_count = 0

    async def fake_execute(stmt):
        nonlocal call_count
        mock_result = MagicMock()
        if call_count == 0:
            mock_result.scalars.return_value.all.return_value = [ms_item]
        else:
            mock_result.scalars.return_value.all.return_value = [alpha_item]
        call_count += 1
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session, now_utc=now_utc)

    assert CategoryType.MAINSTREAM in result
    assert CategoryType.ALPHA in result


@pytest.mark.asyncio
async def test_limit_per_category_honored(now_utc):
    items = [_make_tech_item(CategoryType.MAINSTREAM, float(i) / 10, now_utc) for i in range(10)]
    returned = items[:3]

    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = returned
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session, now_utc=now_utc, limit_per_category=3)

    assert len(result[CategoryType.MAINSTREAM]) == 3


@pytest.mark.asyncio
async def test_empty_result_when_no_items(now_utc):
    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session, now_utc=now_utc)

    assert result[CategoryType.MAINSTREAM] == []
    assert result[CategoryType.ALPHA] == []


@pytest.mark.asyncio
async def test_default_now_utc_used_when_none():
    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_today_briefing(mock_session)
    assert isinstance(result, dict)
    assert CategoryType.MAINSTREAM in result


# ---------------------------------------------------------------------------
# fetch_user_portfolio tests
# ---------------------------------------------------------------------------


def _make_asset_and_item(
    status: AssetStatus = AssetStatus.KEEP,
    updated_at: datetime | None = None,
) -> tuple[MagicMock, MagicMock]:
    tech_id = uuid.uuid4()
    item = MagicMock()
    item.id = tech_id
    item.title = "Test Tech"
    item.source_url = "https://example.com/tech"

    asset = MagicMock()
    asset.id = uuid.uuid4()
    asset.tech_id = tech_id
    asset.status = status
    asset.updated_at = updated_at or datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)
    asset.last_monitored_at = None

    return asset, item


@pytest.mark.asyncio
async def test_fetch_user_portfolio_returns_keep_assets():
    asset, item = _make_asset_and_item(AssetStatus.KEEP)
    rows = [(asset, item)]

    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_user_portfolio(mock_session)
    assert len(result) == 1
    assert result[0] == (asset, item)


@pytest.mark.asyncio
async def test_fetch_user_portfolio_empty_when_no_assets():
    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_user_portfolio(mock_session)
    assert result == []


@pytest.mark.asyncio
async def test_fetch_user_portfolio_query_filters_keep_and_orders_by_updated_at():
    """Verify the SQL statement targets KEEP status and orders by updated_at DESC."""
    captured = {}

    async def fake_execute(stmt):
        captured["stmt"] = stmt
        mock_result = MagicMock()
        mock_result.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    await fetch_user_portfolio(mock_session)

    stmt = captured["stmt"]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "user_assets" in compiled
    assert "Keep" in compiled


# ---------------------------------------------------------------------------
# fetch_user_portfolio — ARG-112: category filter & sort_by options
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_user_portfolio_no_args_parity():
    """Calling with no args must behave identically to the pre-ARG-112 baseline."""
    asset, item = _make_asset_and_item(AssetStatus.KEEP)
    rows = [(asset, item)]

    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_user_portfolio(mock_session)
    assert result == [(asset, item)]


@pytest.mark.asyncio
async def test_fetch_user_portfolio_category_filter_emits_category_clause():
    """category=CategoryType.ALPHA must inject a category filter into the SQL."""
    captured = {}

    async def fake_execute(stmt):
        captured["stmt"] = stmt
        mock_result = MagicMock()
        mock_result.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    await fetch_user_portfolio(mock_session, category=CategoryType.ALPHA)

    stmt = captured["stmt"]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "Alpha" in compiled


@pytest.mark.asyncio
async def test_fetch_user_portfolio_no_category_filter_omits_category_clause():
    """Without category kwarg, the query must NOT contain a category WHERE clause."""
    captured = {}

    async def fake_execute(stmt):
        captured["stmt"] = stmt
        mock_result = MagicMock()
        mock_result.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    await fetch_user_portfolio(mock_session)

    stmt = captured["stmt"]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    # Neither 'Alpha' nor 'Mainstream' should appear as a filter value.
    # The compiled output for category=None should not have either literal.
    assert "Alpha" not in compiled
    assert "Mainstream" not in compiled


@pytest.mark.asyncio
async def test_fetch_user_portfolio_sort_by_trust_emits_trust_score_order():
    """sort_by='trust' must produce an ORDER BY trust_score DESC NULLS LAST clause."""
    captured = {}

    async def fake_execute(stmt):
        captured["stmt"] = stmt
        mock_result = MagicMock()
        mock_result.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    await fetch_user_portfolio(mock_session, sort_by="trust")

    stmt = captured["stmt"]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    # Extract the ORDER BY portion (everything after "ORDER BY")
    order_by_idx = compiled.upper().rfind("ORDER BY")
    assert order_by_idx != -1, "Expected ORDER BY clause"
    order_by_section = compiled[order_by_idx:]
    assert "trust_score" in order_by_section
    assert "updated_at" in order_by_section
    # trust_score must appear before updated_at in the ORDER BY clause
    ts_pos = order_by_section.find("trust_score")
    ua_pos = order_by_section.find("updated_at")
    assert ts_pos < ua_pos, "trust_score ORDER BY must precede updated_at"


@pytest.mark.asyncio
async def test_fetch_user_portfolio_sort_by_date_default_uses_updated_at():
    """Default sort (sort_by='date') must produce ORDER BY updated_at DESC only."""
    captured = {}

    async def fake_execute(stmt):
        captured["stmt"] = stmt
        mock_result = MagicMock()
        mock_result.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    await fetch_user_portfolio(mock_session)

    stmt = captured["stmt"]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    # Extract the ORDER BY portion (everything after "ORDER BY")
    order_by_idx = compiled.upper().rfind("ORDER BY")
    assert order_by_idx != -1, "Expected ORDER BY clause"
    order_by_section = compiled[order_by_idx:]
    assert "updated_at" in order_by_section
    # trust_score must NOT appear in the ORDER BY clause for date sort
    assert "trust_score" not in order_by_section


@pytest.mark.asyncio
async def test_fetch_user_portfolio_returns_filtered_results():
    """Results returned by the session are forwarded as-is (filter is DB-side)."""
    asset_ms, item_ms = _make_asset_and_item(AssetStatus.KEEP)
    item_ms.category = CategoryType.MAINSTREAM
    # Simulate DB already returning only ALPHA items (the DB does the filtering)
    asset_al, item_al = _make_asset_and_item(AssetStatus.KEEP)
    item_al.category = CategoryType.ALPHA

    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.all.return_value = [(asset_al, item_al)]
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await fetch_user_portfolio(mock_session, category=CategoryType.ALPHA)
    assert len(result) == 1
    assert result[0][1].category == CategoryType.ALPHA


# ---------------------------------------------------------------------------
# _cosine_sim
# ---------------------------------------------------------------------------


def test_cosine_sim_identical_vectors():
    from argos.slack.services.briefing_query import _cosine_sim
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(_cosine_sim(v, v) - 1.0) < 1e-6


def test_cosine_sim_orthogonal_vectors():
    from argos.slack.services.briefing_query import _cosine_sim
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(_cosine_sim(a, b)) < 1e-6


def test_cosine_sim_zero_vector_returns_zero():
    from argos.slack.services.briefing_query import _cosine_sim
    a = np.array([1.0, 0.0], dtype=np.float32)
    zero = np.zeros(2, dtype=np.float32)
    assert _cosine_sim(a, zero) == 0.0


# ---------------------------------------------------------------------------
# _kmeans
# ---------------------------------------------------------------------------

def test_kmeans_k1_returns_centroid():
    from argos.slack.services.briefing_query import _kmeans
    vecs = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    centroids = _kmeans(vecs, k=1)
    assert len(centroids) == 1
    np.testing.assert_allclose(centroids[0], [1.0, 0.0], atol=1e-5)


def test_kmeans_k2_separates_clusters():
    from argos.slack.services.briefing_query import _kmeans
    cluster_a = [np.array([10.0, 0.0]) + np.array([0.0, 0.0]) for _ in range(5)]  # deterministic
    cluster_b = [np.array([0.0, 10.0]) + np.array([0.0, 0.0]) for _ in range(5)]  # deterministic
    vecs = cluster_a + cluster_b
    centroids = _kmeans(vecs, k=2, seed=42)
    assert len(centroids) == 2
    centroid_coords = sorted([tuple(c.tolist()) for c in centroids])
    assert centroid_coords[0][0] < 1.0
    assert centroid_coords[1][1] < 1.0


# ---------------------------------------------------------------------------
# _keep_centroids k selection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_keep_centroids_returns_empty_when_no_keeps():
    from argos.slack.services.briefing_query import _keep_centroids

    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.all.return_value = []
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    result = await _keep_centroids(mock_session)
    assert result == []


@pytest.mark.asyncio
async def test_keep_centroids_k1_for_two_items():
    from argos.slack.services.briefing_query import _keep_centroids

    embs = [
        [1.0] + [0.0] * 767,
        [0.0, 1.0] + [0.0] * 766,
    ]
    rows = [(e,) for e in embs]

    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    centroids = await _keep_centroids(mock_session)
    assert len(centroids) == 1


@pytest.mark.asyncio
async def test_keep_centroids_k2_for_five_items():
    from argos.slack.services.briefing_query import _keep_centroids

    embs = [[float(i)] + [0.0] * 767 for i in range(5)]
    rows = [(e,) for e in embs]

    async def fake_execute(stmt):
        mock_result = MagicMock()
        mock_result.all.return_value = rows
        return mock_result

    mock_session = AsyncMock()
    mock_session.execute = fake_execute

    centroids = await _keep_centroids(mock_session)
    assert len(centroids) == 2


# ---------------------------------------------------------------------------
# _score_and_select
# ---------------------------------------------------------------------------

def _make_item_with_embedding(source_url: str, trust_score: float, embedding: list[float]):
    item = MagicMock()
    item.source_url = source_url
    item.trust_score = trust_score
    item.embedding = embedding
    return item


def test_score_and_select_no_data_uses_trust_score():
    from argos.slack.services.briefing_query import _score_and_select

    items = [
        _make_item_with_embedding("https://a.com/1", 0.9, [1.0] + [0.0] * 767),
        _make_item_with_embedding("https://b.com/2", 0.1, [0.0, 1.0] + [0.0] * 766),
    ]
    result = _score_and_select(items, topic_vec=None, centroids=[], limit=2)
    assert len(result) == 2
    assert result[0].trust_score == 0.9


def test_score_and_select_applies_domain_cap():
    from argos.slack.services.briefing_query import _score_and_select

    items = [
        _make_item_with_embedding("https://openai.com/1", 0.9, [1.0] + [0.0] * 767),
        _make_item_with_embedding("https://openai.com/2", 0.85, [1.0] + [0.0] * 767),
        _make_item_with_embedding("https://openai.com/3", 0.8, [1.0] + [0.0] * 767),
        _make_item_with_embedding("https://anthropic.com/1", 0.5, [0.0, 1.0] + [0.0] * 766),
    ]
    result = _score_and_select(items, topic_vec=None, centroids=[], limit=4)
    openai_count = sum(1 for r in result if "openai.com" in r.source_url)
    assert openai_count <= 2
    assert any("anthropic.com" in r.source_url for r in result)


def test_score_and_select_respects_limit():
    from argos.slack.services.briefing_query import _score_and_select

    items = [
        _make_item_with_embedding(f"https://site{i}.com/article", 1.0 - i * 0.1, [1.0] + [0.0] * 767)
        for i in range(10)
    ]
    result = _score_and_select(items, topic_vec=None, centroids=[], limit=3)
    assert len(result) == 3


def test_score_and_select_topic_boosts_matching_items():
    from argos.slack.services.briefing_query import _score_and_select

    topic_vec = np.array([1.0] + [0.0] * 767, dtype=np.float32)
    item_a = _make_item_with_embedding("https://a.com/1", 0.5, [1.0] + [0.0] * 767)
    item_b = _make_item_with_embedding("https://b.com/1", 0.6, [0.0] + [1.0] + [0.0] * 766)

    result = _score_and_select([item_a, item_b], topic_vec=topic_vec, centroids=[], limit=2)
    assert result[0].source_url == "https://a.com/1"


def test_score_and_select_item_without_embedding_uses_zero_scores():
    from argos.slack.services.briefing_query import _score_and_select

    topic_vec = np.array([1.0] + [0.0] * 767, dtype=np.float32)
    centroid = np.array([1.0] + [0.0] * 767, dtype=np.float32)

    item_no_emb = _make_item_with_embedding("https://a.com/1", 0.5, None)
    item_no_emb.embedding = None
    result = _score_and_select([item_no_emb], topic_vec=topic_vec, centroids=[centroid], limit=1)
    assert len(result) == 1  # item is still scored (with 0 topic/keep scores)


@pytest.mark.asyncio
async def test_embed_topics_returns_none_when_empty():
    from argos.slack.services.briefing_query import _embed_topics
    result = await _embed_topics([])
    assert result is None


@pytest.mark.asyncio
async def test_embed_topics_returns_none_on_ollama_failure(monkeypatch):
    import argos.brain.ollama_client as oc

    # Patch batch_embed inside the function's import namespace
    async def failing_embed(topics):
        raise RuntimeError("Ollama not available")

    monkeypatch.setattr(oc, "batch_embed", failing_embed)

    from argos.slack.services.briefing_query import _embed_topics
    result = await _embed_topics(["LLM", "transformer"])
    assert result is None
