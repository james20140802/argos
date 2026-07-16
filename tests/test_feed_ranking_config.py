"""ARG-212: FeedRankingConfig defaults + override tests."""
from __future__ import annotations


def test_feed_ranking_config_defaults_and_override():
    from argos.config import UserConfig

    cfg = UserConfig().feed_ranking
    assert cfg.weight_recency == 0.35
    assert cfg.weight_profile == 0.35
    assert cfg.weight_trust == 0.15
    assert cfg.weight_trending == 0.15
    assert cfg.recency_half_life_hours == 48.0

    overridden = UserConfig.model_validate(
        {"feed_ranking": {"weight_recency": 0.5, "interest_bonus": 0.1}}
    )
    assert overridden.feed_ranking.weight_recency == 0.5
    assert overridden.feed_ranking.interest_bonus == 0.1
