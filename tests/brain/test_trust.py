import pytest
from argos.brain.trust import (
    score_rubric, source_prior, corroboration_score, synthesize_trust,
)

def test_score_rubric_all_max():
    rubric = {
        "is_primary_source": True, "has_evidence_links": True,
        "has_concrete_numbers": True, "claim_evidence_balance": "balanced",
        "marketing_intensity": "low",
    }
    assert score_rubric(rubric) == pytest.approx(1.0)

def test_score_rubric_all_min():
    rubric = {
        "is_primary_source": False, "has_evidence_links": False,
        "has_concrete_numbers": False, "claim_evidence_balance": "unsupported",
        "marketing_intensity": "high",
    }
    assert score_rubric(rubric) == pytest.approx(0.0)

def test_score_rubric_missing_fields_fallback_zero():
    assert score_rubric({}) == pytest.approx(0.0)

def test_source_prior_registered_high():
    assert source_prior("https://arxiv.org/abs/1234", {"arxiv.org": "high"}) == pytest.approx(1.0)

def test_source_prior_unregistered_is_normal():
    assert source_prior("https://example.com/x", {"arxiv.org": "high"}) == pytest.approx(0.5)

def test_source_prior_strips_www():
    assert source_prior("https://www.github.com/a/b", {"github.com": "high"}) == pytest.approx(1.0)

@pytest.mark.parametrize("count,expected", [(0, 0.0), (1, 1/3), (3, 1.0), (5, 1.0)])
def test_corroboration_score_saturates(count, expected):
    assert corroboration_score(count) == pytest.approx(expected)

def test_synthesize_default_weights():
    # rubric=1.0, prior=0.5, corr=0.0 → 0.6*1 + 0.2*0.5 + 0.2*0 = 0.7
    weights = {"rubric": 0.6, "prior": 0.2, "corroboration": 0.2}
    assert synthesize_trust(1.0, 0.5, 0.0, weights) == pytest.approx(0.7)

def test_synthesize_clamps_to_unit():
    weights = {"rubric": 2.0, "prior": 2.0, "corroboration": 2.0}
    assert synthesize_trust(1.0, 1.0, 1.0, weights) == pytest.approx(1.0)


def test_trust_config_defaults():
    from argos.config import TrustConfig
    c = TrustConfig()
    assert c.source_tiers["arxiv.org"] == "high"
    assert (c.weight_rubric, c.weight_prior, c.weight_corroboration) == (0.6, 0.2, 0.2)
    assert c.corroboration_threshold == 0.85
