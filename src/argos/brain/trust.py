"""Deterministic trust-score synthesis (ARG-206).

Replaces the old single-shot LLM ``trust_score`` with a code-scored
composite: ``trust = weight_rubric*rubric + weight_prior*prior +
weight_corroboration*corroboration``.

The triage LLM only extracts a 5-field evidence rubric (temperature 0);
everything else here is pure, deterministic, and side-effect free so T2
(corroboration pipeline) and T3 (UI + backfill) can reuse it directly.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Rubric field -> point value when true (sum of the boolean weights below
# plus the enum tables' max entries is exactly 1.0).
_W_PRIMARY = 0.25  # is_primary_source True
_W_LINKS = 0.20  # has_evidence_links True
_W_NUMBERS = 0.20  # has_concrete_numbers True

_BALANCE_SCORE = {"balanced": 0.20, "mixed": 0.10, "unsupported": 0.0}
_MARKETING_SCORE = {"low": 0.15, "medium": 0.07, "high": 0.0}

# Source-tier seed -> prior score.
_TIER_SCORE = {"high": 1.0, "normal": 0.5, "low": 0.2}
_DEFAULT_TIER = "normal"

_CORROBORATION_CAP = 3


def score_rubric(rubric: dict) -> float:
    """Score a 5-field evidence rubric dict deterministically to 0..1.

    Missing keys or out-of-range enum values fall back conservatively to 0
    (i.e. the worst-case contribution for that field).
    """
    score = 0.0
    if rubric.get("is_primary_source") is True:
        score += _W_PRIMARY
    if rubric.get("has_evidence_links") is True:
        score += _W_LINKS
    if rubric.get("has_concrete_numbers") is True:
        score += _W_NUMBERS
    score += _BALANCE_SCORE.get(rubric.get("claim_evidence_balance"), 0.0)
    score += _MARKETING_SCORE.get(rubric.get("marketing_intensity"), 0.0)
    return max(0.0, min(1.0, score))


def source_prior(source_url: str, tiers: dict[str, str]) -> float:
    """Map a source URL's domain to a prior score via ``tiers``.

    The domain is taken from ``urlparse(source_url).netloc``, lower-cased,
    with a leading ``www.`` stripped. Unregistered domains default to the
    "normal" tier (0.5).
    """
    netloc = urlparse(source_url or "").netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[len("www."):]
    tier = tiers.get(netloc, _DEFAULT_TIER)
    return _TIER_SCORE.get(tier, _TIER_SCORE[_DEFAULT_TIER])


def corroboration_score(count: int) -> float:
    """Saturating corroboration score: min(count, 3) / 3."""
    return min(count, _CORROBORATION_CAP) / _CORROBORATION_CAP


def synthesize_trust(
    rubric_score: float,
    prior_score: float,
    corroboration_score: float,
    weights: dict,
) -> float:
    """Weighted synthesis of the three trust components, clamped to 0..1."""
    total = (
        weights.get("rubric", 0.0) * rubric_score
        + weights.get("prior", 0.0) * prior_score
        + weights.get("corroboration", 0.0) * corroboration_score
    )
    return max(0.0, min(1.0, total))
