"""Tests for brain.preflight heuristic filter."""
from __future__ import annotations

import pytest

from argos.brain.preflight import is_preflight_reject


@pytest.mark.parametrize(
    "text",
    [
        "We're hiring a Senior ML Engineer. Apply now with 5 years of experience.",
        "Now hiring! Send your resume to careers@example.com.",
        "Bachelor's (BS/MS Degree) required. We are an Equal Opportunity Employer.",
        "Salary range $120k-$160k. Competitive compensation and benefits package.",
        "Send your CV to jobs@company.io. Cover letter required.",
        "Limited time offer — 50% off! Act now!",
        "Sign up for free — no credit card required.",
        "30-day money-back guarantee. Satisfaction guaranteed.",
        "Buy one get one free with discount code SAVE50.",
    ],
)
def test_reject_known_patterns(text: str) -> None:
    assert is_preflight_reject(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "PyTorch 2.4 introduces torch.compile with improved backend support for M-series chips.",
        "Ollama now supports batch inference via the /api/embed endpoint.",
        "This paper proposes a new attention mechanism that reduces KV-cache memory by 40%.",
        "",
        "   ",
    ],
)
def test_allow_tech_content(text: str) -> None:
    assert is_preflight_reject(text) is False


def test_only_samples_first_2000_chars() -> None:
    # Pattern appears only past the 2000-char mark — should not reject.
    padding = "A" * 2001
    trigger = "We're hiring"
    assert is_preflight_reject(padding + trigger) is False

    # Pattern within first 2000 chars — should reject.
    short_prefix = "A" * 10
    assert is_preflight_reject(short_prefix + " " + trigger) is True
