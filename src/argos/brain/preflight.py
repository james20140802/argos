"""Heuristic pre-LLM filter that rejects obviously non-tech content.

Enabled when settings.user.triage.preflight_filter is True (the default).
Only patterns with very low false-positive risk are included — when in
doubt, let triage handle it.
"""
from __future__ import annotations

import re

_JOB_AD_PATTERNS = [
    re.compile(r"\b(we[''']re hiring|now hiring|apply now|job opening)\b", re.I),
    re.compile(r"\b(years? of experience|bs/ms degree|equal opportunity employer)\b", re.I),
    re.compile(r"\b(salary range|competitive compensation|benefits package)\b", re.I),
    re.compile(r"\b(send (your )?resume|send (your )?cv|cover letter)\b", re.I),
]

_MARKETING_PATTERNS = [
    re.compile(r"\b(limited time offer|act now|buy (one )?get one|discount code)\b", re.I),
    re.compile(r"\b(sign up (for )?free|free trial|no credit card required)\b", re.I),
    re.compile(r"\b(satisfaction guaranteed|money[- ]back guarantee)\b", re.I),
]

_ALL_PATTERNS = _JOB_AD_PATTERNS + _MARKETING_PATTERNS


def is_preflight_reject(text: str) -> bool:
    """Return True if text should be discarded before any LLM call."""
    if not text:
        return False
    sample = text[:2000]
    return any(pat.search(sample) for pat in _ALL_PATTERNS)
