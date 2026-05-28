"""Shared HTML sanitisation utilities for crawler fetchers (ARG-129).

Public API
----------
- ``clean_title(text: str | None) -> str``
    Strip HTML tags and decode HTML entities from a raw title string,
    returning a normalised plain-text value safe for DB storage and Slack
    rendering.
"""
from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

_WHITESPACE_RE = re.compile(r"\s+")


def clean_title(text: str | None) -> str:
    """Return *text* as a plain-text title (tags stripped, entities decoded).

    Steps
    -----
    1. Return ``""`` for ``None`` or empty input.
    2. Use ``BeautifulSoup.get_text()`` with ``html.parser`` to strip all HTML
       markup while preserving inner text content.
    3. Call ``html.unescape()`` to decode any remaining character references
       (e.g. ``&amp;`` → ``&``, ``&#x2F;`` → ``/``).
    4. Collapse runs of whitespace and strip leading/trailing space.

    This function is intentionally idempotent: calling it on already-clean
    plain text returns the same string unchanged.
    """
    if not text:
        return ""
    # Strip HTML tags via BeautifulSoup (handles malformed/nested HTML)
    stripped = BeautifulSoup(text, "html.parser").get_text()
    # Decode remaining HTML entities
    decoded = html.unescape(stripped)
    # Normalise whitespace
    return _WHITESPACE_RE.sub(" ", decoded).strip()
