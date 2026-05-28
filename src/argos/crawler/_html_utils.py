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
# Matches HTML tags revealed after entity decoding (e.g. <i>, </b>, <!-- -->).
# Anchored to letter/slash/bang so "3 < 5 > 2" comparisons are not stripped.
_TAG_RE = re.compile(r"<[a-zA-Z/!][^>]*>")


def clean_title(text: str | None) -> str:
    """Return *text* as a plain-text title (tags stripped, entities decoded).

    Steps
    -----
    1. Return ``""`` for ``None`` or empty input.
    2. Use ``BeautifulSoup.get_text()`` with ``html.parser`` to strip actual
       HTML markup and decode entity references in text nodes
       (e.g. ``&amp;`` → ``&``).  Entity-encoded tags such as
       ``&lt;i&gt;Foo&lt;/i&gt;`` are decoded by the parser to the literal
       characters ``<i>Foo</i>`` inside the text node — they are NOT stripped
       at this step because the parser never saw them as tags.
    3. Call ``html.unescape()`` to decode any remaining character references.
    4. Strip any HTML tags now visible in the decoded text with a lightweight
       regex (avoids re-parsing through BeautifulSoup, which would garble raw
       ``&`` characters such as those in "AT&T").
    5. Collapse runs of whitespace and strip leading/trailing space.

    This function is intentionally idempotent: calling it on already-clean
    plain text returns the same string unchanged.
    """
    if not text:
        return ""
    # Strip actual HTML markup; html.parser also decodes entity refs in text
    # nodes, so &lt;i&gt; becomes the literal characters <i> in the output.
    stripped = BeautifulSoup(text, "html.parser").get_text()
    # Decode any remaining character references in the plain-text content.
    decoded = html.unescape(stripped)
    # Strip tags that became visible after entity decoding (second-pass strip).
    cleaned = _TAG_RE.sub("", decoded)
    # Normalise whitespace
    return _WHITESPACE_RE.sub(" ", cleaned).strip()
