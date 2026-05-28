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

# Named and numeric HTML character references — their presence signals HTML content.
_HTML_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-z]\w+);", re.IGNORECASE)

# Opening/closing tags whose name is a known HTML element.  Used both to
# detect whether a title contains HTML markup and to strip tags revealed
# after entity decoding.  Generic type-parameter syntax such as Vec<T> or
# Promise<Result<T>> is excluded because "T", "Result", etc. are not in
# the element list, so the regex never matches them.
_HTML_TAG_RE = re.compile(
    r"</?(?:a|abbr|acronym|address|article|aside|"
    r"b|big|blockquote|br|button|"
    r"caption|center|cite|code|"
    r"dd|del|details|dfn|div|dl|dt|"
    r"em|"
    r"fieldset|figcaption|figure|font|footer|form|"
    r"h[1-6]|header|hr|"
    r"i|img|input|ins|"
    r"kbd|"
    r"label|legend|li|"
    r"main|mark|"
    r"nav|noscript|"
    r"ol|option|"
    r"p|pre|"
    r"q|"
    r"s|samp|script|section|select|small|span|strike|strong|style|sub|summary|sup|"
    r"table|tbody|td|textarea|tfoot|th|thead|tr|tt|"
    r"u|ul|"
    r"var|wbr)\b[^>]*>",
    re.IGNORECASE,
)


def _has_html(text: str) -> bool:
    """Return True if *text* contains HTML entities or known HTML element tags."""
    return bool(_HTML_ENTITY_RE.search(text) or _HTML_TAG_RE.search(text))


def clean_title(text: str | None) -> str:
    """Return *text* as a plain-text title (tags stripped, entities decoded).

    Steps
    -----
    1. Return ``""`` for ``None`` or empty input.
    2. If no HTML markup is detected (no entity references, no known HTML
       element tags), normalise whitespace and return as-is.  This preserves
       programming syntax such as ``Vec<T>`` or ``Promise<Result<T>>`` that
       would otherwise be mangled by an HTML parser.
    3. Use ``BeautifulSoup.get_text()`` with ``html.parser`` to strip actual
       HTML markup and decode entity references in text nodes
       (e.g. ``&amp;`` → ``&``).  Entity-encoded tags such as
       ``&lt;i&gt;Foo&lt;/i&gt;`` are decoded by the parser to the literal
       characters ``<i>Foo</i>`` inside the text node — they are NOT stripped
       at this step because the parser never saw them as tags.
    4. Call ``html.unescape()`` to decode any remaining character references.
    5. Strip any known-HTML-element tags now visible in the decoded text.
       Using the same element allowlist as the presence check ensures that
       ``Vec&lt;T&gt;`` decoded to ``Vec<T>`` is not corrupted because ``T``
       is not a known HTML element.
    6. Collapse runs of whitespace and strip leading/trailing space.

    This function is intentionally idempotent: calling it on already-clean
    plain text returns the same string unchanged.
    """
    if not text:
        return ""
    if not _has_html(text):
        return _WHITESPACE_RE.sub(" ", text).strip()
    # Strip actual HTML markup; html.parser also decodes entity refs in text
    # nodes, so &lt;i&gt; becomes the literal characters <i> in the output.
    # separator=" " prevents adjacent tags from merging their surrounding text
    # (e.g. Hello<br>World → "Hello World", not "HelloWorld").
    stripped = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    # Decode any remaining character references in the plain-text content.
    decoded = html.unescape(stripped)
    # Strip known-HTML-element tags revealed after entity decoding.
    cleaned = _HTML_TAG_RE.sub("", decoded)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()
