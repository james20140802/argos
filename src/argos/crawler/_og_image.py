"""Shared og:image extraction helper for crawler fetchers (ARG-149).

Public API
----------
- ``extract_og_image(html: str, base_url: str) -> str | None``
    Pure synchronous helper. Parses HTML, picks the first
    ``<meta property="og:image">`` and falls back to ``twitter:image``,
    resolves relative URLs against ``base_url``, and returns a validated
    absolute http(s) URL no longer than 2048 characters — otherwise ``None``.

The helper is intentionally side-effect free: no network I/O, no logging.
Callers (static_fetcher, dynamic_fetcher, add_url) decide what to do with
a ``None`` result (typically: persist NULL on ``tech_items.image_url``).
"""
from __future__ import annotations

from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

__all__ = ["extract_og_image"]

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_URL_LEN = 2048


def _meta_content(soup: BeautifulSoup, attrs: dict) -> str | None:
    tag = soup.find("meta", attrs=attrs)
    if tag is None or not hasattr(tag, "get"):
        return None
    content = tag.get("content")
    if not isinstance(content, str):
        return None
    content = content.strip()
    return content or None


def _validate(absolute_url: str) -> str | None:
    if not absolute_url or len(absolute_url) > _MAX_URL_LEN:
        return None
    try:
        parts = urlsplit(absolute_url)
    except ValueError:
        return None
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        return None
    if not parts.netloc:
        return None
    return absolute_url


def extract_og_image(html: str, base_url: str) -> str | None:
    """Return the first valid og:image (or twitter:image fallback) URL.

    Parameters
    ----------
    html:
        Full HTML document text. Empty/None-equivalent input yields ``None``.
    base_url:
        URL used to resolve relative image references via :func:`urljoin`.

    Returns
    -------
    str | None
        Absolute http(s) URL no longer than 2048 characters, or ``None`` when
        no usable candidate is present.
    """
    if not html:
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return None

    # 1. og:image first (Open Graph spec).
    raw = _meta_content(soup, {"property": "og:image"})
    # 2. twitter:image fallback — accept both name= and property= attribute forms.
    if raw is None:
        raw = _meta_content(soup, {"name": "twitter:image"})
    if raw is None:
        raw = _meta_content(soup, {"property": "twitter:image"})
    if raw is None:
        return None

    try:
        absolute = urljoin(base_url or "", raw)
    except ValueError:
        return None

    return _validate(absolute)
