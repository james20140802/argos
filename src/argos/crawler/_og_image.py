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

from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

__all__ = ["extract_og_image", "resolve_image", "favicon_for_domain", "ResolvedImage"]

_MIN_IMG_DIM = 100

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


@dataclass(frozen=True)
class ResolvedImage:
    """Result of the prioritized image fallback chain.

    ``favicon_only`` is True only when ``url`` is a domain-derived favicon
    (the lowest-priority fallback), signalling the web layer to render the
    gradient + favicon-chip treatment instead of a full cover image.
    """

    url: str | None
    favicon_only: bool


def _too_small(dim) -> bool:
    """True when a declared width/height attribute is below the icon threshold.

    Missing / unparseable dimensions are treated as "unknown, not too small"
    so images without explicit sizes are still eligible.
    """
    if dim is None:
        return False
    try:
        return int(str(dim).strip().lower().rstrip("px")) < _MIN_IMG_DIM
    except (ValueError, TypeError):
        return False


def _first_body_image(soup: BeautifulSoup, base_url: str) -> str | None:
    """First meaningful body <img>: skip data-URIs and icon/tracking-sized images."""
    for img in soup.find_all("img"):
        if not hasattr(img, "get"):
            continue
        src = img.get("src")
        if not isinstance(src, str):
            continue
        src = src.strip()
        if not src or src.lower().startswith("data:"):
            continue
        if _too_small(img.get("width")) or _too_small(img.get("height")):
            continue
        try:
            absolute = urljoin(base_url or "", src)
        except ValueError:
            continue
        validated = _validate(absolute)
        if validated:
            return validated
    return None


def favicon_for_domain(base_url: str) -> str | None:
    """Derive ``{scheme}://{netloc}/favicon.ico`` purely from ``base_url``.

    No network I/O — the browser loads the favicon at render time. Returns
    ``None`` when ``base_url`` has no valid http(s) scheme + netloc.
    """
    try:
        parts = urlsplit(base_url or "")
    except ValueError:
        return None
    if parts.scheme.lower() not in _ALLOWED_SCHEMES or not parts.netloc:
        return None
    return _validate(f"{parts.scheme.lower()}://{parts.netloc}/favicon.ico")


def resolve_image(html: str, base_url: str) -> ResolvedImage:
    """Resolve the highest-priority image: og → twitter → body img → favicon."""
    og = extract_og_image(html, base_url)
    if og:
        return ResolvedImage(url=og, favicon_only=False)

    if html:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            soup = None
        if soup is not None:
            body = _first_body_image(soup, base_url)
            if body:
                return ResolvedImage(url=body, favicon_only=False)

    fav = favicon_for_domain(base_url)
    if fav:
        return ResolvedImage(url=fav, favicon_only=True)

    return ResolvedImage(url=None, favicon_only=False)
