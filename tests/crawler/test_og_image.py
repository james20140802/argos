"""Unit tests for the shared og:image extraction helper (ARG-149)."""

from __future__ import annotations

import pytest

from argos.crawler._og_image import (
    ResolvedImage,
    extract_og_image,
    favicon_for_domain,
    resolve_image,
)


BASE = "https://example.com/article"


def test_picks_og_image_absolute_url() -> None:
    html = (
        '<html><head>'
        '<meta property="og:image" content="https://cdn.example.com/cover.jpg">'
        '</head><body></body></html>'
    )
    assert extract_og_image(html, BASE) == "https://cdn.example.com/cover.jpg"


def test_prefers_og_image_over_twitter_image() -> None:
    html = (
        '<html><head>'
        '<meta property="og:image" content="https://cdn.example.com/og.jpg">'
        '<meta name="twitter:image" content="https://cdn.example.com/tw.jpg">'
        '</head></html>'
    )
    assert extract_og_image(html, BASE) == "https://cdn.example.com/og.jpg"


def test_falls_back_to_twitter_image_when_og_absent() -> None:
    html = (
        '<html><head>'
        '<meta name="twitter:image" content="https://cdn.example.com/tw.jpg">'
        '</head></html>'
    )
    assert extract_og_image(html, BASE) == "https://cdn.example.com/tw.jpg"


def test_falls_back_to_twitter_image_property_form() -> None:
    """Some sites use property= for twitter:image; accept both forms."""
    html = (
        '<html><head>'
        '<meta property="twitter:image" content="https://cdn.example.com/tw.jpg">'
        '</head></html>'
    )
    assert extract_og_image(html, BASE) == "https://cdn.example.com/tw.jpg"


def test_resolves_relative_url_against_base() -> None:
    html = (
        '<html><head>'
        '<meta property="og:image" content="/static/cover.jpg">'
        '</head></html>'
    )
    assert (
        extract_og_image(html, "https://example.com/blog/post")
        == "https://example.com/static/cover.jpg"
    )


def test_resolves_protocol_relative_url() -> None:
    html = (
        '<html><head>'
        '<meta property="og:image" content="//cdn.example.com/cover.jpg">'
        '</head></html>'
    )
    assert (
        extract_og_image(html, "https://example.com/post")
        == "https://cdn.example.com/cover.jpg"
    )


def test_returns_none_when_neither_present() -> None:
    html = "<html><head><title>nothing here</title></head></html>"
    assert extract_og_image(html, BASE) is None


def test_returns_none_for_empty_html() -> None:
    assert extract_og_image("", BASE) is None


def test_returns_none_for_empty_content() -> None:
    html = '<html><head><meta property="og:image" content=""></head></html>'
    assert extract_og_image(html, BASE) is None


def test_returns_none_for_whitespace_content() -> None:
    html = '<html><head><meta property="og:image" content="   "></head></html>'
    assert extract_og_image(html, BASE) is None


def test_rejects_data_uri() -> None:
    html = (
        '<html><head>'
        '<meta property="og:image" content="data:image/png;base64,iVBORw0KGgo=">'
        '</head></html>'
    )
    assert extract_og_image(html, BASE) is None


def test_rejects_file_scheme() -> None:
    html = (
        '<html><head>'
        '<meta property="og:image" content="file:///etc/passwd">'
        '</head></html>'
    )
    assert extract_og_image(html, BASE) is None


def test_rejects_javascript_scheme() -> None:
    html = (
        '<html><head>'
        '<meta property="og:image" content="javascript:alert(1)">'
        '</head></html>'
    )
    assert extract_og_image(html, BASE) is None


def test_rejects_url_longer_than_2048_chars() -> None:
    huge = "https://example.com/" + ("a" * 2050)
    html = f'<html><head><meta property="og:image" content="{huge}"></head></html>'
    assert extract_og_image(html, BASE) is None


def test_accepts_url_exactly_2048_chars() -> None:
    prefix = "https://example.com/"
    padding_len = 2048 - len(prefix)
    url = prefix + ("a" * padding_len)
    assert len(url) == 2048
    html = f'<html><head><meta property="og:image" content="{url}"></head></html>'
    assert extract_og_image(html, BASE) == url


def test_picks_first_og_image_when_multiple() -> None:
    html = (
        '<html><head>'
        '<meta property="og:image" content="https://cdn.example.com/first.jpg">'
        '<meta property="og:image" content="https://cdn.example.com/second.jpg">'
        '</head></html>'
    )
    assert extract_og_image(html, BASE) == "https://cdn.example.com/first.jpg"


def test_handles_malformed_html_gracefully() -> None:
    """Malformed HTML should not raise; helper returns None or best effort."""
    html = '<html><meta property="og:image" content="https://example.com/x.jpg"'
    # Either None or the URL is acceptable; the helper must not raise.
    result = extract_og_image(html, BASE)
    assert result is None or result == "https://example.com/x.jpg"


@pytest.mark.parametrize("base_url", ["", "not a url", "ftp://example.com/"])
def test_invalid_base_url_does_not_raise(base_url: str) -> None:
    """A malformed base_url must not crash the helper."""
    html = '<html><head><meta property="og:image" content="/cover.jpg"></head></html>'
    # Result may be None (can't resolve) or some value, but no exception.
    extract_og_image(html, base_url)


def test_favicon_for_domain_pure_convention() -> None:
    assert favicon_for_domain("https://example.com/a/b?x=1") == "https://example.com/favicon.ico"
    assert favicon_for_domain("http://sub.example.org/post") == "http://sub.example.org/favicon.ico"


def test_favicon_for_domain_rejects_non_http() -> None:
    assert favicon_for_domain("ftp://example.com/x") is None
    assert favicon_for_domain("not a url") is None
    assert favicon_for_domain("") is None


def test_resolve_prefers_og_then_twitter() -> None:
    html = (
        '<html><head>'
        '<meta property="og:image" content="https://cdn.example.com/og.jpg">'
        '<meta name="twitter:image" content="https://cdn.example.com/tw.jpg">'
        '</head><body><img src="https://cdn.example.com/body.jpg" width="600" height="400"></body></html>'
    )
    r = resolve_image(html, "https://example.com/a")
    assert r == ResolvedImage(url="https://cdn.example.com/og.jpg", favicon_only=False)


def test_resolve_falls_back_to_body_image() -> None:
    html = (
        '<html><head></head><body>'
        '<img src="/spacer.gif" width="1" height="1">'
        '<img src="https://cdn.example.com/hero.png" width="800" height="600">'
        '</body></html>'
    )
    r = resolve_image(html, "https://example.com/a")
    assert r == ResolvedImage(url="https://cdn.example.com/hero.png", favicon_only=False)


def test_resolve_body_image_resolves_relative_and_skips_data_uri() -> None:
    html = (
        '<html><body>'
        '<img src="data:image/gif;base64,AAAA">'
        '<img src="/img/cover.jpg">'
        '</body></html>'
    )
    r = resolve_image(html, "https://example.com/post/1")
    assert r == ResolvedImage(url="https://example.com/img/cover.jpg", favicon_only=False)


def test_resolve_falls_back_to_favicon_when_no_images() -> None:
    html = "<html><head><title>x</title></head><body><p>no images</p></body></html>"
    r = resolve_image(html, "https://example.com/article")
    assert r == ResolvedImage(url="https://example.com/favicon.ico", favicon_only=True)


def test_resolve_empty_html_still_yields_favicon_from_base_url() -> None:
    r = resolve_image("", "https://example.com/article")
    assert r == ResolvedImage(url="https://example.com/favicon.ico", favicon_only=True)


def test_resolve_no_url_when_base_url_unusable() -> None:
    r = resolve_image("<html></html>", "not-a-url")
    assert r == ResolvedImage(url=None, favicon_only=False)
