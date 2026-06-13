"""Unit tests for the shared og:image extraction helper (ARG-149)."""

from __future__ import annotations

import pytest

from argos.crawler._og_image import extract_og_image


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
