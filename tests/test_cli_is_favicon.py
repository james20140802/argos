"""Pure unit tests for ``argos.cli._is_favicon`` (no DB required).

The ``--upgrade-favicons`` path and the templates both classify a cover as a
favicon by the bare ``/favicon.ico`` path. ``_is_favicon`` must agree with that
convention even when the URL carries a cache-busting query string, otherwise an
``og:image`` of ``/favicon.ico?v=2`` would be persisted as a "real" cover and
then stretched across the card.
"""
from __future__ import annotations

import pytest

from argos.cli import _is_favicon


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/favicon.ico",
        "http://example.com/favicon.ico",
        "https://example.com/favicon.ico?v=2",
        "https://example.com/favicon.ico?cachebust=123&x=y",
        "/favicon.ico",
        "/favicon.ico?v=2",
    ],
)
def test_is_favicon_true_for_bare_and_query_string_favicons(url):
    assert _is_favicon(url) is True


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://cdn.example.com/og-card.png",
        "https://example.com/assets/favicon.ico.png",
        "https://example.com/my-favicon.ico-preview",
        "https://example.com/path/to/hero.jpg?favicon.ico",
    ],
)
def test_is_favicon_false_for_real_images(url):
    assert _is_favicon(url) is False
