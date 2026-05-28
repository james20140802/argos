"""Unit tests for src/argos/crawler/_html_utils.py (ARG-129)."""
from __future__ import annotations

import pytest

from argos.crawler._html_utils import clean_title


def test_strips_html_tags() -> None:
    assert clean_title("<p>Hello</p>") == "Hello"


def test_decodes_html_entities_hex() -> None:
    assert clean_title("a &#x2F; b") == "a / b"


def test_decodes_amp_entity() -> None:
    assert clean_title("AT&amp;T") == "AT&T"


def test_strips_tags_and_decodes_entities() -> None:
    assert clean_title("<i>foo</i> &amp; bar") == "foo & bar"


def test_handles_plain_text_unchanged() -> None:
    assert clean_title("Plain text") == "Plain text"


def test_handles_none_input() -> None:
    assert clean_title(None) == ""


def test_handles_empty_string() -> None:
    assert clean_title("") == ""


def test_handles_nested_tags() -> None:
    assert clean_title("<p><i>nested</i></p>") == "nested"


def test_handles_malformed_html() -> None:
    assert clean_title("<b>unclosed") == "unclosed"


def test_strips_anchor_tags_keeps_text() -> None:
    assert clean_title("<a href='x'>link text</a>") == "link text"


def test_strips_and_collapses_whitespace() -> None:
    assert clean_title("  hello   world  ") == "hello world"


def test_real_hn_example() -> None:
    """Regression: actual HN title that triggered ARG-129."""
    raw = (
        "DeepSeek reasonix, ... Related ongoing thread:"
        "<p><i>DeepSeek makes the V4 Pro price discount permanent</i>"
        " - <a href=\"https://news.ycombinator.com/item?id=48237663\">...</a>"
    )
    result = clean_title(raw)
    assert "<p>" not in result
    assert "<i>" not in result
    assert "<a " not in result
    assert "DeepSeek reasonix" in result
