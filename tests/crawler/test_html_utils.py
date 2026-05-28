"""Unit tests for src/argos/crawler/_html_utils.py (ARG-129)."""
from __future__ import annotations

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


def test_strips_escaped_html_tags() -> None:
    """Regression: escaped tags like &lt;i&gt; must be stripped, not left as <i>."""
    assert clean_title("&lt;i&gt;Foo&lt;/i&gt;") == "Foo"


def test_strips_escaped_tags_mixed_with_entities() -> None:
    assert clean_title("&lt;b&gt;Hello&lt;/b&gt; &amp; world") == "Hello & world"


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


def test_inserts_space_when_stripping_br_tag() -> None:
    """Regression: Hello<br>World must not collapse to HelloWorld."""
    assert clean_title("Hello<br>World") == "Hello World"


def test_inserts_space_between_adjacent_block_tags() -> None:
    """Regression: separator tags used for layout must not merge surrounding words."""
    assert clean_title("thread:<p>DeepSeek") == "thread: DeepSeek"


def test_preserves_generic_type_params_literal() -> None:
    """Regression: Vec<T> must not be corrupted by the HTML stripper."""
    assert clean_title("Understanding Vec<T> in Rust") == "Understanding Vec<T> in Rust"


def test_preserves_nested_generic_type_params_literal() -> None:
    assert clean_title("Promise<Result<T, Error>>") == "Promise<Result<T, Error>>"


def test_preserves_generic_type_params_escaped() -> None:
    """Regression: Vec&lt;T&gt; must decode entities but preserve <T> since T is not HTML."""
    assert clean_title("Vec&lt;T&gt;") == "Vec<T>"


def test_preserves_nested_generic_type_params_escaped() -> None:
    assert clean_title("Promise&lt;Result&lt;T&gt;&gt;") == "Promise<Result<T>>"


def test_strips_div_tags() -> None:
    """Regression: <div> was missing from allowlist so _has_html() returned False."""
    assert clean_title("<div>Foo</div>") == "Foo"


def test_strips_li_tags() -> None:
    """Regression: <li> was missing from allowlist so _has_html() returned False."""
    assert clean_title("<li>Foo</li>") == "Foo"


def test_strips_font_tags() -> None:
    """Regression: <font> was missing from allowlist so _has_html() returned False."""
    assert clean_title("<font>Foo</font>") == "Foo"
