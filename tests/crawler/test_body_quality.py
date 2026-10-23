from __future__ import annotations

from argos.crawler.body_quality import (
    BODY_MIN_CHARS,
    body_text,
    has_body,
    is_mojibake,
)


def test_min_chars_constant():
    assert BODY_MIN_CHARS == 500


def test_body_text_strips_title_prefix():
    raw = "My Title\n\nBody goes here."
    assert body_text(raw, "My Title") == "Body goes here."


def test_body_text_without_title_drops_leading_title_paragraph():
    assert body_text("My Title\n\nBody goes here.") == "Body goes here."


def test_body_text_keeps_body_only_content():
    # add_url 형태: 제목 접두 없이 본문만 저장된다.
    raw = "First sentence of a long article. " * 3
    assert body_text(raw, "Other") == raw.strip()


def test_body_text_strips_html_tags_and_collapses_whitespace():
    raw = "T\n\n<p>Hello   <b>world</b></p>\n<div>again</div>"
    assert body_text(raw, "T") == "Hello world again"


def test_body_text_keeps_generic_type_syntax():
    assert "Vec<T>" in body_text("T\n\nUse Vec<T> here", "T")


def test_body_text_handles_none_and_empty():
    assert body_text(None, "x") == ""
    assert body_text("", None) == ""


def test_has_body_boundary():
    assert not has_body("T\n\n" + "a" * 499, "T")
    assert has_body("T\n\n" + "a" * 500, "T")


def test_has_body_counts_after_html_stripping():
    raw = "T\n\n<p>" + "a" * 100 + "</p>" + "<span class='x'></span>" * 50
    assert not has_body(raw, "T")


def test_is_mojibake_detects_c1_and_latin1_artifacts():
    assert is_mojibake("weâ\x80\x99re")
    assert is_mojibake("weâ€re")
    assert is_mojibake("cafÃ© au lait")
    assert is_mojibake("a \x85 b")


def test_is_mojibake_ignores_legit_accents():
    assert not is_mojibake("café naïve résumé Ångström 한글 日本語 ’“”")
    assert not is_mojibake("")
