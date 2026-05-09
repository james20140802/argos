from __future__ import annotations

import uuid
from datetime import date
from types import SimpleNamespace

from argos.models.tech_item import CategoryType
from argos.slack.blocks import (
    SLACK_SECTION_TEXT_LIMIT,
    build_briefing_blocks,
    build_category_header_blocks,
    build_header_blocks,
    build_item_blocks,
)


def _make_item(
    title: str,
    trust_score: float | None,
    tech_id: uuid.UUID | None = None,
    *,
    summary: str | None = None,
):
    item = SimpleNamespace(
        id=tech_id or uuid.uuid4(),
        title=title,
        source_url=f"https://example.com/{title}",
        trust_score=trust_score,
        summary=summary,
    )
    return item


TODAY = date(2026, 5, 4)


def test_empty_input_returns_header_and_no_items_section():
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [], CategoryType.ALPHA: []},
        today=TODAY,
    )
    assert blocks[0]["type"] == "header"
    assert "2026-05-04" in blocks[0]["text"]["text"]
    types = [b["type"] for b in blocks]
    assert "section" in types or "context" in types
    all_text = str(blocks)
    assert "오늘 새로 수집된 기술 신호가 없습니다." in all_text


def test_all_blocks_have_type_key():
    item = _make_item("TestTech", 0.9)
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [item], CategoryType.ALPHA: []},
        today=TODAY,
    )
    for block in blocks:
        assert "type" in block, f"Block missing 'type': {block}"


def test_mainstream_only_produces_three_buttons(tech_id):
    item = _make_item("StreamTech", 0.8, tech_id)
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [item], CategoryType.ALPHA: []},
        today=TODAY,
    )
    actions_blocks = [b for b in blocks if b["type"] == "actions"]
    assert len(actions_blocks) == 1
    elements = actions_blocks[0]["elements"]
    assert len(elements) == 3
    action_ids = [e["action_id"] for e in elements]
    assert "action_keep" in action_ids
    assert "action_pass" in action_ids
    assert "action_deep_dive" in action_ids


def test_button_values_are_tech_id_string(tech_id):
    item = _make_item("TechX", 0.5, tech_id)
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [item], CategoryType.ALPHA: []},
        today=TODAY,
    )
    actions_blocks = [b for b in blocks if b["type"] == "actions"]
    for element in actions_blocks[0]["elements"]:
        assert element["value"] == str(tech_id)


def test_alpha_only_produces_buttons(tech_id):
    item = _make_item("AlphaTech", 0.3, tech_id)
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [], CategoryType.ALPHA: [item]},
        today=TODAY,
    )
    actions_blocks = [b for b in blocks if b["type"] == "actions"]
    assert len(actions_blocks) == 1


def test_both_categories_populated(tech_id, tech_id2):
    m_item = _make_item("MainTech", 0.9, tech_id)
    a_item = _make_item("AlphaTech", 0.4, tech_id2)
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [m_item], CategoryType.ALPHA: [a_item]},
        today=TODAY,
    )
    actions_blocks = [b for b in blocks if b["type"] == "actions"]
    assert len(actions_blocks) == 2
    values = {e["value"] for ab in actions_blocks for e in ab["elements"]}
    assert str(tech_id) in values
    assert str(tech_id2) in values


def test_none_trust_score_displays_na():
    item = _make_item("NoScore", None)
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [item], CategoryType.ALPHA: []},
        today=TODAY,
    )
    all_text = str(blocks)
    assert "N/A" in all_text


def test_empty_category_shows_no_items_today():
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [], CategoryType.ALPHA: [_make_item("X", 0.5)]},
        today=TODAY,
    )
    all_text = str(blocks)
    assert "no items today" in all_text


def test_keep_button_has_primary_style(tech_id):
    item = _make_item("T", 0.5, tech_id)
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [item], CategoryType.ALPHA: []},
        today=TODAY,
    )
    actions = [b for b in blocks if b["type"] == "actions"][0]
    keep_btn = next(e for e in actions["elements"] if e["action_id"] == "action_keep")
    assert keep_btn.get("style") == "primary"


def test_deep_dive_button_has_primary_style(tech_id):
    item = _make_item("T", 0.5, tech_id)
    blocks = build_briefing_blocks(
        {CategoryType.MAINSTREAM: [item], CategoryType.ALPHA: []},
        today=TODAY,
    )
    actions = [b for b in blocks if b["type"] == "actions"][0]
    dd_btn = next(e for e in actions["elements"] if e["action_id"] == "action_deep_dive")
    assert dd_btn.get("style") == "primary"


def test_header_blocks_includes_empty_state_when_no_items():
    blocks = build_header_blocks(TODAY, has_items=False)
    assert blocks[0]["type"] == "header"
    assert "오늘 새로 수집된 기술 신호가 없습니다." in str(blocks)


def test_header_blocks_omits_empty_state_when_items_exist():
    blocks = build_header_blocks(TODAY, has_items=True)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "header"


def test_category_header_blocks_basic():
    blocks = build_category_header_blocks(CategoryType.MAINSTREAM)
    assert blocks[0]["type"] == "header"
    assert "Mainstream" in blocks[0]["text"]["text"]


def test_category_header_blocks_no_items_appends_context():
    blocks = build_category_header_blocks(CategoryType.ALPHA, has_items=False)
    assert any(b["type"] == "context" for b in blocks)


def test_item_blocks_returns_section_and_actions(tech_id):
    item = _make_item("ItemX", 0.7, tech_id)
    blocks = build_item_blocks(item)
    assert [b["type"] for b in blocks] == ["section", "actions"]
    assert str(tech_id) in str(blocks)


def test_item_blocks_renders_url_for_unfurl(tech_id):
    item = _make_item("LinkItem", 0.5, tech_id)
    blocks = build_item_blocks(item)
    assert item.source_url in blocks[0]["text"]["text"]


def test_item_blocks_renders_summary_when_present(tech_id):
    item = _make_item("SumItem", 0.7, tech_id, summary="짧은 요약 한 줄.")
    blocks = build_item_blocks(item)
    text = blocks[0]["text"]["text"]
    assert "짧은 요약 한 줄." in text
    # Layout: title line, summary line, URL line — summary lives between them.
    title_idx = text.index("*SumItem*")
    summary_idx = text.index("짧은 요약 한 줄.")
    url_idx = text.index(item.source_url)
    assert title_idx < summary_idx < url_idx


def test_item_blocks_omits_summary_line_when_none(tech_id):
    item = _make_item("NoSumItem", 0.4, tech_id, summary=None)
    blocks = build_item_blocks(item)
    text = blocks[0]["text"]["text"]
    # Title and URL should be on adjacent lines with no extra blank line.
    assert text == f"*NoSumItem* (trust=0.40)\n{item.source_url}"


def test_item_blocks_treats_blank_summary_as_absent(tech_id):
    item = _make_item("BlankSum", 0.4, tech_id, summary="   \n  ")
    blocks = build_item_blocks(item)
    text = blocks[0]["text"]["text"]
    assert text == f"*BlankSum* (trust=0.40)\n{item.source_url}"


def test_item_blocks_caps_text_at_slack_section_limit(tech_id):
    # Worst case from column maxima: title=500, summary=500, source_url=2048.
    # Combined with formatting this exceeds Slack's 3000-char section limit.
    long_title = "T" * 500
    long_url = "https://example.com/" + ("p" * (2048 - len("https://example.com/")))
    long_summary = "S" * 500
    item = SimpleNamespace(
        id=tech_id,
        title=long_title,
        source_url=long_url,
        trust_score=0.5,
        summary=long_summary,
    )
    blocks = build_item_blocks(item)
    text = blocks[0]["text"]["text"]
    assert len(text) <= SLACK_SECTION_TEXT_LIMIT
    # URL must stay intact for unfurl.
    assert long_url in text
    # Title must stay intact too.
    assert f"*{long_title}*" in text


def test_item_blocks_drops_summary_when_header_and_url_already_overflow(tech_id):
    long_title = "T" * 500
    # URL that, with header and newlines, leaves no room for any summary.
    long_url = "https://example.com/" + ("p" * (SLACK_SECTION_TEXT_LIMIT - 540))
    item = SimpleNamespace(
        id=tech_id,
        title=long_title,
        source_url=long_url,
        trust_score=0.5,
        summary="some summary that should be dropped",
    )
    blocks = build_item_blocks(item)
    text = blocks[0]["text"]["text"]
    assert len(text) <= SLACK_SECTION_TEXT_LIMIT
    assert "some summary" not in text
    assert long_url in text
