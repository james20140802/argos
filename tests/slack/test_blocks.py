from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from argos.models.tech_item import CategoryType
from argos.slack.blocks import (
    PORTFOLIO_MAX_ITEMS,
    SLACK_CONFIRM_TEXT_LIMIT,
    SLACK_MAX_BLOCKS,
    SLACK_SECTION_TEXT_LIMIT,
    build_briefing_blocks,
    build_category_header_blocks,
    build_header_blocks,
    build_item_blocks,
    build_portfolio_blocks,
    build_portfolio_empty_blocks,
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


# ---------------------------------------------------------------------------
# Portfolio block tests
# ---------------------------------------------------------------------------


def _make_portfolio_pair(tech_id: uuid.UUID, *, has_monitored_at: bool = True):
    """Create a (UserAsset, TechItem) mock pair for portfolio block tests."""
    item = SimpleNamespace(
        id=tech_id,
        title="PortfolioTech",
        source_url="https://example.com/portfolio-tech",
    )

    asset = MagicMock()
    asset.updated_at = datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone.utc)
    asset.last_monitored_at = (
        datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc) if has_monitored_at else None
    )
    return asset, item


def test_portfolio_empty_blocks_returns_header_and_empty_message():
    blocks = build_portfolio_empty_blocks()
    assert blocks[0]["type"] == "header"
    all_text = str(blocks)
    assert "포트폴리오" in all_text
    assert "Keep한 기술이 없습니다" in all_text


def test_portfolio_blocks_header_contains_count(tech_id):
    asset, item = _make_portfolio_pair(tech_id)
    blocks = build_portfolio_blocks([(asset, item)])
    assert blocks[0]["type"] == "header"
    assert "1" in blocks[0]["text"]["text"]


def test_portfolio_blocks_section_count_matches_assets(tech_id, tech_id2):
    pair1 = _make_portfolio_pair(tech_id)
    pair2 = _make_portfolio_pair(tech_id2)
    blocks = build_portfolio_blocks([pair1, pair2])
    sections = [b for b in blocks if b["type"] == "section"]
    assert len(sections) == 2


def test_portfolio_blocks_section_contains_title_link(tech_id):
    asset, item = _make_portfolio_pair(tech_id)
    blocks = build_portfolio_blocks([(asset, item)])
    section = next(b for b in blocks if b["type"] == "section")
    text = section["text"]["text"]
    assert item.source_url in text
    assert item.title in text


def test_portfolio_blocks_untrack_button_action_id_and_value(tech_id):
    asset, item = _make_portfolio_pair(tech_id)
    blocks = build_portfolio_blocks([(asset, item)])
    actions = [b for b in blocks if b["type"] == "actions"]
    assert len(actions) == 1
    btn = actions[0]["elements"][0]
    assert btn["action_id"] == "action_untrack"
    assert btn["value"] == str(tech_id)


def test_portfolio_blocks_last_signal_fallback_when_none(tech_id):
    asset, item = _make_portfolio_pair(tech_id, has_monitored_at=False)
    blocks = build_portfolio_blocks([(asset, item)])
    section = next(b for b in blocks if b["type"] == "section")
    assert "—" in section["text"]["text"]


def test_portfolio_blocks_text_does_not_exceed_slack_limit(tech_id):
    asset, item = _make_portfolio_pair(tech_id)
    # Use an extremely long title and URL to stress the limit
    item = SimpleNamespace(
        id=tech_id,
        title="T" * 400,
        source_url="https://example.com/" + "p" * 400,
    )
    blocks = build_portfolio_blocks([(asset, item)])
    for block in blocks:
        if block.get("type") == "section" and block.get("text"):
            assert len(block["text"]["text"]) <= SLACK_SECTION_TEXT_LIMIT


def test_portfolio_blocks_confirm_text_does_not_exceed_slack_limit(tech_id):
    asset, _ = _make_portfolio_pair(tech_id)
    # TechItem.title may be up to 500 chars; confirm text must still be safe.
    item = SimpleNamespace(
        id=tech_id,
        title="T" * 500,
        source_url="https://example.com/long",
    )
    blocks = build_portfolio_blocks([(asset, item)])
    actions = [b for b in blocks if b["type"] == "actions"]
    assert actions, "expected an actions block with Untrack confirm"
    confirm = actions[0]["elements"][0]["confirm"]
    assert len(confirm["text"]["text"]) <= SLACK_CONFIRM_TEXT_LIMIT


def _make_many_portfolio_pairs(count: int):
    pairs = []
    for _ in range(count):
        tid = uuid.uuid4()
        pairs.append(_make_portfolio_pair(tid))
    return pairs


def test_portfolio_blocks_respects_slack_50_block_cap():
    # 30 kept assets would naïvely produce 1 + 30*3 = 91 blocks, well over the
    # Slack 50-block limit. The renderer must clamp the payload.
    pairs = _make_many_portfolio_pairs(30)
    blocks = build_portfolio_blocks(pairs)
    assert len(blocks) <= SLACK_MAX_BLOCKS


def test_portfolio_blocks_renders_at_most_portfolio_max_items():
    pairs = _make_many_portfolio_pairs(PORTFOLIO_MAX_ITEMS + 5)
    blocks = build_portfolio_blocks(pairs)
    actions = [b for b in blocks if b["type"] == "actions"]
    assert len(actions) == PORTFOLIO_MAX_ITEMS


def test_portfolio_blocks_appends_truncation_notice_when_over_cap():
    over_by = 5
    pairs = _make_many_portfolio_pairs(PORTFOLIO_MAX_ITEMS + over_by)
    blocks = build_portfolio_blocks(pairs)
    # The last block should be a context block noting the hidden count.
    assert blocks[-1]["type"] == "context"
    notice_text = blocks[-1]["elements"][0]["text"]
    assert str(over_by) in notice_text


def test_portfolio_blocks_header_reports_full_count_when_truncated():
    over_by = 7
    pairs = _make_many_portfolio_pairs(PORTFOLIO_MAX_ITEMS + over_by)
    blocks = build_portfolio_blocks(pairs)
    header_text = blocks[0]["text"]["text"]
    assert str(PORTFOLIO_MAX_ITEMS + over_by) in header_text


def test_portfolio_blocks_no_truncation_notice_when_under_cap(tech_id):
    asset, item = _make_portfolio_pair(tech_id)
    blocks = build_portfolio_blocks([(asset, item)])
    assert all(b.get("type") != "context" for b in blocks)


# ---------------------------------------------------------------------------
# Weekly Keep summary blocks (ARG-123)
# ---------------------------------------------------------------------------


def _make_weekly_item(
    *,
    title: str,
    signals_7d: int = 0,
    successions_7d: int = 0,
    last_monitored_at: datetime | None = None,
):
    from argos.brain.weekly_report import WeeklyKeepItem

    return WeeklyKeepItem(
        tech_id=uuid.uuid4(),
        title=title,
        signals_7d=signals_7d,
        successions_7d=successions_7d,
        last_monitored_at=last_monitored_at,
    )


def _make_weekly_report(items, *, now_utc: datetime | None = None):
    from datetime import timedelta

    from argos.brain.weekly_report import WeeklyKeepReport

    end = now_utc or datetime(2026, 5, 20, 12, tzinfo=timezone.utc)
    return WeeklyKeepReport(
        total_keep_count=len(items),
        items=list(items),
        window_start=end - timedelta(days=7),
        window_end=end,
    )


def test_weekly_keep_summary_blocks_empty_portfolio_returns_placeholder():
    from argos.slack.blocks import build_weekly_keep_summary_blocks

    report = _make_weekly_report([])
    blocks = build_weekly_keep_summary_blocks(report)

    assert blocks[0]["type"] == "header"
    header_text = blocks[0]["text"]["text"]
    assert "Weekly Keep" in header_text
    # Window dates appear in the header (ISO YYYY-MM-DD).
    assert "2026-05-13" in header_text
    assert "2026-05-20" in header_text

    # A placeholder section explaining the empty state must be present.
    section_texts = [
        b.get("text", {}).get("text", "")
        for b in blocks
        if b.get("type") == "section"
    ]
    assert any("Keep" in t for t in section_texts)
    assert any("없" in t for t in section_texts)


def test_weekly_keep_summary_blocks_renders_each_item():
    from argos.slack.blocks import build_weekly_keep_summary_blocks

    monitored = datetime(2026, 5, 18, 10, tzinfo=timezone.utc)
    items = [
        _make_weekly_item(title="Tech A", signals_7d=3, successions_7d=1, last_monitored_at=monitored),
        _make_weekly_item(title="Tech B", signals_7d=0, successions_7d=0, last_monitored_at=None),
    ]
    report = _make_weekly_report(items)

    blocks = build_weekly_keep_summary_blocks(report)

    # Combined text from every block — easier to assert against than walking
    # the nested Block Kit dict structure.
    text_blob = "\n".join(
        b.get("text", {}).get("text", "")
        for b in blocks
        if b.get("type") == "section"
    )
    for element in blocks:
        if element.get("type") == "context":
            for sub in element.get("elements", []):
                text_blob += "\n" + sub.get("text", "")

    assert "Tech A" in text_blob
    assert "Tech B" in text_blob
    # signal count visible somewhere
    assert "3" in text_blob
    # total count visible (2 items)
    assert "2" in text_blob


def test_weekly_keep_summary_blocks_total_count_in_header_or_summary():
    from argos.slack.blocks import build_weekly_keep_summary_blocks

    items = [_make_weekly_item(title=f"T{i}") for i in range(5)]
    report = _make_weekly_report(items)

    blocks = build_weekly_keep_summary_blocks(report)
    all_text = " ".join(
        b.get("text", {}).get("text", "")
        for b in blocks
        if isinstance(b.get("text"), dict)
    )
    assert "5" in all_text


def test_weekly_keep_summary_blocks_escape_user_titles():
    from argos.slack.blocks import build_weekly_keep_summary_blocks

    # Slack mrkdwn special chars (<, >, &) must be escaped.
    item = _make_weekly_item(title="A & B <C>")
    report = _make_weekly_report([item])

    blocks = build_weekly_keep_summary_blocks(report)
    flat = " ".join(
        b.get("text", {}).get("text", "")
        for b in blocks
        if isinstance(b.get("text"), dict)
    )
    assert "&amp;" in flat
    assert "&lt;C&gt;" in flat
    assert "<C>" not in flat
    # The literal ampersand should NOT appear unescaped.
    assert " & " not in flat


def test_weekly_keep_summary_blocks_caps_at_slack_max():
    from argos.slack.blocks import SLACK_MAX_BLOCKS, build_weekly_keep_summary_blocks

    # 60 items would produce 62 blocks (header + summary + 60) without a cap.
    items = [_make_weekly_item(title=f"Tech {i}") for i in range(60)]
    report = _make_weekly_report(items)
    blocks = build_weekly_keep_summary_blocks(report)

    assert len(blocks) <= SLACK_MAX_BLOCKS
    # A truncation context block must appear when items are hidden.
    context_blocks = [b for b in blocks if b.get("type") == "context"]
    assert len(context_blocks) == 1
    assert "13" in context_blocks[0]["elements"][0]["text"]  # 60 - 47 = 13 hidden


def test_weekly_keep_summary_blocks_succession_marker_when_present():
    from argos.slack.blocks import build_weekly_keep_summary_blocks

    with_succession = _make_weekly_item(title="WithSucc", successions_7d=2)
    without_succession = _make_weekly_item(title="NoSucc", successions_7d=0)
    report = _make_weekly_report([with_succession, without_succession])

    blocks = build_weekly_keep_summary_blocks(report)
    section_texts = [
        b.get("text", {}).get("text", "")
        for b in blocks
        if b.get("type") == "section"
    ]
    # Find the section for WithSucc and assert it carries the succession indicator.
    with_text = next(t for t in section_texts if "WithSucc" in t)
    no_text = next(t for t in section_texts if "NoSucc" in t)
    assert "Succession" in with_text or "후속" in with_text or "⚠️" in with_text
    assert "0" in no_text or "—" in no_text or "없" in no_text
