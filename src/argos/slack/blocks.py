from __future__ import annotations

from datetime import date

from argos.models.tech_item import CategoryType, TechItem
from argos.models.user_asset import AssetStatus

_CATEGORY_LABELS: dict[CategoryType, str] = {
    CategoryType.MAINSTREAM: "Mainstream",
    CategoryType.ALPHA: "Alpha",
}

_ORDERED_CATEGORIES = (CategoryType.MAINSTREAM, CategoryType.ALPHA)

# Slack section blocks reject text longer than 3000 chars with `invalid_blocks`.
SLACK_SECTION_TEXT_LIMIT = 3000

ITEM_STATUS_BLOCK_ID = "argos_item_status"

_STATUS_LABELS: dict[AssetStatus, str] = {
    AssetStatus.KEEP: "✅ Keep — 포트폴리오에 추가됨",
    AssetStatus.TRACKING: "🔭 Tracking",
    AssetStatus.ARCHIVED: "🗄️ Archived — 패스됨",
}


def build_item_status_block(status: AssetStatus) -> dict:
    return {
        "type": "context",
        "block_id": ITEM_STATUS_BLOCK_ID,
        "elements": [{"type": "mrkdwn", "text": _STATUS_LABELS[status]}],
    }


def finalize_item_card_blocks(
    blocks: list[dict], status: AssetStatus
) -> list[dict]:
    """Drop interactive buttons and stamp the resolved status onto the card."""
    filtered = [
        b
        for b in blocks
        if b.get("type") != "actions"
        and b.get("block_id") != ITEM_STATUS_BLOCK_ID
    ]
    return [*filtered, build_item_status_block(status)]


def build_header_blocks(today: date, *, has_items: bool) -> list[dict]:
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"\U0001f6f0️ Argos Daily Briefing — {today.isoformat()}",
                "emoji": True,
            },
        }
    ]
    if not has_items:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "오늘 새로 수집된 기술 신호가 없습니다.",
                },
            }
        )
    return blocks


def build_category_header_blocks(
    category: CategoryType, *, has_items: bool = True
) -> list[dict]:
    label = _CATEGORY_LABELS[category]
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": label, "emoji": False},
        }
    ]
    if not has_items:
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "(no items today)"}],
            }
        )
    return blocks


def build_item_blocks(item: TechItem) -> list[dict]:
    score = f"{item.trust_score:.2f}" if item.trust_score is not None else "N/A"
    tech_id = str(item.id)
    summary = (item.summary or "").strip()
    header = f"*{item.title}* (trust={score})"

    # Title (up to 500), summary (up to 500), and URL (up to 2048) can together
    # exceed Slack's 3000-char section limit. Header and URL must stay intact
    # (URL drives link unfurl), so trim the summary to fit. If header+URL alone
    # already overflow — extremely rare but possible at column maxima — clamp
    # the entire body as a final guard.
    if summary:
        budget = SLACK_SECTION_TEXT_LIMIT - len(header) - len(item.source_url) - 2
        if budget <= 1:
            summary = ""
        elif budget < len(summary):
            summary = summary[: budget - 1].rstrip() + "…"

    body = f"{header}\n{summary}\n{item.source_url}" if summary else f"{header}\n{item.source_url}"
    if len(body) > SLACK_SECTION_TEXT_LIMIT:
        body = body[: SLACK_SECTION_TEXT_LIMIT - 1] + "…"
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": body,
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Keep", "emoji": False},
                    "action_id": "action_keep",
                    "value": tech_id,
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Pass", "emoji": False},
                    "action_id": "action_pass",
                    "value": tech_id,
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Deep Dive",
                        "emoji": False,
                    },
                    "action_id": "action_deep_dive",
                    "value": tech_id,
                    "style": "primary",
                },
            ],
        },
    ]


def build_briefing_blocks(
    items_by_category: dict[CategoryType, list[TechItem]],
    *,
    today: date,
) -> list[dict]:
    has_any = any(items_by_category.get(cat) for cat in _ORDERED_CATEGORIES)
    blocks: list[dict] = list(build_header_blocks(today, has_items=has_any))
    if not has_any:
        return blocks
    for category in _ORDERED_CATEGORIES:
        items = items_by_category.get(category) or []
        blocks.extend(build_category_header_blocks(category, has_items=bool(items)))
        for item in items:
            blocks.extend(build_item_blocks(item))
            blocks.append({"type": "divider"})
    return blocks
