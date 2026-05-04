from __future__ import annotations

from datetime import date

from argos.models.tech_item import CategoryType, TechItem

_CATEGORY_LABELS: dict[CategoryType, str] = {
    CategoryType.MAINSTREAM: "Mainstream",
    CategoryType.ALPHA: "Alpha",
}

_ORDERED_CATEGORIES = (CategoryType.MAINSTREAM, CategoryType.ALPHA)


def build_briefing_blocks(
    items_by_category: dict[CategoryType, list[TechItem]],
    *,
    today: date,
) -> list[dict]:
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

    all_empty = all(
        not items_by_category.get(cat) for cat in _ORDERED_CATEGORIES
    )
    if all_empty:
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

    for category in _ORDERED_CATEGORIES:
        label = _CATEGORY_LABELS[category]
        blocks.append(
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": label,
                    "emoji": False,
                },
            }
        )

        items = items_by_category.get(category) or []
        if not items:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": "(no items today)"}
                    ],
                }
            )
            continue

        for item in items:
            score = f"{item.trust_score:.2f}" if item.trust_score is not None else "N/A"
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{item.title}* (trust={score})\n{item.source_url}",
                    },
                }
            )
            tech_id = str(item.id)
            blocks.append(
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
                }
            )
            blocks.append({"type": "divider"})

    return blocks
