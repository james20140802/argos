"""ARG-308: 사건 상세(요약 + 유형별 소제목) 출력 검증 — 순수 함수."""

from __future__ import annotations

import json

import pytest

from argos.brain.event_detail_schema import (
    EVENT_TYPES,
    HEADINGS,
    EventDetail,
    Section,
    parse_event_detail,
    response_schema,
)


def _raw(**overrides) -> str:
    payload = {
        "title": "Anthropic이 Claude Duet을 공개했다",
        "summary": "두 모델이 함께 답하는 기능이다. 오늘부터 쓸 수 있다.",
        "event_type": "Launch",
        "sections": [
            {"heading": "무엇이 새로운가", "body": "두 모델이 번갈아 답한다."},
            {"heading": "쓸 수 있는 곳·시점", "body": "웹에서 오늘부터."},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_heading_table_covers_every_event_type_with_the_issue_examples():
    assert EVENT_TYPES == ("Launch", "Research", "Incident", "Update", "Adoption")
    assert set(HEADINGS) == set(EVENT_TYPES)
    # 이슈의 성공 장면 예시
    for heading in ("무엇이 새로운가", "이전과 달라진 점", "쓸 수 있는 곳·시점"):
        assert heading in HEADINGS["Launch"]
    assert HEADINGS["Research"] == ("방법", "결과", "한계")
    assert HEADINGS["Adoption"] == ("누가 도입했나", "무엇이 좋아졌나")


def test_parses_a_valid_payload():
    detail = parse_event_detail(_raw())
    assert detail == EventDetail(
        title="Anthropic이 Claude Duet을 공개했다",
        summary="두 모델이 함께 답하는 기능이다. 오늘부터 쓸 수 있다.",
        event_type="Launch",
        sections=(
            Section("무엇이 새로운가", "두 모델이 번갈아 답한다."),
            Section("쓸 수 있는 곳·시점", "웹에서 오늘부터."),
        ),
    )


def test_strips_think_block_and_code_fence():
    raw = "<think>음</think>\n```json\n" + _raw() + "\n```"
    assert parse_event_detail(raw) is not None


def test_event_type_is_case_insensitive_and_normalized_to_pascal_case():
    detail = parse_event_detail(_raw(event_type="launch"))
    assert detail is not None and detail.event_type == "Launch"


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "",
        "[1, 2]",
        _raw(event_type="Rumor"),
        _raw(event_type=None),
        _raw(summary="  "),
        _raw(summary=None),
        _raw(title=""),
        _raw(sections=[]),
        _raw(sections="nope"),
        _raw(sections=[{"heading": "방법", "body": "다른 유형의 소제목"}]),
        _raw(sections=[{"heading": "무엇이 새로운가", "body": "   "}]),
    ],
    ids=[
        "garbage",
        "empty",
        "not-object",
        "unknown-type",
        "missing-type",
        "blank-summary",
        "missing-summary",
        "blank-title",
        "no-sections",
        "sections-not-list",
        "only-foreign-heading",
        "only-empty-body",
    ],
)
def test_invalid_payloads_return_none(raw):
    assert parse_event_detail(raw) is None


def test_drops_off_list_empty_and_duplicate_sections_and_orders_by_the_table():
    raw = _raw(
        sections=[
            {"heading": "쓸 수 있는 곳·시점", "body": "웹."},
            {"heading": "방법", "body": "연구 소제목은 출시에 못 온다."},
            {"heading": "무엇이 새로운가", "body": ""},
            {"heading": "이전과 달라진 점", "body": "빨라졌다."},
            {"heading": "쓸 수 있는 곳·시점", "body": "중복."},
            "not-a-dict",
            {"heading": "무엇이 새로운가", "body": "늦게 온 본문."},
        ]
    )
    detail = parse_event_detail(raw)
    assert detail is not None
    assert [s.heading for s in detail.sections] == [
        h for h in HEADINGS["Launch"] if h in {"무엇이 새로운가", "이전과 달라진 점", "쓸 수 있는 곳·시점"}
    ]
    bodies = {s.heading: s.body for s in detail.sections}
    assert bodies["쓸 수 있는 곳·시점"] == "웹."  # 먼저 온 것이 남는다
    assert bodies["무엇이 새로운가"] == "늦게 온 본문."  # 빈 본문은 없는 셈


def test_title_is_scrubbed_and_truncated():
    detail = parse_event_detail(_raw(title="**" + "가" * 600 + "**"))
    assert detail is not None
    assert len(detail.title) == 500
    assert "*" not in detail.title


def test_keeps_literal_punctuation_in_bodies():
    raw = _raw(sections=[{"heading": "무엇이 새로운가", "body": "C# 14와 snake_case 지원"}])
    detail = parse_event_detail(raw)
    assert detail is not None and detail.sections[0].body == "C# 14와 snake_case 지원"


def test_text_concatenates_every_output_field():
    detail = parse_event_detail(_raw())
    assert detail is not None
    text = detail.text()
    assert "Claude Duet" in text and "웹에서 오늘부터." in text and "두 모델이 함께" in text


def test_sections_as_json_round_trips():
    detail = parse_event_detail(_raw())
    assert detail is not None
    assert detail.sections_json() == [
        {"heading": "무엇이 새로운가", "body": "두 모델이 번갈아 답한다."},
        {"heading": "쓸 수 있는 곳·시점", "body": "웹에서 오늘부터."},
    ]


def test_response_schema_constrains_type_and_heading_vocabulary():
    schema = response_schema()
    props = schema["properties"]
    assert set(schema["required"]) == {"title", "summary", "event_type", "sections"}
    assert props["event_type"]["enum"] == list(EVENT_TYPES)
    heading_enum = props["sections"]["items"]["properties"]["heading"]["enum"]
    assert set(heading_enum) == {h for hs in HEADINGS.values() for h in hs}
    json.dumps(schema)  # Ollama payload로 그대로 직렬화돼야 한다


def test_sections_from_json_skips_malformed_rows():
    from argos.brain.event_detail_schema import sections_from_json

    assert sections_from_json(None) == ()
    assert sections_from_json("x") == ()
    assert sections_from_json(
        [{"heading": "방법", "body": "b"}, {"heading": "", "body": "b"}, 3, {"heading": "결과"}]
    ) == (Section("방법", "b"),)
