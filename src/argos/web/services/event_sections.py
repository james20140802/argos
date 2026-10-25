"""사건 상세 소제목(``tech_events.sections``)을 화면용으로 읽는다 — ARG-308.

``argos.brain``을 import하지 않는다 — 그 패키지는 파이프라인(LLM·그래프 라이브러리)을
통째로 끌어와서 웹 프로세스의 import 그래프를 무겁게 만든다. 그래서 유형 이름표를
여기 따로 두고, ``brain.event_detail_schema.EVENT_TYPE_LABELS``와 같은지는 테스트가
지킨다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

EVENT_TYPE_LABELS: dict[str, str] = {
    "Launch": "출시",
    "Research": "연구",
    "Incident": "사고·논란",
    "Update": "업데이트",
    "Adoption": "도입",
}


@dataclass(frozen=True)
class EventSection:
    heading: str
    body: str


def read_sections(value: object) -> tuple[EventSection, ...]:
    """JSONB 값을 소제목 목록으로. 모양이 틀리거나 빈 항목은 버린다(빈 소제목 없음)."""
    if not isinstance(value, list):
        return ()
    out: list[EventSection] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        heading, body = item.get("heading"), item.get("body")
        if isinstance(heading, str) and isinstance(body, str) and heading.strip() and body.strip():
            out.append(EventSection(heading.strip(), body.strip()))
    return tuple(out)


def type_label(event_type: Optional[str]) -> Optional[str]:
    return EVENT_TYPE_LABELS.get(event_type) if event_type else None
