"""사건 상세 출력 형식과 검증 — ARG-308.

사건 요약은 이제 한 단락이 아니라 **짧은 요약 + 사건 유형별 소제목 상세**다.
모델은 JSON 하나를 내고(Ollama ``format``에 :func:`response_schema`를 건다),
최종 판정은 여기 :func:`parse_event_detail`이 한다 — 스키마 강제는 모델·버전에
따라 새기 때문에 Python 쪽 검증이 기준이다.

소제목은 유형마다 **고정 목록**(:data:`HEADINGS`)에서만 나온다. 목록 밖·빈
본문·중복 소제목은 버리고, 남은 것을 표 순서대로 정렬한다. 소제목이 하나도
남지 않으면 실패(``None``)다 — 빈 상세 화면을 쓰느니 이전 값을 지킨다.

DB도 LLM도 모르는 순수 모듈이다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from argos.brain.titles import MAX_TITLE_CHARS

EVENT_TYPES: tuple[str, ...] = ("Launch", "Research", "Incident", "Update", "Adoption")
"""사건 성격. ``tech_events.event_type`` CHECK 제약과 같은 값(PascalCase)."""

HEADINGS: dict[str, tuple[str, ...]] = {
    "Launch": ("무엇이 새로운가", "주요 수치", "이전과 달라진 점", "쓸 수 있는 곳·시점"),
    "Research": ("방법", "결과", "한계"),
    "Incident": ("무슨 일이 있었나", "영향 범위", "대응"),
    "Update": ("무엇이 바뀌었나", "이전과 비교"),
    "Adoption": ("누가 도입했나", "무엇이 좋아졌나"),
}
"""유형별 고정 소제목 표. 순서가 화면 순서다."""

EVENT_TYPE_LABELS: dict[str, str] = {
    "Launch": "출시",
    "Research": "연구",
    "Incident": "사고·논란",
    "Update": "업데이트",
    "Adoption": "도입",
}
"""화면·프롬프트에 쓰는 사람용 이름."""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
"""줄 맨 앞의 헤딩 표시만 지운다. 문자 하나로 보면 ``C# 14``의 ``#``까지 먹는다."""

# 짝이 맞는 강조/코드 스팬만 벗긴다. 여는·닫는 기호가 공백이 아닌 글자에
# 붙어 있어야 해서, 짝 없이 떠 있는 기호(``2 * 3``, ``C#``)는 그대로 남는다.
# ``_``는 아예 건드리지 않는다 — ``snake_case``·``__init__``을 깨뜨리는 값이
# 잃는 것(밑줄 이탤릭 정리)보다 크다.
_BOLD_RE = re.compile(r"\*\*(\S(?:.*?\S)?)\*\*")
_ITALIC_RE = re.compile(r"\*(\S(?:[^*]*?\S)?)\*")
_CODE_RE = re.compile(r"`(\S(?:[^`]*?\S)?)`")


def clean_raw(raw: str) -> str:
    """``<think>`` 블록과 코드 펜스 줄을 걷어낸다."""
    return _FENCE_RE.sub("", _THINK_RE.sub("", raw)).strip()


def scrub(value: str) -> str:
    """남은 마크다운 껍데기를 벗기고 공백을 접는다 (ARG-274 리뷰 규칙 그대로).

    헤딩 표시는 줄 맨 앞에서만, 강조·코드는 **짝이 맞을 때만** 벗긴다 —
    ``C# 14``·``snake_case``가 깨지지 않게.
    """
    value = _HEADING_RE.sub("", value)
    value = _BOLD_RE.sub(r"\1", value)
    value = _ITALIC_RE.sub(r"\1", value)
    value = _CODE_RE.sub(r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


@dataclass(frozen=True)
class Section:
    heading: str
    body: str


@dataclass(frozen=True)
class EventDetail:
    """검증을 통과한 사건 상세. 모든 필드가 채워졌을 때만 만들어진다."""

    title: str
    summary: str
    event_type: str
    sections: tuple[Section, ...]

    def text(self) -> str:
        """이름 검증용 — 사람이 읽게 될 모든 출력 텍스트를 이어 붙인다."""
        parts = [self.title, self.summary]
        for section in self.sections:
            parts.extend((section.heading, section.body))
        return "\n".join(parts)

    def sections_json(self) -> list[dict[str, str]]:
        """``tech_events.sections`` JSONB에 쓰는 모양."""
        return [{"heading": s.heading, "body": s.body} for s in self.sections]


def response_schema() -> dict:
    """Ollama ``format``에 넘길 JSON schema.

    소제목은 유형별이 아니라 **합집합** enum이다 — 유형과 소제목의 짝 맞춤은
    JSON schema로 표현이 번거롭고 모델별 지원도 들쭉날쭉해서 Python 검증에 맡긴다.
    """
    all_headings = [h for headings in HEADINGS.values() for h in headings]
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "event_type": {"type": "string", "enum": list(EVENT_TYPES)},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string", "enum": all_headings},
                        "body": {"type": "string"},
                    },
                    "required": ["heading", "body"],
                },
            },
        },
        "required": ["title", "summary", "event_type", "sections"],
    }


def _load_object(raw: str) -> dict | None:
    cleaned = clean_raw(raw or "")
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _normalize_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    for event_type in EVENT_TYPES:
        if event_type.lower() == lowered:
            return event_type
    return None


def _text(value: object) -> str:
    return scrub(value) if isinstance(value, str) else ""


def parse_event_detail(raw: str) -> EventDetail | None:
    """모델 출력을 검증해 :class:`EventDetail`로 만든다. 못 쓰면 ``None``."""
    payload = _load_object(raw)
    if payload is None:
        return None
    title = _text(payload.get("title"))[:MAX_TITLE_CHARS]
    summary = _text(payload.get("summary"))
    event_type = _normalize_type(payload.get("event_type"))
    if not title or not summary or event_type is None:
        return None

    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list):
        return None
    allowed = HEADINGS[event_type]
    bodies: dict[str, str] = {}
    for item in raw_sections:
        if not isinstance(item, dict):
            continue
        heading = _text(item.get("heading"))
        body = _text(item.get("body"))
        if heading not in allowed or not body or heading in bodies:
            continue
        bodies[heading] = body
    if not bodies:
        return None
    sections = tuple(Section(h, bodies[h]) for h in allowed if h in bodies)
    return EventDetail(title=title, summary=summary, event_type=event_type, sections=sections)


_LATIN_RE = re.compile(r"[A-Za-z]")
_LOWER_SUFFIX_RE = re.compile(r"-[a-z]+$")
"""``German-language``·``AI-powered`` 같은 '이름-소문자 접미' 꼴 — 형용사지 고유명이 아니다."""

MAX_REQUIRED_NAMES = 3
"""영어 원문 그대로 남아야 하는 이름 수 상한. 많이 걸수록 재생성이 잦아진다."""


def required_names(
    doc_names: Sequence[Iterable[str]],
    evidence_text: str,
    *,
    titles: Sequence[str | None] = (),
    limit: int = MAX_REQUIRED_NAMES,
) -> list[str]:
    """결과에 영어 원문 그대로 들어 있어야 하는 회사·제품 이름.

    후보는 기사에서 뽑힌 이름 중 라틴 문자가 있고 2자 이상이며 **실제 프롬프트
    근거 텍스트와 적어도 한 기사 제목에 등장하는** 것. 등장 문서 수가 많은 순으로
    상위 ``limit``개. 대소문자만 다른 이름은 처음 철자 하나로.

    제목 조건은 실측(ARG-308, gemma4:26b)에서 나왔다: ``document_entities``에는
    ``German-language``·``PDFs`` 같은 일반어가 섞여 있어, 본문에만 나온 이름까지
    필수로 걸면 모델이 자연스럽게 풀어 쓴 일반어 때문에 재시도·실패가 난다.
    제품·회사 이름은 대체로 제목에 나온다. '이름-소문자 접미' 꼴도 뺀다.
    """
    evidence = evidence_text.lower()
    lowered_titles = [(title or "").lower() for title in titles]
    counts: dict[str, int] = {}
    spelling: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    for names in doc_names:
        seen_here: set[str] = set()
        for name in names:
            name = (name or "").strip()
            key = name.lower()
            if len(name) < 2 or not _LATIN_RE.search(name) or key not in evidence:
                continue
            if _LOWER_SUFFIX_RE.search(name):
                continue
            if not any(key in title for title in lowered_titles):
                continue
            if key in seen_here:
                continue
            seen_here.add(key)
            spelling.setdefault(key, name)
            first_seen.setdefault(key, len(first_seen))
            counts[key] = counts.get(key, 0) + 1

    def rank(key: str) -> tuple[int, int]:
        return (-counts[key], first_seen[key])

    return [spelling[key] for key in sorted(counts, key=rank)[:limit]]


def missing_names(output_text: str, required: Sequence[str]) -> list[str]:
    """``required`` 중 출력에 (대소문자 무시) 그대로 없는 이름. 음역되면 여기 걸린다."""
    lowered = output_text.lower()
    return [name for name in required if name.lower() not in lowered]
