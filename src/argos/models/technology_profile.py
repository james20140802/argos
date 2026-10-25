from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from argos.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class TechVerdict(str, enum.Enum):
    """이 이름이 기술인가 — 자동 배정 배치(ARG-313)의 판정."""

    TECHNOLOGY = "Technology"
    NOT_TECHNOLOGY = "NotTechnology"
    # 문장이 모자라거나 LLM이 "모르겠다"고 한 경우. 다음 배치에서 문서가 늘면 다시 본다.
    INSUFFICIENT = "Insufficient"


class NotTechReason:
    """``not_tech_reason`` 값. 열은 문자열이다(새 사유가 enum 마이그레이션을 부르지 않게)."""

    COMMON = "Common"  # 코퍼스 전반에 흔한 이름(ai, openai)
    VENDOR = "Vendor"  # 회사 접두어 뿌리(openai, google)
    PLACE = "Place"  # 장소·국적(spaCy GPE/NORP 다수결 또는 LLM)
    PERSON = "Person"  # 사람(spaCy PERSON 다수결 또는 LLM)
    GENERIC = "Generic"  # 일반 낱말(소문자로 주로 쓰임, 축약형 I'm)
    LLM = "Llm"  # LLM이 기술이 아니라고 함


class TechnologyProfile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """이름 하나(대표 이름)가 무엇인가 — ARG-313 자동 배정의 설명 층.

    ``entities.kind``는 건드리지 않는다. 판정·설명은 이 테이블에만 두고, 배치가
    문서가 충분히 늘었을 때 다시 써서 기사 하나의 틀린 설명이 굳지 않게 한다
    (``described_doc_count``가 그 기준).
    """

    __tablename__ = "technology_profiles"

    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    verdict: Mapped[TechVerdict] = mapped_column(
        Enum(
            TechVerdict,
            name="tech_verdict",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )
    not_tech_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)
    kind: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # 한국어 한 줄 설명(화면·why 용).
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 영어 역할 한 줄(배정 프롬프트 입력 — 시험판 D 방식과 같은 입력).
    role_en: Mapped[str | None] = mapped_column(Text, nullable=True)
    described_doc_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    described_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    def __repr__(self) -> str:
        return f"<TechnologyProfile(entity_id={self.entity_id}, verdict={self.verdict})>"
