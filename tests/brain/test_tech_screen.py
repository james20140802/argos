"""일반어·장소·회사 이름 거르기 (ARG-313) — LLM 없음, DB 불필요."""
from __future__ import annotations

import pytest

from argos.brain.tech_classify import ScreenSignals, majority_ner_label, screen
from argos.config import TaxonomyConfig
from argos.models.technology_profile import NotTechReason

CFG = TaxonomyConfig()


def sig(display, **kw):
    base = dict(
        key=display.lower(),
        display=display,
        common=False,
        lowercase_ratio=0.0,
        is_vendor_root=False,
        ner_labels={},
    )
    base.update(kw)
    return ScreenSignals(**base)


@pytest.mark.parametrize(
    ("signals", "reason"),
    [
        (sig("AI", common=True), NotTechReason.COMMON),
        (sig("API", common=True), NotTechReason.COMMON),
        (sig("I'm"), NotTechReason.GENERIC),
        (sig("Agent", lowercase_ratio=0.88), NotTechReason.GENERIC),
        (sig("San Francisco", ner_labels={"GPE": 7, "ORG": 1}), NotTechReason.PLACE),
        (sig("Sam Altman", ner_labels={"PERSON": 5}), NotTechReason.PERSON),
        # 저장된 첫 표기가 축약형이면(key "im") 표시 이름이 "IM"이어도 거른다.
        (sig("IM", stored_name="I'm"), NotTechReason.GENERIC),
    ],
)
def test_non_technology_names_are_screened(signals, reason):
    assert screen(signals, CFG) == reason


@pytest.mark.parametrize(
    "signals",
    [
        sig("Claude Code", ner_labels={"PRODUCT": 3, "ORG": 2}),
        sig("OpenClaw", lowercase_ratio=0.05),
        # spaCy가 없으면 라벨이 비어 있다 — 그 단계는 건너뛴다.
        sig("Modal", ner_labels={}),
        # 라벨이 몇 개뿐이면 다수결을 믿지 않는다.
        sig("Hermes", ner_labels={"PERSON": 1}),
        # 운영 DB 실측: 흔한 이름 목록에 codex·chatgpt·claude가 있다 — 긴 이름은 LLM이 판정.
        sig("Codex", common=True),
        # 접두어 뿌리(claude code·vllm·mcp도 뿌리였다)는 거르지 않는다 — LLM 힌트로만.
        sig("vLLM", is_vendor_root=True),
    ],
)
def test_technologies_pass(signals):
    assert screen(signals, CFG) is None


def test_company_names_are_left_to_the_llm_with_a_vendor_hint():
    """OpenAI는 LLM 없는 단계에서 거르지 않고, 설명 프롬프트에 '회사일 수 있다' 힌트가 붙는다.
    LLM이 company라고 답하면 Vendor로 걸러진다(test_tech_classify 참고)."""
    import asyncio
    import json

    from argos.brain.tech_classify import TargetInput, build_plan

    seen = []

    class LLM:
        async def query(self, role, prompt, **kw):
            seen.append(prompt)
            return json.dumps({"is_technology": False, "not_tech_kind": "company"})

    s = sig("OpenAI", common=True, is_vendor_root=True)
    assert screen(s, CFG) is None
    t = TargetInput("e", "openai", "OpenAI", 9, ("a.", "b.", "c."), s)
    plan = asyncio.run(build_plan([t], [], LLM(), CFG))
    assert plan.screened == (("OpenAI", NotTechReason.VENDOR),)
    assert "may be a company" in seen[0]


def test_majority_label_needs_enough_mentions_and_a_majority():
    assert majority_ner_label({"GPE": 2}, min_mentions=3) is None
    assert majority_ner_label({"GPE": 2, "ORG": 2}, min_mentions=3) is None
    assert majority_ner_label({"GPE": 3, "ORG": 1}, min_mentions=3) == "GPE"
