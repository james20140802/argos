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
        (sig("OpenAI", is_vendor_root=True), NotTechReason.VENDOR),
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
    ],
)
def test_technologies_pass(signals):
    assert screen(signals, CFG) is None


def test_majority_label_needs_enough_mentions_and_a_majority():
    assert majority_ner_label({"GPE": 2}, min_mentions=3) is None
    assert majority_ner_label({"GPE": 2, "ORG": 2}, min_mentions=3) is None
    assert majority_ner_label({"GPE": 3, "ORG": 1}, min_mentions=3) == "GPE"
