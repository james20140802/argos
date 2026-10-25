"""자동 갈래 배정 LLM 단계 (ARG-313) — 가짜 LLM, DB/Ollama 불필요."""
from __future__ import annotations

import json

import pytest

from argos.brain import taxonomy as tx
from argos.brain.tech_classify import (
    ExistingProfile,
    ScreenSignals,
    TargetInput,
    UnclassifiedReason,
    build_plan,
    leaves_of,
    size_suggestions,
)
from argos.brain.tech_evidence import Neighbor
from argos.config import TaxonomyConfig
from argos.models.technology_profile import NotTechReason, TechVerdict

CFG = TaxonomyConfig()
TREE = tx.SPIKE_SEED
LEAVES = leaves_of(TREE)
CODING = next(lf for lf in LEAVES if lf.name.startswith("코딩 에이전트"))
SERVING = next(lf for lf in LEAVES if lf.name.startswith("모델 실행"))


class FakeLLM:
    """프롬프트 종류(설명/배정)와 이름으로 응답을 고른다. 호출을 기록한다."""

    def __init__(self, describe=None, assign=None):
        self.describe = describe or {}
        self.assign = assign or {}
        self.calls: list[tuple[str, str]] = []

    async def query(self, role, prompt, **kw):
        if prompt.startswith("Below are sentences"):
            name = prompt.split('mention "', 1)[1].split('"', 1)[0]
            self.calls.append(("describe", name))
            answer = self.describe[name]
            if isinstance(answer, Exception):
                raise answer
            return answer if isinstance(answer, str) else json.dumps(answer)
        name = prompt.split("Technology: ", 1)[1].split("\n", 1)[0]
        self.calls.append(("assign", name))
        answer = self.assign[name]
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            answer = answer(prompt)
        return json.dumps(answer) if not isinstance(answer, str) else answer

    async def unload(self, role):
        pass


def target(display, *, sentences=3, neighbors=(), existing=None, needs=True, **sig):
    key = display.lower()
    signals = ScreenSignals(
        key=key, display=display, common=False, lowercase_ratio=0.0,
        is_vendor_root=False, ner_labels={},
    )
    for k, v in sig.items():
        signals = ScreenSignals(**{**signals.__dict__, k: v})
    return TargetInput(
        entity_id=f"id-{key}", key=key, display=display, doc_count=4,
        sentences=tuple(f"{display} sentence {i}." for i in range(sentences)),
        signals=signals, neighbors=tuple(neighbors), existing=existing,
        needs_describe=needs,
    )


TECH = {
    "is_technology": True, "not_tech_kind": "none", "enough": True,
    "kind": "coding agent", "one_liner": "Agent that writes code.",
    "description_ko": "코드를 작성하는 코딩 에이전트",
}


@pytest.mark.asyncio
async def test_technology_is_assigned_without_approval():
    llm = FakeLLM(describe={"Claude Code": TECH}, assign={"Claude Code": {"leaf_index": CODING.index, "reason": "codes"}})
    plan = await build_plan([target("Claude Code")], LEAVES, llm, CFG)
    (a,) = plan.assignments
    assert a.leaf_path == CODING.path
    assert a.evidence["llm_reason"] == "codes"
    assert a.evidence["description"] == "코드를 작성하는 코딩 에이전트"
    (p,) = plan.profiles
    assert p.verdict == TechVerdict.TECHNOLOGY and p.role_en == "Agent that writes code."


@pytest.mark.asyncio
async def test_enterprise_adoption_sentences_only_leave_it_unclassified():
    """GPT-6 Astra: 기업 도입 문장뿐 -> 설명 부족 -> 증거도 없으면 배정 질의 없이 미분류."""
    llm = FakeLLM(describe={"GPT-6 Astra": {**TECH, "enough": False, "kind": "model"}})
    plan = await build_plan([target("GPT-6 Astra")], LEAVES, llm, CFG)
    (a,) = plan.assignments
    assert a.branch_id is None and a.leaf_path is None
    assert a.unclassified_reason == UnclassifiedReason.INSUFFICIENT
    assert plan.profiles[0].verdict == TechVerdict.INSUFFICIENT
    assert plan.profiles[0].description is None  # 추측 설명은 저장하지 않는다
    assert llm.calls == [("describe", "GPT-6 Astra")]


@pytest.mark.asyncio
async def test_too_few_sentences_skips_the_llm():
    llm = FakeLLM()
    plan = await build_plan([target("Scion", sentences=2)], LEAVES, llm, CFG)
    assert llm.calls == []
    # 진짜 근거 부족은 인프라 실패와 달리 그대로 저장한다(문서가 늘면 다시 설명).
    assert plan.profiles[0].verdict == TechVerdict.INSUFFICIENT and plan.skipped == ()
    assert plan.assignments[0].unclassified_reason == UnclassifiedReason.INSUFFICIENT


@pytest.mark.asyncio
async def test_insufficient_but_with_pattern_evidence_still_asks_for_a_branch():
    llm = FakeLLM(assign={"Scion": {"leaf_index": CODING.index, "reason": "listed with git"}})
    t = target("Scion", sentences=1, neighbors=[Neighbor("git", 2, ("Git, Scion",))])
    plan = await build_plan([t], LEAVES, llm, CFG, current={"git": CODING.path})
    (a,) = plan.assignments
    assert a.leaf_path == CODING.path
    assert a.evidence["neighbors"][0] == {
        "name": "git", "count": 2, "branch": CODING.path, "examples": ["Git, Scion"],
    }


@pytest.mark.asyncio
async def test_minus_one_means_unclassified():
    llm = FakeLLM(describe={"Foo": TECH}, assign={"Foo": {"leaf_index": -1, "reason": "none fit"}})
    plan = await build_plan([target("Foo")], LEAVES, llm, CFG)
    assert plan.assignments[0].unclassified_reason == UnclassifiedReason.NO_FIT


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ConnectionError("ollama down"), TimeoutError(), "not json"])
async def test_describe_infra_failure_writes_nothing_and_is_retried(failure):
    """설명 LLM 일시 장애는 '근거 부족'이 아니다 — 프로필·배정을 쓰지 않아 기존 행이
    보존되고(설명·described_doc_count 미전진) 다음 실행에서 다시 대상이 된다."""
    from argos.brain.tech_classify_store import NameRow, select_targets

    llm = FakeLLM(describe={"Foo": failure, "Bar": TECH}, assign={"Bar": {"leaf_index": 0}})
    ex = ExistingProfile(TechVerdict.TECHNOLOGY, None, "agent", "기존 정상 설명", "Old role.", 4)
    plan = await build_plan(
        [target("Foo", existing=ex, needs=True), target("Bar")], LEAVES, llm, CFG
    )
    assert [p.key for p in plan.profiles] == ["bar"]  # Foo의 기존 설명을 덮어쓰지 않는다
    assert [a.key for a in plan.assignments] == ["bar"]
    assert plan.skipped == (("Foo", "describe"),)
    assert "LLM 실패로 건너뜀 1개" in "\n".join(plan.preview_lines())
    # 기존 행이 그대로면(문서 8 >= 4*2) 다음 선정에도 다시 들어간다.
    row = NameRow("id-foo", "foo", "Foo", 8, ex, None)
    assert [r.key for r, d in select_targets([row], CFG)] == ["foo"]
    # 새 이름이면 프로필이 없으니 자연히 다시 고른다.
    assert select_targets([NameRow("id-foo", "foo", "Foo", 3)], CFG)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ConnectionError("ollama down"), "not json"])
async def test_assign_infra_failure_does_not_overwrite_existing_assignment(failure):
    llm = FakeLLM(describe={"Foo": TECH}, assign={"Foo": failure})
    plan = await build_plan([target("Foo")], LEAVES, llm, CFG, current={"foo": CODING.path})
    assert plan.assignments == ()  # 기존 Auto 배정을 미분류로 덮어쓰지 않는다
    assert plan.skipped == (("Foo", "assign"),)
    assert [p.key for p in plan.profiles] == ["foo"]  # 설명은 성공했으니 저장


@pytest.mark.asyncio
async def test_round_two_failure_keeps_round_one_result():
    state = {"n": 0}

    def codex(prompt):
        state["n"] += 1
        if state["n"] == 2:
            raise TimeoutError()
        return {"leaf_index": CODING.index}

    llm = FakeLLM(
        describe={"Codex": TECH, "Cursor": TECH},
        assign={"Codex": codex, "Cursor": {"leaf_index": CODING.index}},
    )
    plan = await build_plan(
        [target("Codex", neighbors=[Neighbor("cursor", 2, ())]), target("Cursor")], LEAVES, llm, CFG
    )
    by = {a.display: a for a in plan.assignments}
    assert by["Codex"].leaf_path == CODING.path and by["Codex"].evidence["round"] == 1
    assert plan.skipped == ()


@pytest.mark.asyncio
async def test_screened_and_llm_rejected_names_are_not_assigned():
    llm = FakeLLM(describe={"Anthropic": {**TECH, "is_technology": False, "not_tech_kind": "company"}})
    plan = await build_plan(
        [target("AI", common=True), target("Anthropic")], LEAVES, llm, CFG
    )
    assert plan.assignments == ()
    assert plan.screened == (("AI", NotTechReason.COMMON), ("Anthropic", NotTechReason.VENDOR))
    assert {d for _e, d in plan.unassign} == {"AI", "Anthropic"}
    assert llm.calls == [("describe", "Anthropic")]  # AI는 LLM 없이 걸렸다


@pytest.mark.asyncio
async def test_second_round_requeries_only_names_whose_neighbors_moved():
    """Codex는 이웃(Cursor)의 갈래를 처음엔 모른다 -> Cursor가 배정되면 Codex만 다시 묻는다."""
    def codex(prompt):
        return {"leaf_index": CODING.index if "currently in" in prompt else SERVING.index}

    llm = FakeLLM(
        describe={"Codex": TECH, "Cursor": TECH, "DuckDB": TECH},
        assign={"Codex": codex, "Cursor": {"leaf_index": CODING.index}, "DuckDB": {"leaf_index": 0}},
    )
    targets = [
        target("Codex", neighbors=[Neighbor("cursor", 3, ())]),
        target("Cursor", neighbors=[]),
        target("DuckDB"),
    ]
    plan = await build_plan(targets, LEAVES, llm, CFG)
    assigns = [c for c in llm.calls if c[0] == "assign"]
    assert assigns == [("assign", "Codex"), ("assign", "Cursor"), ("assign", "DuckDB"), ("assign", "Codex")]
    by = {a.display: a for a in plan.assignments}
    assert by["Codex"].leaf_path == CODING.path
    assert by["Codex"].evidence["round"] == 2


@pytest.mark.asyncio
async def test_existing_profile_is_reused_without_describing_again():
    ex = ExistingProfile(TechVerdict.TECHNOLOGY, None, "database", "분석용 DB", "Analytical DB.", 3)
    llm = FakeLLM(assign={"DuckDB": {"leaf_index": 0}})
    plan = await build_plan([target("DuckDB", existing=ex, needs=False)], LEAVES, llm, CFG)
    assert llm.calls == [("assign", "DuckDB")]
    assert plan.profiles == ()  # 설명을 다시 쓰지 않는다
    assert plan.assignments[0].evidence["description"] == "분석용 DB"


def test_preview_lists_branches_unclassified_and_screened():
    import asyncio

    llm = FakeLLM(describe={"Claude Code": TECH}, assign={"Claude Code": {"leaf_index": CODING.index}})
    plan = asyncio.run(
        build_plan([target("Claude Code"), target("AI", common=True), target("Scion", sentences=1)], LEAVES, llm, CFG)
    )
    text = "\n".join(plan.preview_lines())
    assert CODING.path in text and "Claude Code — 코드를 작성하는 코딩 에이전트" in text
    assert "미분류 1개" in text and "Scion (InsufficientEvidence)" in text
    assert "AI (Common)" in text


def test_size_suggestion_only_suggests():
    cfg = TaxonomyConfig(split_suggest_size=3, split_suggest_ratio=0.9)
    out = size_suggestions({"A > 코딩": 4, "A > 모델": 2}, cfg)
    assert len(out) == 1 and "코딩" in out[0] and "taxonomy split" in out[0]
    out = size_suggestions({"A > 코딩": 5, "A > 모델": 1, "B > x": 1}, TaxonomyConfig())
    assert out and "71%" in out[0]  # 전체의 30% 초과


@pytest.mark.asyncio
async def test_names_rejected_in_this_run_are_dropped_from_neighbor_evidence():
    seen = []

    def gpt(prompt):
        seen.append(prompt)
        return {"leaf_index": 0}

    llm = FakeLLM(
        describe={
            "OpenAI": {**TECH, "is_technology": False, "not_tech_kind": "company"},
            "GPT-6 Astra": TECH,
        },
        assign={"GPT-6 Astra": gpt},
    )
    targets = [
        target("OpenAI"),
        target("GPT-6 Astra", neighbors=[Neighbor("openai", 3, ()), Neighbor("opus 5.5", 3, ())]),
    ]
    plan = await build_plan(targets, LEAVES, llm, CFG)
    assert "with openai" not in seen[0] and "with opus 5.5" in seen[0]
    (a,) = plan.assignments
    assert [n["name"] for n in a.evidence["neighbors"]] == ["opus 5.5"]


def test_prefer_full_name_over_fragment():
    from argos.brain.tech_classify_store import prefer_full_name

    texts = ["GPT-6 Astra beats GPT-6 Astra mini. Also 6 Astra alone once."]
    assert prefer_full_name("6 Astra", ["GPT-6 Astra"], texts) == "GPT-6 Astra"
    # 긴 표기가 드물면(조각의 절반 미만) 바꾸지 않는다.
    texts = ["Claude Code is great. Claude Code again. Claude Code CLI once. Claude Code more."]
    assert prefer_full_name("Claude Code", ["Claude Code CLI"], texts) == "Claude Code"
