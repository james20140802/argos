"""갈래 초안 생성 (ARG-311 T3) — LLM은 전부 목, Ollama를 부르지 않는다."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from argos.brain import taxonomy as tx
from argos.brain import taxonomy_draft as td
from argos.brain.taxonomy_draft import Candidate, TaxonomyDraftError, build_draft, extract_sentences
from argos.config import TaxonomyConfig, UserConfig

DESC = "설명은 열 글자를 넘는 한 문장."


# ---- config ----
def test_taxonomy_config_defaults_and_bounds():
    cfg = TaxonomyConfig()
    assert cfg.model == "gemma4:26b"
    assert cfg.num_ctx == 16384
    assert cfg.draft_limit == 60
    assert isinstance(UserConfig().taxonomy, TaxonomyConfig)
    with pytest.raises(ValueError):
        TaxonomyConfig(draft_limit=0)
    with pytest.raises(ValueError):
        TaxonomyConfig(num_ctx=10)


# ---- sentences ----
def test_extract_sentences_name_boundary_limit_and_length():
    text = (
        "Ollama runs models locally. Nothing here. "
        "Use Ollama for serving! Ollamas is another word. "
        "ollama again.\nOllama fourth mention."
    )
    out = extract_sentences([text], "Ollama")
    assert out == ["Ollama runs models locally.", "Use Ollama for serving!", "ollama again."]
    long = "Git " + "x" * 400 + "."
    assert all(len(s) <= 300 for s in extract_sentences([long], "Git"))
    assert extract_sentences([""], "Git") == []


def test_extract_sentences_dedups():
    assert extract_sentences(["Git is good.", "Git is good."], "Git") == ["Git is good."]


# ---- build_draft ----
def _cands(n):
    return [Candidate(name=f"Tool{i}", sentences=(f"Tool{i} does a thing.",)) for i in range(n)]


def _tree_json():
    return json.dumps(
        {
            "tree": [
                {
                    "name": f"위{i}",
                    "description": DESC,
                    "children": [
                        {"name": f"아래{i}", "description": DESC, "examples": ["Tool0", "Bogus"]}
                    ],
                }
                for i in range(4)
            ]
        },
        ensure_ascii=False,
    )


def _llm(tree_replies):
    llm = AsyncMock()
    replies = iter(tree_replies)

    async def query(role, prompt, **kw):
        if "역할" in prompt and "JSON만 출력: {\"role\"" in prompt:
            return '{"role": "무언가 하는 도구"}'
        return next(replies)

    llm.query = AsyncMock(side_effect=query)
    return llm


@pytest.fixture
def patched(monkeypatch):
    async def fake_fetch(session, limit):
        return _cands(3)

    monkeypatch.setattr(td, "fetch_candidates", fake_fetch)


@pytest.mark.asyncio
async def test_build_draft_calls_n_plus_one_and_unloads(patched):
    llm = _llm([_tree_json()])
    tree = await build_draft(None, llm, limit=3)
    assert llm.query.await_count == 3 + 1
    llm.unload.assert_awaited_once_with("large")
    assert tx.validate(tree) == []
    # examples는 후보 목록에 있는 이름만 남는다
    assert tree[0].children[0].examples == ("Tool0",)
    assert all(b.id is None for b in tree)


@pytest.mark.asyncio
async def test_build_draft_retries_once_on_bad_json_then_succeeds(patched):
    llm = _llm(["이건 JSON이 아님", _tree_json()])
    tree = await build_draft(None, llm, limit=3)
    assert llm.query.await_count == 3 + 2
    assert len(tree) == 4


@pytest.mark.asyncio
async def test_build_draft_raises_after_two_failures_and_still_unloads(patched):
    bad = json.dumps({"tree": [{"name": "하나", "description": DESC, "children": []}]})
    llm = _llm([bad, bad])
    with pytest.raises(TaxonomyDraftError) as ei:
        await build_draft(None, llm, limit=3)
    assert ei.value.errors
    assert llm.query.await_count == 3 + 2
    llm.unload.assert_awaited_once_with("large")


@pytest.mark.asyncio
async def test_keep_top_fixes_top_names_ids_and_reuses_child_ids(patched):
    fixed = (
        tx.Branch("고정A", DESC, id="a", children=(tx.Branch("옛 아래", DESC, id="a1"),)),
        tx.Branch("고정B", DESC, id="b", children=(tx.Branch("옛 아래B", DESC, id="b1"),)),
        tx.Branch("고정C", DESC, id="c", children=(tx.Branch("옛 아래C", DESC, id="c1"),)),
    )
    reply = json.dumps(
        {
            "tree": [
                {"name": "고정A", "description": "LLM이 바꾼 설명!!!!!", "children": [
                    {"name": "옛 아래", "description": DESC},
                    {"name": "새 아래", "description": DESC, "examples": ["Tool1"]},
                ]},
                {"name": "고정B", "children": [{"name": "B 아래", "description": DESC}]},
                {"name": "고정C", "children": [{"name": "C 아래", "description": DESC}]},
                {"name": "LLM이 만든 위", "children": [{"name": "X", "description": DESC}]},
            ]
        },
        ensure_ascii=False,
    )
    llm = _llm([reply])
    tree = await build_draft(None, llm, limit=3, keep_top=fixed)
    assert [(b.name, b.id, b.description) for b in tree] == [
        (b.name, b.id, b.description) for b in fixed
    ]
    a = tree[0]
    assert [c.name for c in a.children] == ["옛 아래", "새 아래"]
    assert a.children[0].id == "a1" and a.children[1].id is None
    # 프롬프트에 고정 위 갈래가 명시됐다
    tree_prompt = llm.query.await_args_list[-1].args[1]
    assert "고정A" in tree_prompt and "고정" in tree_prompt


@pytest.mark.asyncio
async def test_keep_top_missing_top_in_reply_fails(patched):
    fixed = tuple(
        tx.Branch(f"고정{i}", DESC, id=str(i), children=(tx.Branch(f"아래{i}", DESC),))
        for i in range(3)
    )
    reply = json.dumps({"tree": [{"name": "고정0", "children": [{"name": "z", "description": DESC}]}]})
    llm = _llm([reply, reply])
    with pytest.raises(TaxonomyDraftError):
        await build_draft(None, llm, limit=3, keep_top=fixed)


@pytest.mark.asyncio
async def test_no_candidates_means_clear_error(monkeypatch):
    async def none(session, limit):
        return []

    monkeypatch.setattr(td, "fetch_candidates", none)
    llm = _llm([])
    with pytest.raises(TaxonomyDraftError):
        await build_draft(None, llm)
    llm.query.assert_not_awaited()
