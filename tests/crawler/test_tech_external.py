"""기사 링크 기반 외부 증거 (ARG-313) — httpx.MockTransport만 쓴다(실제 네트워크 없음)."""
from __future__ import annotations

import json

import httpx
import pytest

from argos.crawler.tech_external import ExternalEvidence, links_for


def test_links_only_from_article_bodies_and_must_contain_the_name():
    texts = [
        "See https://github.com/vllm-project/vllm and github.com/vllm-project/vllm.git for code.",
        "Unrelated: https://github.com/torvalds/linux and https://github.com/topics/llm",
        "Weights: https://huggingface.co/deepseek-ai/DeepSeek-V4-Base ; docs huggingface.co/docs/hub",
    ]
    gh, hf = links_for(texts, "vLLM")
    assert gh == ["vllm-project/vllm"]
    assert hf == []
    gh, hf = links_for(texts, "DeepSeek V4")
    assert gh == [] and hf == ["deepseek-ai/DeepSeek-V4-Base"]


def test_too_short_names_get_no_links():
    gh, hf = links_for(["https://github.com/pi-org/pi-mono"], "Pi")
    assert gh == [] and hf == []


class Recorder:
    def __init__(self, routes):
        self.routes = routes
        self.seen: list[str] = []
        self.agents: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.seen.append(url)
        self.agents.append(request.headers.get("User-Agent", ""))
        answer = self.routes[url]
        if callable(answer):
            return answer(request)
        status, body = answer
        return httpx.Response(status, json=body)


async def _allow(url, ua):
    return True


def _collector(tmp_path, routes, **kw):
    rec = Recorder(routes)
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    sleeps: list[float] = []

    async def fake_sleep(s):
        sleeps.append(s)

    opts = dict(
        client=client,
        cache_path=tmp_path / "cache.json",
        ttl_seconds=3600,
        max_requests=10,
        robots=_allow,
        sleep=fake_sleep,
        now=lambda: 1000.0,
    )
    opts.update(kw)
    return ExternalEvidence(**opts), rec, sleeps


GH = "https://api.github.com/repos/vllm-project/vllm"
HF = "https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Base"


@pytest.mark.asyncio
async def test_github_topics_and_hf_pipeline_tag(tmp_path):
    ext, rec, _ = _collector(
        tmp_path,
        {GH: (200, {"topics": ["inference", "llm-serving"]}), HF: (200, {"pipeline_tag": "text-generation"})},
    )
    tags = await ext.tags_for(["vllm-project/vllm"], ["deepseek-ai/DeepSeek-V4-Base"])
    assert tags == ["gh:inference", "gh:llm-serving", "hf:text-generation"]
    # 크롤러와 같은 브라우저 UA 로테이션을 쓴다(httpx 기본 UA가 아니다).
    assert all(a.startswith("Mozilla/5.0") for a in rec.agents)


@pytest.mark.asyncio
async def test_robots_denied_skips_the_request(tmp_path):
    async def deny(url, ua):
        return False

    ext, rec, _ = _collector(tmp_path, {}, robots=deny)
    assert await ext.tags_for(["vllm-project/vllm"], []) == []
    assert rec.seen == []


@pytest.mark.asyncio
async def test_429_backs_off_then_succeeds(tmp_path):
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        if state["n"] < 3:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={"topics": ["llm"]})

    ext, rec, sleeps = _collector(tmp_path, {GH: flaky})
    assert await ext.tags_for(["vllm-project/vllm"], []) == ["gh:llm"]
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_persistent_failure_empties_only_that_evidence(tmp_path):
    def boom(request):
        raise httpx.ConnectError("down")

    ext, _rec, _ = _collector(tmp_path, {GH: boom, HF: (200, {"pipeline_tag": "text-generation"})})
    tags = await ext.tags_for(["vllm-project/vllm"], ["deepseek-ai/DeepSeek-V4-Base"])
    assert tags == ["hf:text-generation"]


@pytest.mark.asyncio
async def test_cache_is_reused_within_ttl_and_refreshed_after(tmp_path):
    ext, rec, _ = _collector(tmp_path, {GH: (200, {"topics": ["a"]})})
    await ext.tags_for(["vllm-project/vllm"], [])
    ext.save()
    cached = json.loads((tmp_path / "cache.json").read_text())
    assert cached[GH]["tags"] == ["gh:a"]

    ext2, rec2, _ = _collector(tmp_path, {GH: (200, {"topics": ["b"]})}, now=lambda: 2000.0)
    assert await ext2.tags_for(["vllm-project/vllm"], []) == ["gh:a"]
    assert rec2.seen == []

    ext3, rec3, _ = _collector(tmp_path, {GH: (200, {"topics": ["b"]})}, now=lambda: 1000.0 + 7200)
    assert await ext3.tags_for(["vllm-project/vllm"], []) == ["gh:b"]
    assert rec3.seen == [GH]


@pytest.mark.asyncio
async def test_per_run_request_cap(tmp_path):
    ext, rec, _ = _collector(
        tmp_path, {GH: (200, {"topics": ["a"]}), HF: (200, {"pipeline_tag": "x"})}, max_requests=1
    )
    tags = await ext.tags_for(["vllm-project/vllm"], ["deepseek-ai/DeepSeek-V4-Base"])
    assert tags == ["gh:a"]
    assert rec.seen == [GH]


@pytest.mark.asyncio
async def test_404_is_cached_as_empty(tmp_path):
    ext, rec, _ = _collector(tmp_path, {GH: (404, {"message": "Not Found"})})
    assert await ext.tags_for(["vllm-project/vllm"], []) == []
    assert await ext.tags_for(["vllm-project/vllm"], []) == []
    assert rec.seen == [GH]
