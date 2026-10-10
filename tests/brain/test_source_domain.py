"""ARG-295: 등록 도메인 규칙 — 결정적, 의존성 없음."""

from __future__ import annotations

import pytest

from argos.brain.source_domain import registered_domain


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://openai.com/index/sora-2", "openai.com"),
        ("https://www.openai.com/index/x", "openai.com"),
        ("https://blog.OpenAI.com:443/x", "openai.com"),
        ("https://huggingface.co/blog/x", "huggingface.co"),
        ("https://www.bbc.co.uk/news/x", "bbc.co.uk"),
        ("https://news.naver.co.kr/a", "naver.co.kr"),
        ("https://stephen.bochinski.dev/blog/x", "bochinski.dev"),
        ("openai.com/index/x", "openai.com"),
        ("https://openai.com./x", "openai.com"),
        ("http://127.0.0.1:8000/x", "127.0.0.1"),
        ("http://localhost/x", "localhost"),
        ("https://co.uk/x", "co.uk"),
        ("", None),
        (None, None),
        ("not a url", None),
    ],
)
def test_registered_domain_table(url, expected):
    assert registered_domain(url) == expected


def test_registered_domain_is_deterministic():
    url = "https://techcrunch.com/2026/09/08/x/"
    assert {registered_domain(url) for _ in range(5)} == {"techcrunch.com"}
