from __future__ import annotations

from argos.brain._language import language_directive


def test_language_directive_names_language():
    out = language_directive("Korean")
    assert "Korean" in out
    assert out.startswith("\n")  # 프롬프트 끝에 이어붙는 블록


def test_language_directive_is_emphatic():
    out = language_directive("Korean")
    assert "IMPORTANT" in out
    # 원문 언어와 무관하게 출력 언어를 고정하라는 지시가 있어야 한다
    assert "regardless" in out.lower()
