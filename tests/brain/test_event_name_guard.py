"""ARG-308: 회사·제품 이름이 영어 원문 그대로 남았는지 보는 순수 함수."""

from __future__ import annotations

from argos.brain.event_detail_schema import missing_names, required_names
from argos.brain.event_summary_input import per_doc_budget


def test_transliterated_name_is_detected_as_missing():
    required = required_names([["Claude Duet"]], "Anthropic shipped Claude Duet today")
    assert required == ["Claude Duet"]
    assert missing_names("앤트로픽이 클라우드 둈트를 공개했다", required) == ["Claude Duet"]
    assert missing_names("Anthropic이 Claude Duet을 공개했다", required) == []


def test_missing_check_ignores_case():
    assert missing_names("openai가 발표", ["OpenAI"]) == []


def test_only_latin_names_of_two_or_more_chars_that_appear_in_the_evidence():
    evidence = "OpenAI and X released GPT-6. 오픈AI 발표"
    required = required_names([["OpenAI", "X", "엔비디아", "GPT-6", "Gemini"]], evidence)
    # X: 1자, 엔비디아: 라틴 문자 없음, Gemini: 근거 텍스트에 없음
    assert required == ["OpenAI", "GPT-6"]


def test_ranks_by_document_count_with_title_names_first_and_caps_at_three():
    evidence = "Alpha Beta Gamma Delta Epsilon"
    doc_names = [
        ["Beta", "Gamma", "Delta"],
        ["Beta", "Gamma"],
        ["Beta", "Epsilon", "Alpha"],
    ]
    required = required_names(doc_names, evidence, titles=["Alpha launches", None, ""])
    assert required == ["Alpha", "Beta", "Gamma"]


def test_dedupes_case_insensitively_keeping_the_first_spelling():
    required = required_names([["OpenAI"], ["openai"]], "OpenAI openai")
    assert required == ["OpenAI"]


def test_no_names_means_nothing_required():
    assert required_names([], "text") == []
    assert missing_names("anything", []) == []


def test_per_doc_budget_splits_and_clamps():
    assert per_doc_budget(budget=60_000, n_docs=4, floor=1500, cap=6000) == 6000
    assert per_doc_budget(budget=60_000, n_docs=20, floor=1500, cap=6000) == 3000
    assert per_doc_budget(budget=60_000, n_docs=100, floor=1500, cap=6000) == 1500
    assert per_doc_budget(budget=60_000, n_docs=0, floor=1500, cap=6000) == 6000
    # cap below floor (tiny doc_max_chars setting) wins
    assert per_doc_budget(budget=60_000, n_docs=100, floor=1500, cap=800) == 800
