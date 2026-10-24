"""ARG-308: 회사·제품 이름이 영어 원문 그대로 남았는지 보는 순수 함수."""

from __future__ import annotations

from argos.brain.event_detail_schema import missing_names, required_names
from argos.brain.event_summary_input import per_doc_budget


def test_transliterated_name_is_detected_as_missing():
    required = required_names(
        [["Claude Duet"]],
        "Anthropic shipped Claude Duet today",
        titles=["Anthropic launches Claude Duet"],
    )
    assert required == ["Claude Duet"]
    assert missing_names("앤트로픽이 클라우드 둈트를 공개했다", required) == ["Claude Duet"]
    assert missing_names("Anthropic이 Claude Duet을 공개했다", required) == []


def test_missing_check_ignores_case():
    assert missing_names("openai가 발표", ["OpenAI"]) == []


def test_only_latin_names_of_two_or_more_chars_in_the_evidence_and_a_title():
    evidence = "OpenAI and X released GPT-6. 오픈AI 발표"
    required = required_names(
        [["OpenAI", "X", "엔비디아", "GPT-6", "Gemini"]],
        evidence,
        titles=["OpenAI and X release GPT-6, Gemini rival"],
    )
    # X: 1자, 엔비디아: 라틴 문자 없음, Gemini: 근거 텍스트에 없음
    assert required == ["OpenAI", "GPT-6"]


def test_body_only_generic_entities_are_not_required():
    """실측 잡음(ARG-308): 본문에만 나온 일반어는 필수 이름이 아니다."""
    evidence = (
        "Claude Duet reads PDFs. A German-language campaign was disrupted. Claude Duet"
    )
    required = required_names(
        [["Claude Duet", "PDFs", "German-language"]],
        evidence,
        titles=["Anthropic launches Claude Duet"],
    )
    assert required == ["Claude Duet"]


def test_lowercase_suffix_form_is_dropped_even_in_a_title():
    required = required_names(
        [["German-language", "OpenAI"]],
        "OpenAI German-language",
        titles=["OpenAI disrupts German-language influence operation"],
    )
    assert required == ["OpenAI"]


def test_openai_influence_report_scenario_requires_only_title_names():
    """사건 bcac21ae: 기사 7건, 'German-language'가 두 번 다 빠져 실패하던 사건."""
    titles = [
        "OpenAI disrupts covert influence operations",
        "OpenAI report: state actors used ChatGPT for propaganda",
        "OpenAI bans accounts tied to influence campaigns",
        "Inside OpenAI's latest threat report",
        "ChatGPT misuse report from OpenAI",
        "OpenAI says Russian, Chinese groups used its models",
        "OpenAI threat intelligence update",
    ]
    bodies = " ".join(
        f"{title}. A German-language network posted PDFs and memes via ChatGPT." for title in titles
    )
    doc_names = [["OpenAI", "ChatGPT", "German-language", "PDFs"] for _ in titles]
    required = required_names(doc_names, bodies, titles=titles)
    assert "German-language" not in required
    assert "PDFs" not in required
    assert required == ["OpenAI", "ChatGPT"]
    output = "OpenAI가 ChatGPT를 악용한 독일어 영향 공작을 차단했다."
    assert missing_names(output, required) == []


def test_ranks_by_document_count_and_caps_at_three():
    evidence = "Alpha Beta Gamma Delta Epsilon"
    doc_names = [
        ["Beta", "Gamma", "Delta"],
        ["Beta", "Gamma"],
        ["Beta", "Epsilon", "Alpha"],
    ]
    required = required_names(
        doc_names, evidence, titles=["Alpha Beta Gamma Delta Epsilon roundup"]
    )
    assert required == ["Beta", "Gamma", "Delta"]


def test_dedupes_case_insensitively_keeping_the_first_spelling():
    required = required_names([["OpenAI"], ["openai"]], "OpenAI openai", titles=["OpenAI news"])
    assert required == ["OpenAI"]


def test_no_names_or_no_titles_means_nothing_required():
    assert required_names([], "text") == []
    assert required_names([["OpenAI"]], "OpenAI") == []
    assert missing_names("anything", []) == []


def test_per_doc_budget_splits_and_clamps():
    assert per_doc_budget(budget=60_000, n_docs=4, floor=1500, cap=6000) == 6000
    assert per_doc_budget(budget=60_000, n_docs=20, floor=1500, cap=6000) == 3000
    assert per_doc_budget(budget=60_000, n_docs=100, floor=1500, cap=6000) == 1500
    assert per_doc_budget(budget=60_000, n_docs=0, floor=1500, cap=6000) == 6000
    # cap below floor (tiny doc_max_chars setting) wins
    assert per_doc_budget(budget=60_000, n_docs=100, floor=1500, cap=800) == 800
