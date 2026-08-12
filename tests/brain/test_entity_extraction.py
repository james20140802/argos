import unicodedata

import pytest

from argos.brain.entity_extraction import ExtractedName, extract_names
from argos.brain.entity_names import canonical_name
from argos.config import EventDetectionConfig, settings


def canonicals(documents, **kwargs):
    """배치 결과를 문서별 정규형 집합으로 납작하게 만든다."""
    return [{name.canonical for name in doc} for doc in extract_names(documents, **kwargs)]


def test_multi_word_product_name_stays_one_name():
    [names] = canonicals(["Anthropic today shipped Claude Sonnet 5 to the API."])
    assert "claude sonnet 5" in names
    # 조각으로 쪼개져 나오면 안 된다.
    assert "claude" not in names
    assert "sonnet" not in names


def test_versioned_name_is_extracted():
    [names] = canonicals(["The lab compared GPT-5.2 against its own baseline."])
    assert "gpt 5.2" in names


def test_sentence_initial_capital_is_not_a_name():
    [names] = canonicals(
        ["The pricing page changed quietly. Available today for every customer."]
    )
    assert "the" not in names
    assert "available" not in names


def test_common_words_are_not_names():
    [names] = canonicals(["Report says the rollout slipped. New guidance lands Today."])
    assert names.isdisjoint({"report", "new", "today"})


def test_sentence_initial_name_is_kept_when_it_recurs_mid_sentence():
    # 문장 첫 단어라는 이유만으로 버리면 진짜 이름을 놓친다. 같은 배치 안에서
    # 문장 중간에도 대문자로 나오면 그건 문장부호가 아니라 이름이라는 증거다.
    [names] = canonicals(
        ["Blackwell shipped in volume. Demand for Blackwell outran supply."]
    )
    assert "blackwell" in names


def test_sentence_initial_capital_stays_out_without_that_evidence():
    [names] = canonicals(["Available capacity fell sharply last quarter."])
    assert "available" not in names


def test_results_align_with_input_order_and_length():
    docs = [
        "Engineers at Anthropic tuned Claude Sonnet 5 overnight.",
        "The rival lab shipped GPT-5.2 the same week.",
        "",
    ]
    results = canonicals(docs)
    assert len(results) == 3
    assert "claude sonnet 5" in results[0]
    assert "gpt 5.2" in results[1]
    assert results[2] == set()


def test_names_are_folded_by_the_canonical_form():
    # 정규형 계약의 소유자는 ARG-250이다. 여기서는 그걸 쓴다는 사실만 고정한다.
    docs = ["Benchmarks put Sonnet-5 ahead.", "Reviewers preferred Sonnet 5 overall."]
    first, second = canonicals(docs)
    assert "sonnet 5" in first
    assert "sonnet 5" in second


def test_surface_form_is_preserved_for_display():
    [names] = extract_names(["Engineers at Anthropic tuned Claude Sonnet 5 overnight."])
    by_key = {name.canonical: name for name in names}
    assert by_key["claude sonnet 5"].surface == "Claude Sonnet 5"
    assert isinstance(by_key["claude sonnet 5"], ExtractedName)


def test_deterministic_across_calls():
    docs = [
        "Engineers at Anthropic tuned Claude Sonnet 5 overnight.",
        "Reviewers at OpenAI measured GPT-5.2 on the same suite.",
    ]
    assert extract_names(docs) == extract_names(docs)


def test_results_are_sorted_by_canonical_form():
    [names] = extract_names(["Reviewers compared Claude Sonnet 5 with GPT-5.2 directly."])
    keys = [name.canonical for name in names]
    assert keys == sorted(keys)


def test_name_in_too_many_documents_is_dropped(monkeypatch):
    monkeypatch.setattr(
        settings.user,
        "event_detection",
        EventDetectionConfig(entity_max_doc_ratio=0.5, entity_df_min_batch=5),
    )
    # 5건 전부에 나오는 이름(비율 1.0)은 변별력이 없다 — 사건을 가르지 못한다.
    docs = [f"Analysts at Acme Corp reviewed report number {i} closely." for i in range(5)]
    for names in canonicals(docs):
        assert "acme corp" not in names


def test_name_below_the_ratio_cut_survives(monkeypatch):
    monkeypatch.setattr(
        settings.user,
        "event_detection",
        EventDetectionConfig(entity_max_doc_ratio=0.5, entity_df_min_batch=5),
    )
    docs = [
        "Engineers at Acme Corp shipped the update.",
        "Reviewers at Acme Corp confirmed the numbers.",
        "The city council approved a bicycle lane downtown.",
        "Farmers reported a dry season across the valley.",
        "Rail operators extended the weekend timetable.",
    ]
    assert "acme corp" in canonicals(docs)[0]


def test_ratio_cut_is_skipped_for_small_batches(monkeypatch):
    monkeypatch.setattr(
        settings.user,
        "event_detection",
        EventDetectionConfig(entity_max_doc_ratio=0.5, entity_df_min_batch=5),
    )
    # 문서 1건짜리 배치에서 비율 컷을 적용하면 모든 이름이 탈락한다.
    [names] = canonicals(["Engineers at Acme Corp shipped Claude Sonnet 5 overnight."])
    assert "acme corp" in names


def test_ngram_width_comes_from_config(monkeypatch):
    monkeypatch.setattr(
        settings.user, "event_detection", EventDetectionConfig(entity_max_ngram=2)
    )
    [names] = canonicals(["Reviewers tested Claude Sonnet 5 last week."])
    assert "claude sonnet 5" not in names
    assert "claude sonnet" in names


def test_explicit_max_ngram_overrides_config(monkeypatch):
    monkeypatch.setattr(
        settings.user, "event_detection", EventDetectionConfig(entity_max_ngram=2)
    )
    [names] = canonicals(["Reviewers tested Claude Sonnet 5 last week."], max_ngram=4)
    assert "claude sonnet 5" in names


def test_comma_separated_names_do_not_merge():
    # 쉼표로 나열된 서로 다른 이름들이 한 덩어리로 붙으면, 뒤쪽 이름은 n-gram
    # 폭에 잘려 아예 사라진다 — 사건 묶기의 재료가 통째로 없어진다.
    [names] = canonicals(
        ["Analysts compared Acme Corp, Globex, Initech, and Umbrella this quarter."]
    )
    assert {"acme corp", "globex", "initech", "umbrella"} <= names
    assert "acme corp globex initech" not in names


def test_long_capitalized_run_drops_nothing():
    # 제목처럼 대문자가 길게 이어지면 n-gram 폭을 넘는다. 앞쪽만 남기고 조용히
    # 버리면 뒤쪽 이름은 어디에도 나타나지 않는다.
    [names] = canonicals(
        ["Reporters watched Alpha Beta Gamma Delta Epsilon Zeta ship the update."]
    )
    assert "epsilon zeta" in names


def test_latin_names_inside_korean_text_are_extracted():
    # 한글 자체는 대소문자가 없어 규칙 경로가 잡을 근거가 없다. 다만 한글 기사에
    # 섞인 라틴 표기 이름은 잡혀야 한다 — 국내 기사의 실제 모습이다.
    [names] = canonicals(
        ["앤트로픽이 Claude Sonnet 5를 오늘 공개했다. 성능이 올랐다고 밝혔다."]
    )
    assert "claude sonnet 5" in names


def test_lowercase_leading_product_names_are_extracted():
    # 첫 글자만 보면 'iOS'·'macOS'처럼 소문자로 시작하는 상표명이 통째로
    # 사라진다 — 이 프로젝트가 쫓는 이름들이 정확히 이 모양이다.
    [names] = canonicals(["Reviewers compared iOS with macOS yesterday."])
    assert {"ios", "macos"} <= names


def test_lowercase_leading_company_names_are_extracted():
    [names] = canonicals(["Analysts said eBay and xAI both shipped."])
    assert {"ebay", "xai"} <= names


@pytest.mark.parametrize(
    "surface,expected,bogus",
    [("3M", "3m", "m"), ("1Password", "1password", "password"), ("7-Zip", "7 zip", "zip")],
)
def test_digit_leading_names_are_extracted(surface, expected, bogus):
    # 숫자로 시작하는 상표명은 숫자 토큰과 꼬리로 갈라져, 진짜 이름은 사라지고
    # 없는 이름('m' 'password' 'zip')이 문서빈도를 오염시킨다.
    [names] = canonicals([f"Reviewers compared {surface} with rivals yesterday."])
    assert expected in names
    assert bogus not in names


def test_possessive_stays_when_it_is_part_of_the_name():
    # 'Moody's'·'McDonald's'의 's는 소유격이 아니라 이름의 일부다. 무조건 끊으면
    # 진짜 회사 이름이 'moody'로 잘리고, spaCy가 넘긴 'moodys'와 갈려 한 회사가
    # 두 키로 쪼개진다.
    [names] = canonicals(["Analysts compared Moody's with S&P Global."])
    assert "moodys" in names
    assert "moody" not in names


def test_middle_initial_stays_with_the_surname():
    # 'George W. Bush'의 W는 가운데 이름이다. 거기서 끊으면 성이 문장 첫 단어로
    # 둔갑해 탈락하고 'george w'라는 반쪽 이름만 남는다.
    [names] = canonicals(["Reporters met George W. Bush yesterday."])
    assert "george w bush" in names


def test_possessive_folds_when_the_bare_name_appears_in_the_batch():
    # 배치 어딘가에 맨 이름이 나오면 그 's는 소유격이었다는 뜻이다. 안 접으면
    # 같은 회사가 'anthropic'과 'anthropics' 두 키로 쪼개진다.
    first, _ = canonicals(
        [
            "Analysts discussed Anthropic's new model.",
            "Reviewers said Anthropic shipped it fast.",
        ]
    )
    assert "anthropic" in first
    assert "anthropics" not in first


def test_folded_possessive_drops_the_s_from_the_display_surface():
    # 정규형만 접고 표시용을 그대로 두면 같은 키를 "Anthropic's"로 보여 준다.
    first, _ = extract_names(
        [
            "Analysts discussed Anthropic's new model.",
            "Reviewers said Anthropic shipped it fast.",
        ]
    )
    assert "Anthropic" in {name.surface for name in first}
    assert "Anthropic's" not in {name.surface for name in first}


def test_possessive_does_not_fold_without_a_bare_mention():
    # 반대 방향 경계. 배치가 소유격만 담고 있으면 그 's가 이름의 일부인지
    # 소유격인지 가릴 근거가 없다 — 접지 않는 쪽이 'McDonald's'를 지킨다.
    first, _ = canonicals(
        [
            "Analysts reviewed McDonald's quarterly filing.",
            "Reporters cited McDonald's earlier statement.",
        ]
    )
    assert "mcdonalds" in first
    assert "mcdonald" not in first


def test_possessive_still_splits_before_another_name():
    # 반대 방향 경계. 소유격 뒤에 다른 이름이 오면 거기서 끊어야 한다.
    [names] = canonicals(["Reviewers tested Anthropic's Claude against GPT-5.2."])
    assert {"anthropic", "claude"} <= names
    assert "anthropics claude" not in names


def test_surface_keeps_the_crawled_spelling():
    # 정규형은 표기를 접지만 표시용 원문은 본 그대로여야 한다. 전각으로 실린
    # 기사를 반각으로 바꿔 보여 주면 원문 표기를 되살릴 방법이 없다.
    names = extract_names(["Reviewers compared Ｃ＋＋ with Ｃ＃ yesterday."])[0]
    assert {"Ｃ＋＋", "Ｃ＃"} <= {name.surface for name in names}
    assert {"c++", "c#"} <= {name.canonical for name in names}


def test_fullwidth_forms_fold_to_their_ascii_names():
    # CJK 본문은 라틴 글자를 전각으로 적는 일이 흔하다. 호환 형태를 접지 않으면
    # 전각 기호가 토큰화에서 버려져 'Ｃ＋＋'와 'Ｃ＃'이 둘 다 'c'로 뭉개진다.
    [symbols] = canonicals(["Reviewers compared Ｃ＋＋ with Ｃ＃ yesterday."])
    assert {"c++", "c#"} <= symbols
    [versioned] = canonicals(["Engineers benchmarked ＧＰＴ－５ against rivals."])
    assert "gpt 5" in versioned


def test_fullwidth_leading_dot_opens_a_name():
    # 전각 마침표를 무조건 문장 끝으로 옮기면 '．ＮＥＴ'의 앞점이 사라지고
    # 이름이 통째로 결과에서 빠진다. 반각으로 쓴 같은 기술은 '.net'으로 잡힌다.
    names = extract_names(["Reviewers compared ．ＮＥＴ with Java yesterday."])[0]
    assert ".net" in {name.canonical for name in names}
    assert "．ＮＥＴ" in {name.surface for name in names}


def test_fullwidth_leading_dot_survives_opening_punctuation():
    # 여는 괄호·따옴표 뒤도 이름이 시작하는 자리다. 앞이 공백일 때만 앞점으로
    # 보면 '（．ＮＥＴ）'의 이름이 다시 통째로 사라진다.
    [names] = canonicals(["Reviewers compared （．ＮＥＴ） with Java yesterday."])
    assert ".net" in names


def test_symbol_suffix_survives_a_name_joiner():
    # 'C++/CLI'와 'F#/.NET'은 각각 기술 하나다. 빗금에서 끊으면 정규형이 이미
    # 한 덩어리로 접는 이름이 문서빈도에서 둘로 쪼개진다.
    [names] = canonicals(["Reviewers tested C++/CLI with F#/.NET yesterday."])
    assert {"c++ cli", "f# .net"} <= names
    assert names.isdisjoint({"cli", "c++", "f#"})


@pytest.mark.parametrize("surface", ["DALL·E", "DALL-E", "DALL E"])
def test_middle_dot_stays_inside_the_name(surface):
    # 'DALL·E'의 가운뎃점은 이름의 일부다. 거기서 끊으면 제품 하나가 사라지고
    # 조각 둘이 남는다. 붙임표·공백으로 쓴 같은 제품과도 한 키로 모여야 한다 —
    # 아니면 표기만 다른 같은 이름이 문서빈도를 나눠 갖는다.
    [names] = canonicals([f"Analysts examined {surface} and GPT-5 today."])
    assert "dall e" in names
    assert names.isdisjoint({"dall", "e", "dalle"})


def test_quoted_sentence_inside_prose_starts_a_sentence():
    # 문장 중간에서 인용이 열리면 그 안은 새 문장의 첫 자리다. 앞에 글이 있었다는
    # 이유로 문장 중간으로 세면 인용문 첫 단어가 보통 명사인데도 이름이 된다.
    [straight] = canonicals(['Analysts said, "Customers pay monthly."'])
    assert "customers" not in straight
    [curly] = canonicals(["The CEO said “Users subscribe online.”"])
    assert "users" not in curly


@pytest.mark.parametrize("space", [" ", " "])
def test_opening_quote_starts_a_sentence_across_whitespace(space):
    # 크롤한 본문은 여는 따옴표와 첫 낱말 사이에 공백을 두는 일이 흔하다
    # (줄바꿈 없는 공백도 마찬가지). 붙어 있을 때만 여는 것으로 보면 인용문
    # 첫 단어가 문장 중간 대문자로 위장해 배치 문서빈도를 오염시킨다.
    [names] = canonicals([f"Analysts said,{space}“{space}Customers pay monthly.”"])
    assert "customers" not in names


def test_closing_straight_quote_still_does_not_open_a_sentence():
    # 반대 방향 경계. 곧은 따옴표는 여닫는 모양이 같아서 공백까지 허용하면
    # 닫는 자리가 열림으로 세어져 뒤의 진짜 이름이 통째로 탈락한다.
    [names] = canonicals(['Reviewers said "hello" Anthropic shipped it fast.'])
    assert "anthropic" in names


def test_name_connectors_keep_the_organization_whole():
    # 'Bank of America'의 of에서 끊으면 회사 하나가 사라지고 조각 둘이 남는다.
    [names] = canonicals(["Analysts covered Bank of America closely this quarter."])
    assert "bank of america" in names
    assert "bank" not in names
    assert "america" not in names


@pytest.mark.parametrize(
    "sentence, name",
    [
        ("Reviewers interviewed José de la Cruz yesterday.", "josé de la cruz"),
        ("Reviewers cited Ludwig van der Waals yesterday.", "ludwig van der waals"),
    ],
)
def test_consecutive_connectors_keep_the_person_whole(sentence, name):
    # 이음말이 둘 이어지는 이름. 하나만 볼 수 있으면 'de' 뒤의 'la'가 이름이
    # 아니라서 거기서 끊기고, 사람 하나가 조각 둘로 남는다.
    [names] = canonicals([sentence])
    assert name in names


@pytest.mark.parametrize(
    "sentence, name",
    [
        ("Reviewers visited University of the Arts yesterday.", "university of the arts"),
        ("Reviewers watched House of the Dragon yesterday.", "house of the dragon"),
    ],
)
def test_an_article_inside_a_name_does_not_break_it(sentence, name):
    # 'of the'는 이름 안에서 흔하다. 관사에서 끊으면 이름 하나가 사라지고
    # 조각 둘이 문서빈도 자리를 차지한다.
    [names] = canonicals([sentence])
    assert name in names


def test_a_connector_chain_ending_in_a_common_noun_still_breaks():
    # 반대 방향 경계. 관사를 받되 연쇄 끝이 보통 명사면 문장의 전치사다.
    [names] = canonicals(["Analysts watched Acme Corp of the year for months."])
    assert "acme corp" in names
    assert not any(key.startswith("acme corp of") for key in names)


def test_a_connector_chain_without_a_name_still_breaks():
    # 반대 방향 경계. 이음말 연쇄 끝에 이름이 없으면 그냥 문장의 전치사다.
    [names] = canonicals(["Analysts said Acme Corp de la mode sells shoes daily."])
    assert "acme corp" in names
    assert not any(key.startswith("acme corp de") for key in names)


@pytest.mark.parametrize(
    "sentence, name",
    [
        ("Reviewers called “Claude” promising yesterday.", "claude"),
        ("Reviewers discussed «Anthropic» yesterday.", "anthropic"),
        ('Reviewers called "Claude" promising yesterday.', "claude"),
    ],
)
def test_emphasis_quotes_do_not_hide_the_name_inside(sentence, name):
    # 따옴표로 감싼 이름 하나는 인용문이 아니라 강조다. 문장 첫 자리로 세면
    # 한 낱말짜리라 배치에 다른 언급이 없는 한 통째로 탈락한다.
    [names] = canonicals([sentence])
    assert name in names


@pytest.mark.parametrize(
    "sentence, absent",
    [
        ("Analysts said: Customers pay monthly.", "customers"),
        ("The report found — Users prefer lower prices.", "users"),
    ],
)
def test_a_clause_introducer_starts_a_sentence(sentence, absent):
    # 쌍점·줄표 뒤도 새 절의 첫 자리다. 아니라고 보면 절 첫 단어인 보통 명사가
    # 이름 행세를 하며 배치 문서빈도 자리를 차지한다.
    [names] = canonicals([sentence])
    assert absent not in names


def test_a_name_after_a_colon_is_still_a_name():
    # 반대 방향 경계. 쌍점 뒤에 이름 하나만 딸린 건 절이 아니다 — 절로 보면
    # 'Report: Anthropic' 꼴의 제목에서 회사 이름이 통째로 탈락한다.
    [names] = canonicals(["Analysts published the note: Anthropic."])
    assert "anthropic" in names


def test_a_bracketed_clause_starts_a_sentence():
    # 대괄호로 끼워 넣은 편집자 주도 새 문장이다. 아니라고 보면 그 안 첫 단어인
    # 보통 명사가 이름 행세를 한다.
    [names] = canonicals(["Analysts said [Customers pay monthly.]"])
    assert "customers" not in names


def test_a_bracketed_name_is_not_a_sentence():
    # 반대 방향 경계. 대괄호 안의 이름 하나는 문장이 아니다.
    [names] = canonicals(["Analysts covered Acme Corp [Globex] closely this quarter."])
    assert "globex" in names


def test_a_parenthesized_clause_starts_a_sentence():
    # 문장 중간에 괄호로 문장이 들어오면 그 안은 새 문장의 첫 자리다. 아니라고
    # 보면 괄호 안 첫 단어인 보통 명사가 이름 행세를 한다.
    [names] = canonicals(["Analysts said, (Customers pay monthly.)"])
    assert "customers" not in names


def test_a_parenthesized_name_is_not_a_sentence():
    # 반대 방향 경계. 문장 중간 괄호 안에는 문장보다 이름이 훨씬 흔하다 —
    # 첫 자리로 세면 그 이름이 통째로 탈락한다.
    [names] = canonicals(["Analysts covered Acme Corp (Globex) closely this quarter."])
    assert "globex" in names


def test_listing_conjunction_still_separates_names():
    # 반대 방향 경계. 'and'는 이름 안의 이음말이 아니라 나열 기호다 — 이어 붙이면
    # 서로 다른 회사 둘이 사라지고 없는 이름 하나가 생긴다.
    [names] = canonicals(["Reviewers compared Anthropic and Google this quarter."])
    assert {"anthropic", "google"} <= names
    assert "anthropic and google" not in names


@pytest.mark.parametrize(
    "document,bogus",
    [
        ("Analysts said I agree with the plan.", "i"),
        ("Analysts met in March to review the plan.", "march"),
    ],
)
def test_intrinsically_capitalized_words_are_not_names(document, bogus):
    # 문장 중간에서도 대문자로 쓰는 말들이다. 문장 첫 단어 규칙이 못 걸러내므로
    # 그냥 두면 없는 이름이 배치 문서빈도 자리를 차지한다.
    [names] = canonicals([document])
    assert bogus not in names


def test_fullwidth_version_keeps_its_decimal_point():
    # 전각 마침표를 문장 끝으로 옮길 때 숫자 사이까지 옮기면 버전이 거기서
    # 끊긴다 — 반각으로 쓴 같은 제품과 다른 키가 되어 문서빈도가 쪼개진다.
    [names] = canonicals(["Engineers benchmarked ＧＰＴ－５．２ against rivals."])
    assert "gpt 5.2" in names


@pytest.mark.parametrize(
    "sentence, name",
    [
        ("Reviewers compared Ｎｏｄｅ．ｊｓ with Deno.", "node js"),
        ("Reviewers compared ＡＳＰ．ＮＥＴ with Deno.", "asp net"),
    ],
)
def test_fullwidth_dot_inside_a_name_is_not_a_sentence_end(sentence, name):
    # 이름 가운데의 전각 마침표를 문장 끝으로 옮기면 제품 이름이 거기서 잘려
    # 앞 토막만 남는다 — 반각으로 쓴 같은 제품과 다른 키가 된다.
    [names] = canonicals([sentence])
    assert name in names


def test_fullwidth_dot_after_a_digit_stays_inside_the_name():
    # '３．ｊｓ'의 점은 이름 안이다. 문장 끝으로 옮기면 제품이 통째로 사라진다 —
    # 반각으로 쓴 '3.js'는 '3js'로 잡히던 터라 표기에 따라 결과가 갈렸다.
    [names] = canonicals(["Reviewers compared ３．ｊｓ with Deno."])
    assert "3js" in names


def test_a_fullwidth_list_number_still_closes_the_line():
    # 반대 방향 경계. CJK 목록은 '３．'처럼 공백 없이 번호를 매긴다 — 이름 안의
    # 점으로 보면 번호가 첫 낱말과 붙어 보통 명사가 이름 행세를 한다.
    [names] = canonicals(["３．Customers pay monthly for the tier."])
    assert "customers" not in names


def test_leading_dot_stays_in_the_surface():
    # 표시용 원문은 본 그대로여야 한다. 앞의 점을 떼면 '.NET'이 'NET'으로 보이고,
    # 약어 'NET'과 구별할 방법도 사라진다.
    [names] = extract_names(["Reviewers compared .NET with Java yesterday."])
    assert ".NET" in {name.surface for name in names}
    assert ".net" in {name.canonical for name in names}


def test_lowercase_digit_leading_names_are_kept():
    # '3.js' '4chan' '500px'는 대문자가 하나도 없다. 첫 글자 대문자만 보면
    # 통째로 사라지는데, 하필 이 프로젝트가 쫓는 이름들이 그 모양이다.
    [names] = canonicals(["Reviewers compared 3.js with 4chan and 500px daily."])
    assert {"3js", "4chan", "500px"} <= names


def test_ordinals_are_not_names():
    # 숫자머리를 받되 서수는 이름이 아니다. 받아 주면 'the 3rd quarter'의 '3rd'가
    # 배치 문서빈도 자리를 차지한다.
    [names] = canonicals(["Reviewers compared revenue in the 3rd quarter with Java."])
    assert "3rd" not in names


def test_an_ellipsis_tail_is_not_a_leading_dot():
    # 'slipped...Next'의 마지막 점은 줄임표의 꼬리지 '.NET'의 앞점이 아니다.
    # 붙잡으면 없는 이름 '.next'가 생기고, 근접중복 쪽 정규화와도 어긋난다.
    [names] = canonicals(["The rollout slipped...Next quarter the team revisits it."])
    assert ".next" not in names


def test_a_year_does_not_join_the_name():
    # 이름 뒤의 연도는 버전이 아니다. 붙이면 같은 회사가 'nvidia'와 'nvidia
    # 2025' 두 키로 쪼개져 문서빈도가 갈린다.
    [names] = canonicals(["Analysts reviewed Nvidia 2025 revenue forecasts."])
    assert "nvidia" in names
    assert "nvidia 2025" not in names


def test_a_product_number_still_joins_the_name():
    # 반대 방향 경계. 네 자리라고 다 연도는 아니다 — 제품 번호는 붙어야 한다.
    [names] = canonicals(["Reviewers benchmarked RTX 4090 cards last week."])
    assert "rtx 4090" in names


def test_leading_dot_survives_straight_quotes():
    # 곧은 따옴표로 감싼 이름도 앞점을 잃으면 안 된다 — 표시용 원문이 'NET'이
    # 되고 약어 'NET'과 구별할 방법도 사라진다. 활자 따옴표는 이미 잡힌다.
    [names] = extract_names(['Reviewers compared ".NET" with Java yesterday.'])
    assert ".NET" in {name.surface for name in names}
    assert ".net" in {name.canonical for name in names}


def test_fullwidth_dot_before_an_uppercase_version_stays_in_the_name():
    # 'ＧＰＴ－３．Ｘ'의 점은 이름 안이다 — 앞 숫자가 이미 이름에 붙어 있다.
    # 문장 끝으로 옮기면 반각 'GPT-3.X'(gpt 3x)와 다른 키가 된다.
    [names] = canonicals(["Reviewers compared ＧＰＴ－３．Ｘ with Deno."])
    assert "gpt 3x" in names


@pytest.mark.parametrize(
    "sentence, name",
    [
        ("Reviewers discussed Acme, Inc. yesterday.", "acme inc"),
        ("Reviewers discussed Foo Co., Ltd. yesterday.", "foo co ltd"),
        # 점을 찍어 쓰는 꼬리도 같은 회사다 — 'Acme S.A.'와 한 키여야 한다.
        ("Reviewers discussed Acme, S.A. yesterday.", "acme sa"),
        ("Reviewers discussed Acme, L.L.C. yesterday.", "acme llc"),
        # 크롤한 표기가 쉼표 뒤를 안 띄우기도 한다. 그래도 같은 회사다.
        ("Reviewers discussed Acme,Inc. yesterday.", "acme inc"),
        ("Reviewers discussed Acme,S.A. yesterday.", "acme sa"),
    ],
)
def test_a_comma_before_a_corporate_suffix_stays_in_the_name(sentence, name):
    # 회사 이름 안의 쉼표는 나열 기호가 아니다. 끊으면 회사 하나가 조각 둘로
    # 남고, 쉼표 없이 쓴 'Acme Inc.'와 다른 키가 된다.
    [names] = canonicals([sentence])
    assert name in names


def test_pure_numbers_stay_numbers():
    # 숫자로 시작하는 이름을 받되 순수한 숫자는 그대로 숫자여야 한다 — 버전
    # 숫자가 이름에 붙고, 항목 번호는 번호로 남는다.
    [versioned] = canonicals(["Anthropic today shipped Claude Sonnet 5 to the API."])
    assert "claude sonnet 5" in versioned
    [numbered] = canonicals(["2.1) Customers pay monthly for the tier."])
    assert "customers" not in numbered


def test_combining_mark_without_a_precomposed_form_keeps_the_name_whole():
    # NFC로도 자음에 합성되지 않는 결합 기호가 있다. 낱말이 거기서 끊기면 이름이
    # 두 동강 나고 한 글자짜리 가짜 이름이 남는다 — 정규형 쪽은 결합 기호를 이미
    # 남기므로 두 경로의 답이 갈린다.
    given = unicodedata.normalize("NFC", "Ọ́lá")
    [names] = canonicals([f"Reviewers quoted {given} Brown on the benchmark."])
    assert canonical_name(f"{given} Brown") in names
    assert canonical_name(given[0]) not in names


def test_accented_name_stays_one_name():
    # 악센트 글자에서 낱말이 끊기면 이름이 조각난다 — "François"가 "Fran"과
    # "ois"로 갈라지고, 남은 "Fran"은 사람 이름 행세를 한다.
    [names] = canonicals(["Reviewers quoted François Chollet on the benchmark."])
    assert "françois chollet" in names
    assert "fran" not in names
    assert "chollet" not in names


def test_vietnamese_name_stays_one_name():
    # 이름 글자 범위를 블록 열거로 정하면 빠뜨린 블록에서 이름이 조각난다.
    # 베트남어 성조 부호(ễ)는 라틴 확장 추가 블록에 있다.
    [names] = canonicals(["Reviewers quoted Nguyễn Văn An on the benchmark."])
    assert "nguyễn văn an" in names
    assert "nguy" not in names


def test_line_break_starts_a_new_sentence():
    # 크롤한 본문은 소제목 뒤에 마침표 없이 줄만 바뀐다. 한 문장으로 붙여 두면
    # 문단 첫 단어가 '문장 중간 대문자'로 위장해 문장 첫 단어 필터를 통과한다.
    [names] = canonicals(["Pricing details\nCustomers pay monthly for the tier."])
    assert "customers" not in names
    assert not any("customers" in name for name in names)


@pytest.mark.parametrize(
    "sentence, name",
    [
        ("Reviewers visited St. Louis with Acme Corp.", "st louis"),
        ("Reviewers climbed Mt. Fuji with Acme Corp.", "mt fuji"),
    ],
)
def test_a_place_abbreviation_stays_with_the_name(sentence, name):
    # 'St. Louis'의 St는 호칭이 아니라 이름의 일부다. 점에서 끊으면 도시 하나가
    # 'louis'로 남고, spaCy가 통째로 넘기는 'st louis'와 갈려 두 키가 된다.
    [names] = canonicals([sentence])
    assert name in names


def test_a_title_before_a_name_does_not_end_the_sentence():
    # 'Dr.'의 점을 문장 끝으로 읽으면 뒤따르는 진짜 이름이 문장 첫 단어로
    # 둔갑해 탈락하고, 정작 호칭만 이름 행세를 하며 남는다.
    [names] = canonicals(["Reviewers quoted Dr. Smith on the benchmark."])
    assert "smith" in names
    assert "dr" not in names


def test_slash_separated_technology_names_stay_whole():
    # 'HTTP/2'·'TCP/IP'는 이름 하나다. 빗금에서 끊으면 버전 숫자가 떨어져 나가고
    # 한 이름이 둘로 쪼개진다 — 정규형 쪽은 이미 빗금을 구분자로 접는다.
    [names] = canonicals(["Reviewers tested HTTP/2 with TCP/IP."])
    assert {"http 2", "tcp ip"} <= names
    assert "http" not in names


def test_spaced_initials_stay_with_the_surname():
    # 'J. K. Rowling'처럼 띄어 쓴 머리글자에서 끊으면 성이 떨어져 나간다.
    # 통째로 묶여야 spaCy가 같은 이름에 붙이는 키('j k rowling')와 만난다.
    [names] = canonicals(["Reviewers quoted J. K. Rowling yesterday."])
    assert "j k rowling" in names


def test_an_initial_before_a_name_does_not_end_the_sentence():
    # 머리글자 약어도 마찬가지다. 'U.S.'에서 끊으면 'Army'가 사라진다.
    [names] = canonicals(["Officials briefed the U.S. Army about deployment."])
    assert "army" in names


def test_a_sentence_ending_in_a_single_letter_still_ends():
    # 머리글자 예외를 '한 글자 낱말' 전부로 넓히면 반대로 망가진다. 'option A.'
    # 처럼 평범하게 끝난 문장이 다음 문장과 붙어, 다음 문장 첫 단어가 문장 중간
    # 대문자로 위장한다.
    [names] = canonicals(["We selected option A. Customers pay monthly."])
    assert "customers" not in names


def test_leading_initial_stays_with_the_personal_name():
    # 'W. Somerset Maugham'의 W는 이름의 첫머리다. 끊으면 성만 남고 한 글자짜리
    # 가짜 이름이 하나 더 생긴다. 앞 낱말('discussed')은 소문자라 근거가 못 되고,
    # 뒤에 대문자 낱말이 **둘** 이어지는 것이 'option A. Customers pay'와
    # 갈라내는 단서다.
    [names] = canonicals(["Reviewers discussed W. Somerset Maugham yesterday."])
    assert "w somerset maugham" in names
    assert "w" not in names


def test_hierarchical_list_numbers_are_list_markers():
    # '2.1)'은 숫자 한 덩어리가 아니라서 항목 번호로 안 걸렸다. 번호가 내용으로
    # 세어지면 목록 첫 단어가 문장 첫 단어 필터를 그대로 통과한다.
    [names] = canonicals(["2.1) Customers pay monthly for the tier."])
    assert "customers" not in names


def test_ellipsis_ends_a_sentence():
    # 활자 줄임표는 크롤한 산문에 흔하다. 문장 끝으로 보지 않으면 다음 문장 첫
    # 단어가 문장 중간 대문자로 위장한다.
    [names] = canonicals(["Prices rose… Customers pay monthly."])
    assert "customers" not in names


def test_typographic_closing_quote_ends_a_sentence():
    # 크롤한 본문의 인용은 곧은 따옴표가 아니라 활자 따옴표로 닫힌다. 닫는
    # 따옴표를 못 알아보면 문장이 안 끊겨 다음 문장 첫 단어가 '문장 중간
    # 대문자'로 위장하고 보통 명사가 이름 행세를 한다.
    [names] = canonicals(["“Customers pay monthly.” Users subscribe online."])
    assert "users" not in names


@pytest.mark.parametrize("dash", ["-", "‑", "–"])
def test_unicode_dashes_stay_inside_versioned_names(dash):
    # HTML 본문은 'GPT‑5'를 줄바꿈 없는 붙임표(U+2011)나 반각 줄표(U+2013)로
    # 적는 일이 흔하다. 거기서 낱말을 끊으면 버전 숫자가 떨어져 나가 'gpt'만
    # 남는다 — 정규형 쪽은 이미 이 부호들을 붙임표와 같게 접는다.
    [names] = canonicals([f"Engineers benchmarked GPT{dash}5 against rivals."])
    assert "gpt 5" in names


def test_em_dash_still_separates_names():
    # 전각 줄표는 이름 안이 아니라 절 사이에 쓴다. 붙임표류와 같이 묶으면
    # 서로 다른 이름 둘이 'claude anthropic'이라는 없는 이름 하나로 붙는다.
    [names] = canonicals(["Reviewers praised Claude—Anthropic shipped it fast."])
    assert {"claude", "anthropic"} <= names
    assert "claude anthropic" not in names


def test_latin_name_after_korean_text_is_mid_sentence():
    # 문장 첫 단어 여부를 토큰 순번으로 보면, 이름 글자가 아닌 앞부분이 통째로
    # 없던 일이 된다 — 한글 뒤에 나온 이름이 문장 첫 단어로 둔갑해 탈락한다.
    [names] = canonicals(["연구진은 Anthropic과 협력했다."])
    assert "anthropic" in names


def test_possessive_does_not_swallow_the_next_name():
    # 소유격은 소유자에서 이름이 끝난다. 이어 붙이면 'anthropics claude'라는
    # 없는 이름이 되고, 진짜 이름 둘은 어디에도 나오지 않는다.
    [names] = canonicals(["Reviewers tested Anthropic's Claude against GPT-5.2."])
    assert {"anthropic", "claude", "gpt 5.2"} <= names
    assert "anthropics claude" not in names


def test_decomposed_accents_are_normalized_before_tokenizing():
    # 크롤한 글이 NFC라는 보장이 없다. 결합 기호가 분리된 채로 오면 토큰이
    # 거기서 끊겨 'franc' 같은 조각이 나온다.
    decomposed = unicodedata.normalize(
        "NFD", "Reviewers quoted François Chollet on the benchmark."
    )
    [names] = canonicals([decomposed])
    assert "françois chollet" in names
    assert "franc" not in names


@pytest.mark.parametrize(
    "document",
    ['"Customers pay monthly."', "(Customers pay monthly.)", "- Customers pay monthly."],
)
def test_opening_punctuation_does_not_hide_a_sentence_start(document):
    # 따옴표·괄호·목록 기호가 앞에 있어도 그 단어는 여전히 문장의 첫 단어다.
    # 앞에 뭐라도 있으면 문장 중간으로 치면, 인용문과 목록에서 보통 명사가
    # 이름 행세를 한다.
    [names] = canonicals([document])
    assert "customers" not in names


@pytest.mark.parametrize(
    "document",
    ["1) Customers pay monthly.", "[1] Customers pay monthly.", "(a) Customers pay monthly."],
)
def test_list_enumerator_does_not_hide_a_sentence_start(document):
    # 항목 번호는 문장 내용이 아니라 여는 표시다. 이걸 '앞선 내용'으로 세면
    # 목록 첫 단어가 문장 첫 단어 필터를 그대로 통과한다.
    [names] = canonicals([document])
    assert "customers" not in names


def test_ampersand_joins_a_name_instead_of_splitting_it():
    # '&'는 이름을 가르는 구분자가 아니라 붙이는 이음말이다. 여기서 끊으면
    # 회사 하나가 사라지고 한 글자짜리 가짜 이름이 생긴다.
    [names] = canonicals(["Reviewers compared AT&T with Verizon this quarter."])
    assert "at t" in names
    assert "t" not in names
    assert "verizon" in names


def test_spaced_ampersand_keeps_one_company_name():
    [names] = canonicals(["Analysts covered Johnson & Johnson closely this quarter."])
    assert "johnson johnson" in names


@pytest.mark.parametrize(
    "document",
    [
        "背景です。Customers pay monthly.",
        "背景です！Customers pay monthly.",
        "背景です？Customers pay monthly.",
        # 전각 마침표는 NFKC가 ASCII 마침표로 접는다. ASCII 마침표는 뒤에 공백을
        # 요구하므로, 접히기 전에 옮겨 두지 않으면 이 경계가 사라진다.
        "背景です．Customers pay monthly.",
    ],
)
def test_unicode_terminator_starts_a_new_sentence(document):
    # 일본어·중국어 본문의 문장 끝은 ASCII 마침표가 아니다. 게다가 뒤에 공백도
    # 없어서, 종결부호로 안 쳐 주면 다음 문장 첫 단어가 문장 중간으로 위장한다.
    [names] = canonicals([document])
    assert "customers" not in names


@pytest.mark.parametrize(
    "document",
    ["1: Customers pay monthly.", "1 - Customers pay monthly.", "(ii) Customers pay monthly."],
)
def test_more_list_enumerator_forms_do_not_hide_a_sentence_start(document):
    [names] = canonicals([document])
    assert "customers" not in names


def test_ampersand_is_kept_in_the_display_surface():
    # 이음말을 빼고 낱말만 이어 붙이면 표시용 원문이 'AT T'로 망가진다.
    [names] = extract_names(["Reviewers compared AT&T with Verizon this quarter."])
    by_key = {name.canonical: name for name in names}
    assert "at t" in by_key
    assert by_key["at t"].surface == "AT&T"


def test_korean_particle_does_not_stick_to_a_latin_name():
    # 이름 글자 범위를 넓힐 때 대소문자 없는 글자까지 넣으면 조사가 이름에
    # 붙어 버린다 — "Claude를"은 어느 배치에서도 "Claude"와 같은 이름이 아니다.
    [names] = canonicals(["앤트로픽은 Claude Sonnet 5와 Claude를 함께 갱신했다."])
    assert {"claude sonnet 5", "claude"} <= names
    assert not any(name.endswith("를") for name in names)


def test_single_letter_name_before_a_colon_is_not_a_list_marker():
    # 'X:'는 항목 번호가 아니라 회사 이름이다. 목록 기호로 오인해 건너뛰면
    # 한 글자짜리 진짜 이름이 결과 어디에도 남지 않는다.
    first, second = canonicals(
        ["X: Grok 5 ships new features today.", "Analysts said X keeps shipping."]
    )
    assert "x" in first
    assert "x" in second


def test_symbol_suffixed_technology_names_stay_distinct():
    # 낱말 끝에 붙은 '+'·'#'을 버리면 'C++'와 'C#'이 둘 다 'C' 한 글자로
    # 잘려 하나로 합쳐진다 — 서로 다른 기술 둘이 사라지고 없는 이름이 생긴다.
    [names] = canonicals(["Reviewers compared C++ with C# yesterday."])
    assert {"c++", "c#"} <= names
    assert "c" not in names


@pytest.mark.parametrize("documents", [[], [""], ["   "], ["...  ---  "], ["한글만 있는 문장이다."]])
def test_degenerate_input_yields_no_names(documents):
    assert all(names == set() for names in canonicals(documents))
