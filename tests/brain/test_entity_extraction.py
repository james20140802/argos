import pytest

from argos.brain.entity_extraction import ExtractedName, extract_names
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


def test_korean_particle_does_not_stick_to_a_latin_name():
    # 이름 글자 범위를 넓힐 때 대소문자 없는 글자까지 넣으면 조사가 이름에
    # 붙어 버린다 — "Claude를"은 어느 배치에서도 "Claude"와 같은 이름이 아니다.
    [names] = canonicals(["앤트로픽은 Claude Sonnet 5와 Claude를 함께 갱신했다."])
    assert {"claude sonnet 5", "claude"} <= names
    assert not any(name.endswith("를") for name in names)


@pytest.mark.parametrize("documents", [[], [""], ["   "], ["...  ---  "], ["한글만 있는 문장이다."]])
def test_degenerate_input_yields_no_names(documents):
    assert all(names == set() for names in canonicals(documents))
