"""spaCy 보조 경로와 미설치 폴백 — ARG-253.

실제 모델은 내려받지 않는다. 파이프라인 자리에 가짜를 끼워 병합 규칙을
고정하고, import 실패는 `sys.modules`로 흉내 낸다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

from argos.brain import entity_spacy
from argos.brain.entity_extraction import extract_names
from argos.config import EventDetectionConfig, settings


@dataclass(frozen=True)
class FakeEnt:
    text: str
    label_: str


@dataclass(frozen=True)
class FakeDoc:
    ents: tuple[FakeEnt, ...]


class FakePipeline:
    """문서 원문 -> (표면형, 라벨) 목록을 돌려주는 가짜 nlp.

    진짜 spaCy처럼 단건 호출과 배치 `.pipe` 둘 다 받는다. 넘겨받은 배치를
    기록해 두어서 호출 쪽이 배치 API를 실제로 쓰는지 확인할 수 있다.
    """

    def __init__(self, mapping):
        self.mapping = mapping
        self.batches: list[list[str]] = []

    def __call__(self, text):
        return FakeDoc(
            tuple(FakeEnt(t, label) for t, label in self.mapping.get(text, ()))
        )

    def pipe(self, texts):
        batch = list(texts)
        self.batches.append(batch)
        return [self(text) for text in batch]


def fake_pipeline(mapping):
    return FakePipeline(mapping)


@pytest.fixture(autouse=True)
def _clear_pipeline_cache():
    entity_spacy.load_pipeline.cache_clear()
    yield
    entity_spacy.load_pipeline.cache_clear()


def canonicals(documents):
    return [{name.canonical for name in doc} for doc in extract_names(documents)]


# ---------------------------------------------------------------- 미설치 폴백


def test_missing_spacy_returns_no_pipeline(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy", None)  # import spacy -> ImportError
    assert entity_spacy.load_pipeline() is None


def test_missing_model_returns_no_pipeline(monkeypatch):
    class FakeSpacy:
        @staticmethod
        def load(name):
            raise OSError(f"[E050] Can't find model '{name}'")

    monkeypatch.setitem(sys.modules, "spacy", FakeSpacy)
    assert entity_spacy.load_pipeline() is None


def test_extraction_still_works_without_spacy(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacy", None)
    [names] = canonicals(["Reviewers tested Claude Sonnet 5 all week."])
    assert "claude sonnet 5" in names


def test_spacy_names_are_empty_without_a_pipeline(monkeypatch):
    monkeypatch.setattr(entity_spacy, "load_pipeline", lambda: None)
    assert entity_spacy.spacy_names(["Anthropic said nothing.", ""]) == [[], []]


def test_documents_are_parsed_in_one_batch(monkeypatch):
    # 배치 API로 받아 놓고 문서마다 파이프라인을 따로 부르면 spaCy 내부 배치가
    # 통째로 놀게 된다. 기사 수백 건짜리 크롤 배치에서 그 차이가 그대로 난다.
    pipeline = fake_pipeline({})
    monkeypatch.setattr(entity_spacy, "load_pipeline", lambda: pipeline)
    documents = ["Anthropic said nothing.", "", "Reviewers measured GPT-5.2."]
    assert entity_spacy.spacy_names(documents) == [[], [], []]
    assert pipeline.batches == [documents]


# ------------------------------------------------------------------- 보강 경로


NGRAM_BLIND = "Anthropic declined to comment on the timeline."


def test_ngram_path_alone_misses_a_sentence_initial_name():
    # 보강이 필요한 이유를 먼저 고정한다: 문장 첫 단어로만 등장하는 이름은
    # 규칙 경로가 문장부호와 구별할 근거가 없어 버린다.
    [names] = canonicals([NGRAM_BLIND])
    assert "anthropic" not in names


def test_spacy_recovers_that_name(monkeypatch):
    monkeypatch.setattr(
        entity_spacy,
        "load_pipeline",
        lambda: fake_pipeline({NGRAM_BLIND: [("Anthropic", "ORG")]}),
    )
    [names] = canonicals([NGRAM_BLIND])
    assert "anthropic" in names


def test_irrelevant_entity_labels_are_ignored(monkeypatch):
    monkeypatch.setattr(
        entity_spacy,
        "load_pipeline",
        lambda: fake_pipeline(
            {NGRAM_BLIND: [("the timeline", "DATE"), ("42 percent", "PERCENT")]}
        ),
    )
    [names] = canonicals([NGRAM_BLIND])
    assert names.isdisjoint({"the timeline", "42 percent"})


def test_spacy_and_ngram_hits_fold_into_one_name(monkeypatch):
    text = "Reviewers tested Sonnet 5 all week."
    monkeypatch.setattr(
        entity_spacy,
        "load_pipeline",
        lambda: fake_pipeline({text: [("Sonnet-5", "PRODUCT")]}),
    )
    [names] = extract_names([text])
    keys = [name.canonical for name in names]
    assert keys.count("sonnet 5") == 1
    # 규칙 경로가 이미 잡은 이름의 표시형은 규칙 경로 것을 쓴다 — 주 경로가 주다.
    assert next(n for n in names if n.canonical == "sonnet 5").surface == "Sonnet 5"


def test_spacy_names_obey_the_document_frequency_cut(monkeypatch):
    monkeypatch.setattr(
        settings.user,
        "event_detection",
        EventDetectionConfig(entity_max_doc_ratio=0.5, entity_df_min_batch=5),
    )
    docs = [f"Acme Corp declined to comment on report {i}." for i in range(5)]
    monkeypatch.setattr(
        entity_spacy,
        "load_pipeline",
        lambda: fake_pipeline({doc: [("Acme Corp", "ORG")] for doc in docs}),
    )
    for names in canonicals(docs):
        assert "acme corp" not in names


def test_spacy_assist_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(
        settings.user, "event_detection", EventDetectionConfig(entity_spacy_enabled=False)
    )
    monkeypatch.setattr(
        entity_spacy,
        "load_pipeline",
        lambda: fake_pipeline({NGRAM_BLIND: [("Anthropic", "ORG")]}),
    )
    [names] = canonicals([NGRAM_BLIND])
    assert "anthropic" not in names


def test_merged_result_stays_sorted_and_deterministic(monkeypatch):
    text = "Reviewers compared Claude Sonnet 5 with GPT-5.2 directly."
    monkeypatch.setattr(
        entity_spacy,
        "load_pipeline",
        lambda: fake_pipeline({text: [("Acme Corp", "ORG"), ("Blackwell", "PRODUCT")]}),
    )
    [names] = extract_names([text])
    keys = [name.canonical for name in names]
    assert keys == sorted(keys)
    assert extract_names([text]) == extract_names([text])
