from pathlib import Path

import pytest
from pydantic import ValidationError

from argos.config import EventDetectionConfig, UserConfig


def test_defaults_match_design_decision():
    cfg = EventDetectionConfig()
    assert cfg.simhash_hamming_max == 3
    assert cfg.simhash_shingle_size == 4
    assert cfg.entity_max_ngram == 4
    assert cfg.entity_min_doc_count == 1
    assert cfg.entity_max_doc_ratio == pytest.approx(0.5)
    assert cfg.entity_df_min_batch == 5
    assert cfg.entity_spacy_model == "en_core_web_sm"


def test_user_config_exposes_section_with_defaults():
    # 섹션을 아예 안 쓴 config.toml도 기본값으로 채워져야 한다.
    user = UserConfig.model_validate({})
    assert user.event_detection.simhash_hamming_max == 3


def test_partial_section_keeps_other_defaults():
    user = UserConfig.model_validate({"event_detection": {"simhash_hamming_max": 6}})
    assert user.event_detection.simhash_hamming_max == 6
    assert user.event_detection.simhash_shingle_size == 4


@pytest.mark.parametrize(
    "payload",
    [
        {"simhash_hamming_max": -1},
        {"simhash_hamming_max": 65},
        {"simhash_shingle_size": 0},
        {"entity_max_ngram": 0},
        {"entity_max_ngram": 9},
        {"entity_min_doc_count": 0},
        {"entity_max_doc_ratio": 0.0},
        {"entity_max_doc_ratio": 1.5},
        {"entity_df_min_batch": 0},
    ],
)
def test_out_of_range_values_rejected(payload):
    with pytest.raises(ValidationError):
        EventDetectionConfig(**payload)


def test_edge_weights_default_to_the_confirmed_design():
    config = EventDetectionConfig()
    assert config.weight_cosine == 0.55
    assert config.weight_entity == 0.25
    assert config.weight_time == 0.15
    assert config.weight_keyword == 0.05


def test_join_threshold_has_a_default_and_a_range():
    assert EventDetectionConfig().join_threshold == 0.55
    with pytest.raises(ValidationError):
        EventDetectionConfig(join_threshold=1.5)
    with pytest.raises(ValidationError):
        EventDetectionConfig(join_threshold=-0.1)


def test_negative_weights_are_rejected():
    with pytest.raises(ValidationError):
        EventDetectionConfig(weight_cosine=-0.1)


def test_loads_from_toml_file(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[event_detection]\nsimhash_hamming_max = 5\nentity_max_doc_ratio = 0.8\n",
        encoding="utf-8",
    )
    user = UserConfig.load(path)
    assert user.event_detection.simhash_hamming_max == 5
    assert user.event_detection.entity_max_doc_ratio == pytest.approx(0.8)
    # 지정하지 않은 항목은 기본값 유지
    assert user.event_detection.entity_max_ngram == 4


def test_candidate_window_and_k_have_defaults():
    config = EventDetectionConfig()
    assert config.window_days == 14
    assert config.candidate_k == 25


def test_window_and_k_reject_nonsense():
    with pytest.raises(ValidationError):
        EventDetectionConfig(window_days=0)
    with pytest.raises(ValidationError):
        EventDetectionConfig(candidate_k=0)


def test_leiden_knobs_have_defaults_and_ranges():
    config = EventDetectionConfig()
    assert config.leiden_objective == "cpm"
    # 해상도는 값을 복사해 박아 두지 않는다 — 비어 있는 게 기본이다.
    assert config.leiden_resolution is None
    assert config.leiden_seed == 42

    with pytest.raises(ValidationError):
        EventDetectionConfig(leiden_objective="louvain")
    with pytest.raises(ValidationError):
        EventDetectionConfig(leiden_resolution=-0.1)
    with pytest.raises(ValidationError):
        EventDetectionConfig(leiden_seed=-1)


def test_leiden_resolution_follows_join_threshold_unless_overridden():
    # 두 값이 프로즈로만 묶여 있으면 join_threshold만 내렸을 때 조용히
    # 어긋난다(γ가 채택된 간선보다 높아 오히려 더 잘게 쪼개진다). 실효값
    # 해석은 config 한 곳에만 있어야 코어와 CLI 리포트가 같은 숫자를 본다.
    assert EventDetectionConfig().effective_leiden_resolution == pytest.approx(0.55)
    assert EventDetectionConfig(
        join_threshold=0.35
    ).effective_leiden_resolution == pytest.approx(0.35)
    # 명시 오버라이드는 그대로 살아 있다 — 조율할 여지를 없애지는 않았다.
    assert EventDetectionConfig(
        join_threshold=0.35, leiden_resolution=0.9
    ).effective_leiden_resolution == pytest.approx(0.9)
