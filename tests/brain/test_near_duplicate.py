import pytest

from argos.config import EventDetectionConfig, settings
from argos.brain.near_duplicate import hamming_distance, is_near_duplicate, simhash

# 실제 기사 길이에 가까운 본문. 재배포 판정은 본문이 짧으면 제목 한 줄의
# 비중이 과대평가되므로 픽스처를 일부러 길게 잡는다.
_BODY = (
    "Anthropic said the new model improves coding accuracy across long "
    "context windows and reduces refusal rates on benign requests. "
    "The company reported a score of 82 percent on an internal agentic "
    "benchmark, up from 74 percent for the previous generation. "
    "Pricing stays unchanged for existing customers, and the model is "
    "available today through the API and the developer console. "
    "Enterprise customers can request early access to the extended "
    "context tier, which supports one million tokens of input. "
    "Independent researchers noted that the evaluation suite has not yet "
    "been published, and asked for more detail on the training data. "
    "The company said a technical report would follow later this quarter."
)

ARTICLE = "Anthropic ships a faster coding model. " + _BODY
# 같은 기사, 제목 문구만 교체
RETITLED = "New Anthropic release targets developers. " + _BODY
# 같은 기사, 문단 순서만 바꿈
_SENTENCES = [s.strip() for s in _BODY.split(". ") if s.strip()]
REORDERED = "Anthropic ships a faster coding model. " + ". ".join(
    _SENTENCES[3:] + _SENTENCES[:3]
)
UNRELATED = (
    "The city council approved a new bicycle lane along the riverfront "
    "after two years of public consultation. Construction begins in "
    "March and is expected to take eight months, with detours posted "
    "for drivers. Local shop owners welcomed the plan but asked for "
    "additional parking near the market square. The budget comes from "
    "a regional transport fund and does not raise municipal taxes."
)


def test_simhash_is_64_bit_int():
    value = simhash(ARTICLE)
    assert isinstance(value, int)
    assert 0 <= value < 2**64


def test_retitled_article_is_near_duplicate():
    assert is_near_duplicate(ARTICLE, RETITLED) is True


def test_paragraph_reorder_is_near_duplicate():
    assert is_near_duplicate(ARTICLE, REORDERED) is True


def test_unrelated_articles_are_not_near_duplicate():
    assert is_near_duplicate(ARTICLE, UNRELATED) is False


def test_identical_text_distance_is_zero():
    assert hamming_distance(simhash(ARTICLE), simhash(ARTICLE)) == 0


def test_threshold_comes_from_config(monkeypatch):
    # 거리 컷을 0으로 조이면 제목만 바뀐 기사도 더 이상 같은 기사가 아니다.
    strict = EventDetectionConfig(simhash_hamming_max=0)
    monkeypatch.setattr(settings.user, "event_detection", strict)
    assert is_near_duplicate(ARTICLE, RETITLED) is False

    # 컷을 크게 열면 무관한 기사도 같다고 판정된다 — 설정이 실제로 먹는다는 증거.
    loose = EventDetectionConfig(simhash_hamming_max=64)
    monkeypatch.setattr(settings.user, "event_detection", loose)
    assert is_near_duplicate(ARTICLE, UNRELATED) is True


def test_explicit_max_distance_overrides_config(monkeypatch):
    strict = EventDetectionConfig(simhash_hamming_max=0)
    monkeypatch.setattr(settings.user, "event_detection", strict)
    assert is_near_duplicate(ARTICLE, RETITLED, max_distance=64) is True


def test_deterministic_across_processes():
    # 시드 의존적인 내장 hash()를 쓰면 PYTHONHASHSEED에 따라 값이 흔들린다.
    import subprocess
    import sys

    code = (
        "from argos.brain.near_duplicate import simhash;"
        "print(simhash('Anthropic ships a faster coding model today'))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1")
    ]
    assert runs[0] == runs[1]
    assert runs[0] == str(simhash("Anthropic ships a faster coding model today"))


def test_empty_text_is_handled():
    assert simhash("") == 0
    assert is_near_duplicate("", "") is True


@pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (0b1011, 0b1001, 1), (0, 2**64 - 1, 64)])
def test_hamming_distance(a, b, expected):
    assert hamming_distance(a, b) == expected
