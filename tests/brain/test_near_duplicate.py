import unicodedata

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


# 한글은 글자당 정보량이 라틴 문자보다 훨씬 크다. 같은 "글자 수"라도 훨씬 짧은
# 기사이므로, 영문 픽스처와 비교 가능한 분량이 되도록 본문을 잡는다.
_KO_BODY = (
    "회사는 새 모델이 긴 문맥에서 코딩 정확도를 높였다고 밝혔다. "
    "내부 평가에서 이전 세대보다 점수가 올랐으며, 기존 고객의 가격은 그대로다. "
    "일부 연구자는 평가 세트가 공개되지 않았다며 학습 데이터에 대한 설명을 요구했다. "
    "회사는 기술 보고서를 이번 분기 안에 내겠다고 답했다. "
    "업계는 이번 발표가 가격 경쟁을 다시 촉발할지 주시하고 있다. "
    "경쟁사는 다음 달 자체 모델을 내놓겠다고 예고한 상태다. "
    "국내 개발자들은 한국어 처리 성능이 실제로 개선됐는지 확인하겠다는 반응이다. "
    "회사 측은 별도의 벤치마크 결과를 공개하지 않았다. "
    "엔터프라이즈 고객은 확장 문맥 등급에 우선 접근을 신청할 수 있다. "
    "이 등급은 백만 토큰 규모의 입력을 처리한다고 회사는 설명했다. "
    "가격 정책 변경은 기존 계약에는 소급 적용되지 않는다. "
    "회사는 다음 분기에 별도의 기술 문서를 공개하겠다고 덧붙였다."
)
KO_ARTICLE = "새 코딩 모델 공개. " + _KO_BODY
KO_RETITLED = "개발자 겨냥한 신규 공개. " + _KO_BODY
KO_UNRELATED = (
    "시의회가 강변 자전거 도로 설치를 승인했다. 공사는 봄에 시작해 여덟 달이 걸린다. "
    "인근 상인들은 계획을 반겼지만 주차 공간을 더 달라고 요청했다. "
    "예산은 광역 교통 기금에서 나오며 지방세를 올리지 않는다."
)


def test_non_latin_articles_are_not_all_the_same_article():
    # 라틴 문자만 남기고 버리면 한국어 기사는 전부 빈 글이 되어 SimHash 0으로
    # 뭉개지고, 서로 무관한 기사끼리 같은 기사로 판정된다.
    assert simhash(KO_ARTICLE) != 0
    assert is_near_duplicate(KO_ARTICLE, KO_UNRELATED) is False


def test_non_latin_retitled_article_is_near_duplicate():
    assert is_near_duplicate(KO_ARTICLE, KO_RETITLED) is True


def test_empty_text_is_handled():
    assert simhash("") == 0
    assert is_near_duplicate("", "") is True


def test_symbol_only_texts_are_not_all_the_same_article():
    # 글자·숫자가 하나도 없는 글(렌더 전에 긁힌 자리표시자 페이지 등)을 전부
    # SimHash 0으로 뭉개면, 서로 무관한 것끼리 같은 기사로 판정된다.
    left, right = "!!! ---- ???", "... === +++"
    assert simhash(left) != 0
    assert simhash(left) != simhash(right)
    assert is_near_duplicate(left, right) is False


@pytest.mark.parametrize(
    "body",
    [
        "François Chollet reviewed the release notes carefully. ",
        "앤트로픽이 새 모델을 공개했다. 성능이 크게 올랐다고 밝혔다. ",
    ],
)
def test_encoding_variants_are_the_same_article(body):
    # 같은 글을 NFC로 저장한 곳과 NFD로 저장한 곳에서 각각 긁어 오는 일이 있다.
    # 바이트 표현만 다른 같은 글이 다른 기사로 판정되면 안 된다.
    article = body * 10
    composed = unicodedata.normalize("NFC", article)
    decomposed = unicodedata.normalize("NFD", article)
    assert composed != decomposed
    assert simhash(composed) == simhash(decomposed)
    assert is_near_duplicate(composed, decomposed) is True


@pytest.mark.parametrize("a,b,expected", [(0, 0, 0), (0b1011, 0b1001, 1), (0, 2**64 - 1, 64)])
def test_hamming_distance(a, b, expected):
    assert hamming_distance(a, b) == expected


@pytest.mark.parametrize(
    "left,right",
    [
        ("कि खबर पढ़ें और समझें। ", "कु खबर पढ़ें और समझें। "),
        ("กิน ข้าว ทุก วัน อย่าง สม่ำเสมอ. ", "กุน ข้าว ทุก วัน อย่าง สม่ำเสมอ. "),
    ],
)
def test_combining_vowel_signs_keep_articles_apart(left, right):
    # 데바나가리·타이 문자는 모음이 자음에 붙는 결합 기호다. NFC가 합성해 주지
    # 않으므로 결합 기호를 지우면 모음이 통째로 사라져, 서로 다른 낱말로 쓰인
    # 두 기사가 같은 shingle을 내고 같은 기사로 판정된다.
    assert simhash(left * 10) != simhash(right * 10)
    assert is_near_duplicate(left * 10, right * 10) is False
