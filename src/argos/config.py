from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# RSS feed defaults — 5 AI-company mainstream blogs + 2 Reddit subs (Alpha)
_DEFAULT_RSS_FEEDS: list[dict[str, str]] = [
    {"url": "https://openai.com/blog/rss.xml", "category": "Mainstream"},
    {"url": "https://blog.google/technology/ai/rss/", "category": "Mainstream"},
    {"url": "https://ai.meta.com/blog/rss/", "category": "Mainstream"},
    {"url": "https://mistral.ai/rss", "category": "Mainstream"},
    {"url": "https://huggingface.co/blog/feed.xml", "category": "Mainstream"},
    {"url": "https://www.reddit.com/r/MachineLearning/.rss", "category": "Alpha"},
    {"url": "https://www.reddit.com/r/LocalLLaMA/.rss", "category": "Alpha"},
]

_DEFAULT_SPA_SOURCES: list[dict[str, Any]] = [
    {
        "listing_url": "https://www.anthropic.com/news",
        "category": "Mainstream",
        "link_pattern": r"^/news/[^/]+$",
        "base_url": "https://www.anthropic.com",
        "max_items": 10,
        "name": "anthropic",
    }
]

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-reuse-import]

logger = logging.getLogger(__name__)


def _resolve_env_file() -> Path | None:
    """Resolve the .env path without reading it yet.

    Resolution order:
    1. ``ARGOS_ENV_FILE`` environment variable (absolute escape hatch).
    2. XDG path (``${XDG_CONFIG_HOME:-~/.config}/argos/.env``) when it exists.
    3. Repo-root ``./.env`` (cwd-relative) when it exists — deprecated; emits
       a WARNING telling the user to run ``argos config migrate-env``.

    Returns ``None`` when no candidate file exists (pydantic-settings will then
    skip file loading and fall back to environment variables / defaults).
    """
    # 1. Explicit override — always wins, even if the path does not exist.
    env_file_override = os.environ.get("ARGOS_ENV_FILE")
    if env_file_override:
        return Path(env_file_override)

    # 2. XDG path.
    xdg = os.environ.get("XDG_CONFIG_HOME")
    xdg_base = Path(xdg) if xdg else Path.home() / ".config"
    xdg_path = xdg_base / "argos" / ".env"
    if xdg_path.exists():
        return xdg_path

    # 3. Deprecated cwd-relative .env (repo-root fallback).
    cwd_env = Path(".env")
    if cwd_env.exists():
        logger.warning(
            "Loading secrets from repo-root .env is deprecated — run "
            "`argos config migrate-env` to move it to %s",
            xdg_path,
        )
        return cwd_env

    return None


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,  # file resolution is handled in __init__
        env_file_encoding="utf-8",
        extra="ignore",
    )

    POSTGRES_USER: str = "argos"
    POSTGRES_PASSWORD: str = "argos_dev_password"
    POSTGRES_DB: str = "argos"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    SLACK_BOT_TOKEN: str = ""
    SLACK_APP_TOKEN: str = ""

    def __init__(self, **kwargs: Any) -> None:
        # When the caller explicitly passes ``_env_file`` (including
        # ``_env_file=None`` from tests), honour it without interference.
        if "_env_file" in kwargs:
            super().__init__(**kwargs)
            return
        super().__init__(_env_file=_resolve_env_file(), **kwargs)


class SlackConfig(BaseModel):
    channel_id: str = ""
    summary_language: str = "Korean"


class BriefingConfig(BaseModel):
    time: str = "07:00"
    weekdays: list[str] = Field(
        default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        min_length=1,
    )
    limit_per_category: int = Field(default=10, ge=1)
    # ARG-132: how many days back from now to include items (based on published_at).
    lookback_days: int = Field(default=7, ge=1)
    # ARG-124: weekly Keep summary scheduling. weekly_time defaults to the
    # same value as `time` so most users only set one knob. weekly_weekday
    # uses 3-letter names (Sun..Sat) matching `weekdays` and the scheduler's
    # _weekday_to_launchd mapping (Sun=0..Sat=6).
    weekly_enabled: bool = True
    weekly_time: str | None = None  # None → derived from `time` by validator
    weekly_weekday: str = "Mon"

    @model_validator(mode="after")
    def _derive_weekly_time(self) -> BriefingConfig:
        if self.weekly_time is None:
            self.weekly_time = self.time
        return self


class RunConfig(BaseModel):
    time: str = "06:00"
    daily_limit: int = Field(default=150, ge=0)


class InterestsConfig(BaseModel):
    topics: list[str] = []
    exclusions: list[str] = []


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model_triage: str = "qwen3:8b"
    model_deepdive: str = "qwen3:32b"


class LLMConfig(BaseModel):
    backend: Literal["ollama"] = "ollama"


class TriageConfig(BaseModel):
    preflight_filter: bool = True
    num_ctx: int = Field(default=2048, ge=512)


class GenealogistConfig(BaseModel):
    min_db_items: int = Field(default=50, ge=0)
    trust_skip_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # Model and context window for the genealogist node.
    # Defaults preserve the pre-ARG-91 behaviour (qwen3:32b, 3072 tokens).
    # Switch to qwen3:32b-q4_K_M + num_ctx=6144 once the benchmark in
    # docs/benchmarks/genealogist-quantized.md confirms acceptable quality.
    model: str = Field(default="qwen3:32b")
    num_ctx: int = Field(default=3072, ge=512)
    context_top_n: int = Field(default=3, ge=1, le=10)
    context_max_chars: int = Field(default=300, ge=50)
    embed_search_concurrency: int = Field(default=4, ge=1)


class DigestConfig(BaseModel):
    # ARG-173 상세 페이지 롱폼 다이제스트 노드. triage(8B)와 별개 모델.
    # config 필드로 두어 로컬 벤치 후 기본값 교체 가능(GenealogistConfig와 동일 관행).
    model: str = Field(default="qwen3:14b")
    num_ctx: int = Field(default=4096, ge=512)
    # 프롬프트에 넣는 raw_content 상한(문자). triage 2000자보다 크게.
    input_max_chars: int = Field(default=6000, ge=500)
    # 이 미만이면 롱폼을 만들지 않고 NULL(헛수고/환각 방지).
    min_content_chars: int = Field(default=1000, ge=0)
    # 생성 결과가 이 미만이면 버리고 NULL.
    min_output_chars: int = Field(default=150, ge=0)


class RSSFeedConfig(BaseModel):
    url: str
    category: Literal["Mainstream", "Alpha"] = "Mainstream"


class RSSConfig(BaseModel):
    feeds: list[RSSFeedConfig] = Field(
        default_factory=lambda: [RSSFeedConfig(**f) for f in _DEFAULT_RSS_FEEDS]
    )


class SPASourceConfig(BaseModel):
    listing_url: str
    category: Literal["Mainstream", "Alpha"] = "Mainstream"
    link_pattern: str
    base_url: str
    max_items: int = Field(default=10, ge=1)
    name: str = ""


class SPAConfig(BaseModel):
    sources: list[SPASourceConfig] = Field(
        default_factory=lambda: [SPASourceConfig(**s) for s in _DEFAULT_SPA_SOURCES]
    )


class WebConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    launchd_enabled: bool = False


class TrackingConfig(BaseModel):
    # ARG-204: cosine similarity threshold above which a new TechItem is
    # considered a "follow-up signal" for a Keep-ed asset. Mirrors the
    # module-local SIGNAL_SIMILARITY_THRESHOLD default in
    # argos.slack.services.track_check (kept there as the fallback constant).
    signal_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class TrustConfig(BaseModel):
    # ARG-206: deterministic trust synthesis. source_tiers maps a lower-cased,
    # www.-stripped domain (see argos.brain.trust.source_prior) to a tier;
    # unregistered domains fall back to "normal" (0.5).
    source_tiers: dict[str, str] = Field(
        default_factory=lambda: {
            "arxiv.org": "high", "github.com": "high",
            "openai.com": "high", "anthropic.com": "high",
            "ai.googleblog.com": "high", "huggingface.co": "high",
            # First-party official feeds also shipped in _DEFAULT_RSS_FEEDS —
            # their exact netlocs must be listed here or source_prior() falls
            # them back to "normal" (0.5), under-scoring first-party sources.
            "blog.google": "high", "ai.meta.com": "high", "mistral.ai": "high",
        }
    )
    weight_rubric: float = Field(default=0.6, ge=0.0)
    weight_prior: float = Field(default=0.2, ge=0.0)
    weight_corroboration: float = Field(default=0.2, ge=0.0)
    # T2 (corroboration pipeline) fields — pre-declared here so config
    # schema doesn't need another migration once T2 lands.
    corroboration_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    corroboration_lookback_days: int = Field(default=7, ge=1)


class FeedRankingConfig(BaseModel):
    # ARG-212: feed_score = weighted sum of recency decay, user-profile cosine
    # similarity, trust, and trending (corroboration reuse), plus a small
    # interest-topic bonus. Weights don't need to sum to 1.0 (interest_bonus
    # is an additive nudge, not a fifth weighted term).
    weight_recency: float = Field(default=0.35, ge=0.0)
    weight_profile: float = Field(default=0.35, ge=0.0)
    weight_trust: float = Field(default=0.15, ge=0.0)
    weight_trending: float = Field(default=0.15, ge=0.0)
    recency_half_life_hours: float = Field(default=48.0, ge=0.0)
    # Weight applied to the mean Pass(=Archived)-embedding when subtracting it
    # from the Keep-embedding mean to build the user profile vector.
    pass_weight: float = Field(default=0.3, ge=0.0)
    # Small additive bonus when title/summary matches an interests.topics term.
    interest_bonus: float = Field(default=0.05, ge=0.0)


class EventDetectionConfig(BaseModel):
    # ARG-249: 근접중복(SimHash)·고유명사 추출 임계값의 유일한 소유처.
    # 코드 상수로 두면 "판정 기준의 엄격함을 설정으로 조절할 수 있다"가 깨진다.
    # SimHash 64bit / 해밍 거리 <= 3 은 확정된 설계 결정(ARG-239)이다.
    simhash_hamming_max: int = Field(default=3, ge=0, le=64)
    # 문자 n-gram 폭. 실측상 4가 재배포본(제목 교체·문단 재배열)은 거리 1로,
    # 무관한 기사는 22로 벌려 놓는다. 3으로 좁히면 제목 교체가 거리 3이라
    # 컷에 딱 붙어 마진이 사라진다.
    simhash_shingle_size: int = Field(default=4, ge=1, le=10)
    # 후보 이름의 최대 단어 수 ("Claude Sonnet 5" = 3).
    entity_max_ngram: int = Field(default=4, ge=1, le=8)
    # 배치 내 문서빈도(DF) 컷. 배치가 entity_df_min_batch 미만이면
    # 비율 컷은 적용하지 않는다 — 문서 1건짜리 배치에서 전부 탈락하는 걸 막는다.
    entity_min_doc_count: int = Field(default=1, ge=1)
    entity_max_doc_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    entity_df_min_batch: int = Field(default=5, ge=1)
    # spaCy 보강(ARG-253)은 켜져 있어도 미설치면 조용히 건너뛴다.
    entity_spacy_enabled: bool = True
    # 열어 둘 파이프라인 이름. 코드에 박아 두면 다른 호환 모델을 설치해
    # 벤치마크해도 보조 경로가 조용히 꺼진 채로 돈다.
    entity_spacy_model: str = "en_core_web_sm"
    # ARG-264: 간선 가중치 네 항. 합이 1일 필요는 없다 — 판정 전에 합으로
    # 정규화하므로 join_threshold의 뜻("네 항의 가중 평균")이 유지된다.
    weight_cosine: float = Field(default=0.55, ge=0.0)
    weight_entity: float = Field(default=0.25, ge=0.0)
    weight_time: float = Field(default=0.15, ge=0.0)
    weight_keyword: float = Field(default=0.05, ge=0.0)
    # 이 값 이상이면 기존 사건에 붙고, 아니면 새 사건이 생긴다.
    join_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    # ARG-265: 후보 이웃을 찾는 시간 창(일)과 창 안에서 가져올 상위 K.
    # 창은 문서 발행 시각 앞뒤로 각각 window_days — 늦게 크롤된 과거 발행
    # 기사도 이미 저장된 더 최신 기사를 이웃으로 본다.
    # 실측(2026-08-23, 코퍼스 1071건): 7일 60건 / 14일 140건 / 30일 268건.
    # 창을 넓히면 **점수를 매길 후보 수**가 는다 — 이 값이 조절하는 건 그쪽이다.
    # 쿼리 비용 자체는 창과 거의 무관하다: 필터가 COALESCE(published_at,
    # created_at) 위에 걸려 ix_tech_items_published_at을 못 타고 매번 tech_items
    # 전수 순차 스캔이 돈다(2026-08-26, 1091건 기준 Rows Removed by Filter 979,
    # 7~8ms) — 비용은 코퍼스 전체 크기를 따라 자란다. COALESCE 표현식에 부분
    # 인덱스(WHERE embedding IS NOT NULL)를 걸면 정확한 정렬을 유지한 채 인덱스를
    # 다시 타게 되지만, 마이그레이션이 필요해 후속으로 미뤄 뒀다. ANN 인덱스를
    # 두지 않는 이유는 성능이 아니라 결정성이다 — event_candidates 모듈 docstring.
    window_days: float = Field(default=14.0, gt=0.0)
    candidate_k: int = Field(default=25, ge=1)

    # ARG-279: 야간 재군집의 Leiden 노브. 간선 채택 컷은 여기 없다 — 낮 배정과
    # 같은 join_threshold를 그대로 쓰기 때문이다(밤 전용 기준을 만들면 매일 밤
    # 뒤집기만 반복된다).
    # 다만 **값이 같을 뿐 기준이 같지는 않다.** 낮(event_scoring.choose_event)은
    # 이웃 표의 **합**을 이 값과 견주고, 밤(recluster_core.build_edges)은 **한
    # 쌍의 가중치**를 견준다. 같은 숫자라도 밤이 훨씬 엄격하다 — 재보정은 사용자
    # 판단이 필요한 열린 문제라 이 브랜치에서 건드리지 않았다(ARG-245가 소비자).
    # CPM을 기본으로 두는 건 modularity의 resolution limit 때문이다: 작은
    # 사건들이 큰 덩어리에 삼켜지는 쪽으로 기울어, "약하게만 이어진 기사들이
    # 한 덩어리가 되지 않는다"는 기준과 정면으로 부딪친다. CPM은 해상도
    # 파라미터가 "이 밀도 이상이어야 한 덩어리"라는 절대 기준이라 그 편향이 없다.
    leiden_objective: Literal["cpm", "modularity"] = "cpm"
    # CPM 해상도 γ. **None(기본)은 "join_threshold를 따라간다"는 뜻이다.**
    # 간선 가중치가 join_threshold 이상만 남으므로, 같은 값을 해상도로 두면
    # "임계값을 겨우 넘긴 간선들만으로 이어진 묶음"은 뭉치는 이득이 없고
    # (CPM 품질 = 내부 가중치합 - γ × 쌍의 수), 그보다 확실히 진한 묶음만
    # 살아남는다. 사슬 저항과 정상 병합을 동시에 만족하는 자리다.
    # 예전에는 0.55를 **복사해** 박아 뒀는데, 그러면 프로즈로만 묶인 독립 필드
    # 둘이라 한쪽만 내렸을 때 조용히 어긋난다. 실측: join_threshold=0.35 /
    # γ=0.55면 채택된 간선(0.4498)이 전부 γ보다 낮아, "더 잘 묶이라고" 내린
    # 임계값이 오히려 한 사건 문서 넷을 넷으로 흩어 놓는다. 그래서 기본값을
    # None으로 바꿔 코드로 묶었다 — 조율용 명시 오버라이드는 그대로 남는다.
    # 실측(2026-09-10, A-B-C 사슬 간선 ≈0.60 / x-y-z 삼각형 간선 1.0, 0.05
    # 간격으로 스윕): γ <= 0.30에서는 사슬이 안 갈라지고(한 덩어리),
    # 0.35~0.95 구간에서 사슬은 갈라지면서 삼각형은 뭉친 채 유지되고, 1.0부터는
    # 삼각형마저 쪼개지기 시작한다. 즉 "둘 다 만족"하는 구간은 [0.35, 0.95]이고
    # 그 중앙은 ≈0.65다. join_threshold 기본값 0.55는 중앙은 아니지만 구간
    # 안쪽이라 양쪽으로 마진이 있다.
    # `leiden_objective="modularity"`에서는 CPM 노브가 아니므로 무시된다.
    leiden_resolution: float | None = Field(default=None, ge=0.0)
    # leidenalg는 내부적으로 난수를 쓴다. 시드를 고정하지 않으면 같은 입력에서
    # 실행마다 다른 파티션이 나와 "교정"이 아니라 소음이 된다.
    leiden_seed: int = Field(default=42, ge=0)

    @property
    def effective_leiden_resolution(self) -> float:
        """실제로 CPM에 넘어가는 해상도 γ — 코어와 CLI 리포트의 **공용 한 자리**.

        `leiden_resolution`이 None이면 `join_threshold`를 따라가되,
        `MAX_TRACKING_LEIDEN_RESOLUTION`에서 멈춘다. 해석을 여기 한 곳에 두는
        이유는 리포트 때문이다: CLI가 `None`을 그대로 찍으면 사용자는 실제로
        어떤 γ로 돈 결과인지 알 수 없다.

        **왜 천장이 필요한가:** config가 `join_threshold=1.0`을 허용하는데
        (`le=1.0`), γ까지 1.0이면 가중치 1.0짜리 간선의 CPM 이득이 정확히 0이
        된다. 그리고 가중치 1.0은 이론값이 아니다 — `edge_weight`가 네 항의
        가중평균이라 **완전히 동일한 문서**(같은 기사를 두 소스에서 받은 흔한
        경우)에서 실제로 나온다. 이득이 0이면 Leiden은 붙일 이유가 없어 동일
        문서를 싱글턴으로 남기고, 재군집은 그걸 거짓 "가를 후보"로 올린다.
        실측(2026-09-11): 동일 문서 3개가 가중치 1.0 간선으로 전부 이어져
        있어도 γ=1.0이면 크기 [1,1,1], γ가 조금이라도 낮으면 [3].

        명시 오버라이드에는 천장을 걸지 않는다 — 그건 운영자가 직접 고른
        값이라 조용히 깎으면 조율 자체가 불가능해진다.

        `leiden_objective="modularity"`일 때는 CPM 파라미터가 아니라서 이 값이
        쓰이지 않는다.
        """
        if self.leiden_resolution is None:
            return min(self.join_threshold, MAX_TRACKING_LEIDEN_RESOLUTION)
        return self.leiden_resolution


MAX_TRACKING_LEIDEN_RESOLUTION = 0.99
"""`leiden_resolution=None`이 `join_threshold`를 따라갈 때의 천장.

간선 가중치의 상한이 1.0이므로 γ는 거기 못 미쳐야 최대 유사도 간선도 CPM
이득이 양수다. 0.99인 이유는 두 가지다: (1) 이득 0.01은 leidenalg가 병합을
포기하는 하한(실측 ~1e-7)보다 네 자릿수 여유가 있고, (2) 실측된 양호 구간
[0.35, 0.95] **바깥**이라 현실적인 설정은 이 천장에 닿지 않는다 — 클램프가
무는 구간은 join_threshold > 0.99뿐이다."""


class UserConfig(BaseModel):
    slack: SlackConfig = SlackConfig()
    briefing: BriefingConfig = BriefingConfig()
    run: RunConfig = RunConfig()
    interests: InterestsConfig = InterestsConfig()
    ollama: OllamaConfig = OllamaConfig()
    llm: LLMConfig = LLMConfig()
    triage: TriageConfig = TriageConfig()
    genealogist: GenealogistConfig = GenealogistConfig()
    digest: DigestConfig = DigestConfig()
    rss: RSSConfig = Field(default_factory=RSSConfig)
    spa: SPAConfig = Field(default_factory=SPAConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    trust: TrustConfig = Field(default_factory=TrustConfig)
    feed_ranking: FeedRankingConfig = Field(default_factory=FeedRankingConfig)
    event_detection: EventDetectionConfig = Field(default_factory=EventDetectionConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> UserConfig:
        if path is None:
            path = Path.home() / ".config" / "argos" / "config.toml"
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            return cls.model_validate(data)
        except FileNotFoundError:
            return cls()
        except OSError as exc:
            logger.warning(
                "Could not read config file %s (%s); using defaults.", path, exc
            )
            return cls()
        except UnicodeDecodeError as exc:
            logger.warning(
                "Config file %s is not valid UTF-8 (%s); using defaults.", path, exc
            )
            return cls()
        except tomllib.TOMLDecodeError as exc:
            logger.warning(
                "Config file %s contains invalid TOML (%s); using defaults.", path, exc
            )
            return cls()
        except ValidationError as exc:
            logger.warning(
                "Config file %s failed schema validation (%s); using defaults.",
                path,
                exc,
            )
            return cls()

    @classmethod
    def load_strict(cls, *, path: Path) -> UserConfig:
        """Load ``path`` without the silent-fallback behavior of :meth:`load`.

        Unlike :meth:`load`, this re-raises:
          - :class:`FileNotFoundError` / :class:`OSError` when the file can't be read,
          - :class:`UnicodeDecodeError` when the file is not valid UTF-8,
          - :class:`tomllib.TOMLDecodeError` when the file isn't valid TOML,
          - :class:`pydantic.ValidationError` when the parsed payload fails schema.

        Callers (e.g. ``argos schedule install --config <path>``) use this so an
        explicit operator-supplied config doesn't silently fall back to defaults.
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)


class Settings:
    def __init__(self) -> None:
        self.secrets = Secrets()
        self.user = UserConfig.load()

    @property
    def database_url(self) -> str:
        user = quote(self.secrets.POSTGRES_USER, safe="")
        password = quote(self.secrets.POSTGRES_PASSWORD, safe="")
        return (
            f"postgresql+asyncpg://{user}:{password}"
            f"@{self.secrets.POSTGRES_HOST}:{self.secrets.POSTGRES_PORT}/{self.secrets.POSTGRES_DB}"
        )


settings = Settings()
