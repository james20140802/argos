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
