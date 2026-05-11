from __future__ import annotations

import logging
from typing import Literal

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-reuse-import]
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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


class SlackConfig(BaseModel):
    channel_id: str = ""
    summary_language: str = "Korean"


class BriefingConfig(BaseModel):
    time: str = "07:00"
    weekdays: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    limit_per_category: int = Field(default=10, ge=1)


class RunConfig(BaseModel):
    time: str = "06:00"


class InterestsConfig(BaseModel):
    topics: list[str] = []
    exclusions: list[str] = []


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model_triage: str = "qwen3:8b"
    model_deepdive: str = "qwen3:32b"


class LLMConfig(BaseModel):
    backend: Literal["ollama"] = "ollama"


class GenealogistConfig(BaseModel):
    min_db_items: int = Field(default=50, ge=0)


class UserConfig(BaseModel):
    slack: SlackConfig = SlackConfig()
    briefing: BriefingConfig = BriefingConfig()
    run: RunConfig = RunConfig()
    interests: InterestsConfig = InterestsConfig()
    ollama: OllamaConfig = OllamaConfig()
    llm: LLMConfig = LLMConfig()
    genealogist: GenealogistConfig = GenealogistConfig()

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
