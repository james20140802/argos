from __future__ import annotations

import tomllib
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class InterestsConfig(BaseModel):
    topics: list[str] = []
    exclusions: list[str] = []


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model_triage: str = "qwen3:8b"
    model_deepdive: str = "qwen3:32b"


class LLMConfig(BaseModel):
    backend: str = "ollama"


class UserConfig(BaseModel):
    slack: SlackConfig = SlackConfig()
    briefing: BriefingConfig = BriefingConfig()
    interests: InterestsConfig = InterestsConfig()
    ollama: OllamaConfig = OllamaConfig()
    llm: LLMConfig = LLMConfig()

    @classmethod
    def load(cls, path: Path | None = None) -> UserConfig:
        if path is None:
            path = Path.home() / ".config" / "argos" / "config.toml"
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            return cls()
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
