from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from argos.config import BriefingConfig, Secrets, Settings, UserConfig, _resolve_env_file


def test_secrets_loads_from_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    s = Secrets()
    assert s.POSTGRES_USER == "testuser"
    assert s.POSTGRES_PASSWORD == "testpass"
    assert s.SLACK_BOT_TOKEN == "xoxb-test"
    assert s.SLACK_APP_TOKEN == "xapp-test"


def test_secrets_defaults_without_env(monkeypatch):
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    s = Secrets(_env_file=None)
    assert s.POSTGRES_USER == "argos"
    assert s.SLACK_BOT_TOKEN == ""


def test_user_config_loads_from_toml(tmp_path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(
        b'[slack]\nchannel_id = "C999"\nsummary_language = "English"\n'
        b'[interests]\ntopics = ["RAG", "LLM"]\n'
    )
    cfg = UserConfig.load(path=toml_file)
    assert cfg.slack.channel_id == "C999"
    assert cfg.slack.summary_language == "English"
    assert cfg.interests.topics == ["RAG", "LLM"]


def test_user_config_falls_back_to_defaults_when_missing(tmp_path):
    missing = tmp_path / "nonexistent.toml"
    cfg = UserConfig.load(path=missing)
    assert cfg.slack.channel_id == ""
    assert cfg.slack.summary_language == "Korean"
    assert cfg.interests.topics == []
    assert cfg.ollama.model_triage == "qwen3:8b"
    assert cfg.llm.backend == "ollama"


def test_user_config_partial_toml_preserves_defaults(tmp_path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(b'[briefing]\ntime = "09:00"\n')
    cfg = UserConfig.load(path=toml_file)
    assert cfg.briefing.time == "09:00"
    assert cfg.slack.summary_language == "Korean"
    assert cfg.ollama.model_deepdive == "qwen3:32b"


def test_briefing_config_limit_per_category_default():
    cfg = UserConfig()
    assert cfg.briefing.limit_per_category == 10


def test_briefing_config_limit_per_category_toml_override(tmp_path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(b"[briefing]\nlimit_per_category = 3\n")
    cfg = UserConfig.load(path=toml_file)
    assert cfg.briefing.limit_per_category == 3
    # Other briefing defaults are preserved
    assert cfg.briefing.time == "07:00"


def test_briefing_config_limit_per_category_rejects_non_positive(tmp_path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(b"[briefing]\nlimit_per_category = -1\n")
    cfg = UserConfig.load(path=toml_file)
    # Invalid value triggers ValidationError fallback to defaults
    assert cfg.briefing.limit_per_category == 10


def test_settings_facade_exposes_secrets_and_user(tmp_path, monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "facadeuser")
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(b'[slack]\nchannel_id = "C123"\n')

    s = Settings.__new__(Settings)
    s.secrets = Secrets(_env_file=None)
    s.user = UserConfig.load(path=toml_file)

    assert s.secrets.POSTGRES_USER == "facadeuser"
    assert s.user.slack.channel_id == "C123"


def test_user_config_falls_back_on_malformed_toml(tmp_path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(b"[slack\nchannel_id = oops\n")  # invalid TOML
    cfg = UserConfig.load(path=toml_file)
    assert cfg.slack.channel_id == ""
    assert cfg.slack.summary_language == "Korean"


def test_user_config_falls_back_on_schema_validation_error(tmp_path):
    toml_file = tmp_path / "config.toml"
    # ollama.host must be a string; supplying an integer causes ValidationError
    toml_file.write_bytes(b"[ollama]\nhost = 99\n")
    cfg = UserConfig.load(path=toml_file)
    assert cfg.ollama.host == "http://localhost:11434"


def test_user_config_falls_back_on_permission_error(tmp_path, monkeypatch):
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(b'[slack]\nchannel_id = "C1"\n')

    original_open = open

    def raise_permission(*args, **kwargs):
        if str(toml_file) in str(args[0]):
            raise PermissionError("no read permission")
        return original_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", raise_permission)
    cfg = UserConfig.load(path=toml_file)
    assert cfg.slack.channel_id == ""


def test_genealogist_config_default():
    cfg = UserConfig()
    assert cfg.genealogist.min_db_items == 50


def test_genealogist_config_toml_override(tmp_path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(b"[genealogist]\nmin_db_items = 12\n")
    cfg = UserConfig.load(path=toml_file)
    assert cfg.genealogist.min_db_items == 12


def test_genealogist_config_zero_is_allowed(tmp_path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(b"[genealogist]\nmin_db_items = 0\n")
    cfg = UserConfig.load(path=toml_file)
    assert cfg.genealogist.min_db_items == 0


def test_genealogist_config_rejects_negative(tmp_path):
    toml_file = tmp_path / "config.toml"
    toml_file.write_bytes(b"[genealogist]\nmin_db_items = -3\n")
    cfg = UserConfig.load(path=toml_file)
    # Negative value triggers ValidationError fallback to defaults.
    assert cfg.genealogist.min_db_items == 50


# ---------------------------------------------------------------------------
# Finding 1 — empty weekdays silently disables briefing job
# Pydantic-level guard: BriefingConfig must reject an empty weekdays list.
# ---------------------------------------------------------------------------


def test_briefing_config_rejects_empty_weekdays() -> None:
    """weekdays=[] must raise ValidationError at pydantic construction time.

    An empty weekdays list causes launchd to receive ``StartCalendarInterval:
    <array/>``, which it interprets as "no schedule" — the job loads but never
    fires. Rejecting at config-load time prevents this silent failure.
    """
    with pytest.raises(ValidationError):
        BriefingConfig(weekdays=[])


def test_briefing_config_default_weekdays_is_all_seven() -> None:
    """The default weekdays value must cover all 7 days (non-empty invariant)."""
    cfg = BriefingConfig()
    assert len(cfg.weekdays) == 7
    assert set(cfg.weekdays) == {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}


def test_settings_database_url_encodes_special_chars(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "user@name")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/word")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "argos")

    s = Settings.__new__(Settings)
    s.secrets = Secrets(_env_file=None)
    s.user = UserConfig()

    url = s.database_url
    assert "user%40name" in url
    assert "p%40ss%2Fword" in url
    assert url.startswith("postgresql+asyncpg://")


# ---------------------------------------------------------------------------
# XDG env-file resolution (ARG-74)
# ---------------------------------------------------------------------------


def _clear_postgres_env(monkeypatch) -> None:
    """Remove POSTGRES_* vars from os.environ to prevent load_dotenv() bleed-through.

    database.rebuild() calls load_dotenv(override=True) which permanently writes
    env values into os.environ for the remainder of the test session.  Tests that
    rely on Secrets reading from an _env_file kwarg must clear those vars first so
    the env-var layer (higher priority in pydantic-settings) does not shadow the
    file.
    """
    for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
                "POSTGRES_HOST", "POSTGRES_PORT"):
        monkeypatch.delenv(key, raising=False)


def test_secrets_reads_from_xdg_path_by_default(tmp_path, monkeypatch):
    """Secrets must load from the XDG path when it exists and no override is set."""
    # Point HOME at tmp so Path.home() / ".config" / "argos" / ".env" is under tmp.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("ARGOS_ENV_FILE", raising=False)
    _clear_postgres_env(monkeypatch)

    xdg_env = tmp_path / ".config" / "argos" / ".env"
    xdg_env.parent.mkdir(parents=True, exist_ok=True)
    xdg_env.write_text("POSTGRES_USER=xdguser\nPOSTGRES_PASSWORD=xdgpass\n", encoding="utf-8")

    # _resolve_env_file reads os.environ live — no reload needed.
    resolved = _resolve_env_file()
    assert resolved is not None
    assert resolved == xdg_env

    # Secrets reads from the resolved file directly via _env_file kwarg.
    s = Secrets(_env_file=str(xdg_env))
    assert s.POSTGRES_USER == "xdguser"
    assert s.POSTGRES_PASSWORD == "xdgpass"


def test_secrets_honors_argos_env_file_override(tmp_path, monkeypatch):
    """ARGOS_ENV_FILE must win over the XDG path."""
    override_env = tmp_path / "custom.env"
    override_env.write_text("POSTGRES_USER=overrideuser\n", encoding="utf-8")

    monkeypatch.setenv("ARGOS_ENV_FILE", str(override_env))
    # Even if XDG path would also exist, override wins.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("HOME", raising=False)
    _clear_postgres_env(monkeypatch)

    resolved = _resolve_env_file()
    assert resolved == override_env

    s = Secrets(_env_file=str(override_env))
    assert s.POSTGRES_USER == "overrideuser"


def test_secrets_falls_back_to_repo_env_with_deprecation_warning(
    tmp_path, monkeypatch, caplog
):
    """Fallback to cwd ./.env emits a deprecation WARNING."""
    # Ensure XDG path does not exist.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("ARGOS_ENV_FILE", raising=False)
    _clear_postgres_env(monkeypatch)

    # Write a cwd .env in tmp_path and chdir there.
    cwd_env = tmp_path / ".env"
    cwd_env.write_text("POSTGRES_USER=cwduser\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING, logger="argos.config"):
        resolved = _resolve_env_file()

    assert resolved is not None
    assert "deprecated" in caplog.text.lower() or "migrate-env" in caplog.text

    s = Secrets(_env_file=str(cwd_env))
    assert s.POSTGRES_USER == "cwduser"


def test_secrets_xdg_takes_precedence_over_repo_env(tmp_path, monkeypatch, caplog):
    """When both XDG path and cwd .env exist, XDG wins without a deprecation warning."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("ARGOS_ENV_FILE", raising=False)
    _clear_postgres_env(monkeypatch)

    # Create XDG env.
    xdg_env = tmp_path / ".config" / "argos" / ".env"
    xdg_env.parent.mkdir(parents=True, exist_ok=True)
    xdg_env.write_text("POSTGRES_USER=xdgwins\n", encoding="utf-8")

    # Also create a cwd .env.
    cwd_env = tmp_path / ".env"
    cwd_env.write_text("POSTGRES_USER=cwdloser\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING, logger="argos.config"):
        resolved = _resolve_env_file()

    assert resolved == xdg_env
    assert "deprecated" not in caplog.text.lower()

    s = Secrets(_env_file=str(xdg_env))
    assert s.POSTGRES_USER == "xdgwins"


def test_resolve_env_file_returns_none_when_nothing_exists(tmp_path, monkeypatch):
    """_resolve_env_file returns None when no candidate files exist."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("ARGOS_ENV_FILE", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in this dir

    resolved = _resolve_env_file()
    assert resolved is None
