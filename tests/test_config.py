from __future__ import annotations

from argos.config import Secrets, Settings, UserConfig


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
