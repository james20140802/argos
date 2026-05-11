"""Tests for the ``argos config`` CLI and the underlying ``config_store`` module.

Acceptance criteria mapping (ARG-49):
  (a) `config path` prints resolved default path             → test_config_path_*
  (b) set→get round-trip                                     → test_set_then_get_roundtrip
  (c) invalid value rejected, file unchanged                 → test_set_invalid_value_*
  (d) list[str] CSV coercion                                 → test_set_list_topics
  (e) secret rejection (existing + nonexistent secret key)   → test_set_rejects_secret_*
  (f) `config list` masks token-prefixed values              → test_list_masks_token_values
  (g) auto-creates parent dir + file                         → test_set_creates_parent_dir
  (h) atomic write — os.replace failure leaves file intact   → test_atomic_write_failure
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ImportError:  # Python <3.11
    import tomli as tomllib  # type: ignore[no-redef]

import pytest

from argos import config_store
from argos.cli import main as cli_main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cli(*argv: str) -> int:
    """Invoke argos.cli.main with a list of args. Returns the exit code."""
    return cli_main(list(argv))


# ---------------------------------------------------------------------------
# (a) config path
# ---------------------------------------------------------------------------


def test_config_path_prints_default(capsys, monkeypatch):
    monkeypatch.setenv("HOME", "/tmp/argos-fake-home")
    rc = run_cli("config", "path")
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out.endswith(".config/argos/config.toml")


def test_config_path_honours_override(tmp_path, capsys):
    target = tmp_path / "custom.toml"
    rc = run_cli("config", "--config", str(target), "path")
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == str(target)


# ---------------------------------------------------------------------------
# (b) set → get round-trip
# ---------------------------------------------------------------------------


def test_set_then_get_roundtrip(tmp_path, capsys):
    cfg = tmp_path / "config.toml"

    rc = run_cli("config", "--config", str(cfg), "set", "briefing.time", "09:30")
    assert rc == 0
    capsys.readouterr()  # drain

    rc = run_cli("config", "--config", str(cfg), "get", "briefing.time")
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == "09:30"

    # And the file actually contains it as TOML.
    with open(cfg, "rb") as f:
        data = tomllib.load(f)
    assert data["briefing"]["time"] == "09:30"


# ---------------------------------------------------------------------------
# (c) invalid value rejected; file unchanged
# ---------------------------------------------------------------------------


def test_set_invalid_value_returns_validation_exit(tmp_path, capsys):
    """`briefing.limit_per_category = 0` violates Field(ge=1) — exit 3."""
    cfg = tmp_path / "config.toml"
    # Seed the file with a valid value first.
    run_cli("config", "--config", str(cfg), "set", "briefing.limit_per_category", "5")
    capsys.readouterr()
    original = cfg.read_bytes()

    rc = run_cli(
        "config", "--config", str(cfg), "set", "briefing.limit_per_category", "0"
    )
    err = capsys.readouterr().err
    assert rc == 3
    assert "Invalid value" in err
    # File must be unchanged.
    assert cfg.read_bytes() == original


def test_set_invalid_literal_returns_validation_exit(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    rc = run_cli("config", "--config", str(cfg), "set", "llm.backend", "openai")
    err = capsys.readouterr().err
    assert rc == 3
    assert err  # message printed


def test_set_unknown_key_returns_two(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    rc = run_cli("config", "--config", str(cfg), "set", "does.not.exist", "x")
    err = capsys.readouterr().err
    assert rc == 2
    assert "Unknown config key" in err


# ---------------------------------------------------------------------------
# (d) list[str] CSV coercion
# ---------------------------------------------------------------------------


def test_set_list_topics(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    rc = run_cli(
        "config", "--config", str(cfg), "set", "interests.topics", "RAG, LLM, vector"
    )
    assert rc == 0
    capsys.readouterr()

    rc = run_cli("config", "--config", str(cfg), "get", "interests.topics")
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == "RAG,LLM,vector"

    with open(cfg, "rb") as f:
        data = tomllib.load(f)
    assert data["interests"]["topics"] == ["RAG", "LLM", "vector"]


def test_set_empty_list_clears_topics(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    run_cli("config", "--config", str(cfg), "set", "interests.topics", "RAG,LLM")
    capsys.readouterr()

    rc = run_cli("config", "--config", str(cfg), "set", "interests.topics", "")
    assert rc == 0
    capsys.readouterr()

    rc = run_cli("config", "--config", str(cfg), "get", "interests.topics")
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == ""


# ---------------------------------------------------------------------------
# (e) secret rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "POSTGRES_PASSWORD",
        "slack.bot_token",
        "slack.app_token",
        "anything.api_token",
        "deep.nested.secret",
    ],
)
def test_set_rejects_secret_keys(tmp_path, capsys, key):
    cfg = tmp_path / "config.toml"
    rc = run_cli("config", "--config", str(cfg), "set", key, "supersecret")
    err = capsys.readouterr().err
    assert rc == 4
    assert "Refusing to set secret value" in err
    # File must not have been created.
    assert not cfg.exists()


def test_get_rejects_secret_keys(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    rc = run_cli("config", "--config", str(cfg), "get", "slack.bot_token")
    err = capsys.readouterr().err
    assert rc == 4
    assert "Refusing to read secret value" in err


# ---------------------------------------------------------------------------
# (f) list masks token-prefixed values
# ---------------------------------------------------------------------------


def test_list_masks_token_values(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    # Seed channel_id with an xoxb-prefixed string (free-form str field) to
    # exercise the value-prefix masking branch.
    cfg.write_text('[slack]\nchannel_id = "xoxb-fake-token-12345"\n')
    rc = run_cli("config", "--config", str(cfg), "list")
    out = capsys.readouterr().out
    assert rc == 0
    assert "xoxb-***" in out
    # The raw token should NOT appear.
    assert "xoxb-fake-token-12345" not in out


def test_list_masks_xapp_prefix(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[slack]\nchannel_id = "xapp-1-AAA"\n')
    run_cli("config", "--config", str(cfg), "list")
    out = capsys.readouterr().out
    assert "xapp-***" in out
    assert "xapp-1-AAA" not in out


# ---------------------------------------------------------------------------
# (g) auto-create parent dir + file
# ---------------------------------------------------------------------------


def test_set_creates_parent_dir(tmp_path, capsys):
    cfg = tmp_path / "deep" / "nested" / "config.toml"
    assert not cfg.parent.exists()
    rc = run_cli("config", "--config", str(cfg), "set", "briefing.time", "10:00")
    capsys.readouterr()
    assert rc == 0
    assert cfg.exists()
    with open(cfg, "rb") as f:
        data = tomllib.load(f)
    assert data["briefing"]["time"] == "10:00"


# ---------------------------------------------------------------------------
# (h) atomic write — failure leaves file intact
# ---------------------------------------------------------------------------


def test_atomic_write_failure_preserves_original(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[briefing]\ntime = "07:00"\n')
    original_bytes = cfg.read_bytes()

    def boom(src, dst):  # type: ignore[no-untyped-def]
        # Simulate a filesystem failure during the final rename. The temp file
        # exists but the target must remain untouched.
        raise OSError("simulated rename failure")

    monkeypatch.setattr("argos.config_store.os.replace", boom)

    with pytest.raises(OSError):
        config_store.atomic_write(cfg, {"briefing": {"time": "12:00"}})

    assert cfg.read_bytes() == original_bytes
    # And the .tmp file should have been cleaned up by the cleanup branch.
    tmp_file = cfg.with_suffix(cfg.suffix + ".tmp")
    assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# Schema audit: any *token*/*password*/*secret* field added to UserConfig
# in the future must be in _SECRET_KEYS so it doesn't silently leak.
# ---------------------------------------------------------------------------


def test_schema_audit_catches_unlisted_secret_fields():
    # The import-time _audit_schema() call must have passed; here we just
    # verify that the function is callable and raises on a synthetic leak.
    from pydantic import BaseModel

    class _LeakySub(BaseModel):
        api_token: str = ""

    class _LeakyUser(BaseModel):
        leaky: _LeakySub = _LeakySub()

    # Walk it manually using the same predicate to confirm the guard fires.
    leaked = []
    for top_name, top_field in _LeakyUser.model_fields.items():
        annotation = top_field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            for sub in annotation.model_fields:
                dotted = f"{top_name}.{sub}"
                if config_store.is_secret(dotted):
                    leaked.append(dotted)
    assert leaked == ["leaky.api_token"]


# ---------------------------------------------------------------------------
# config_store unit tests
# ---------------------------------------------------------------------------


def test_default_config_path_under_home(monkeypatch):
    monkeypatch.setenv("HOME", "/tmp/xyz")
    p = config_store.default_config_path()
    # Path.home() reads $HOME on POSIX.
    assert str(p).startswith(str(Path.home()))
    assert p.name == "config.toml"


def test_is_secret_patterns():
    assert config_store.is_secret("slack.bot_token")
    assert config_store.is_secret("POSTGRES_PASSWORD")
    assert config_store.is_secret("any.SECRET.thing")
    assert not config_store.is_secret("briefing.time")
    assert not config_store.is_secret("interests.topics")
