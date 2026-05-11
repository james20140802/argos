from __future__ import annotations

import pytest

from argos.init_wizard import WizardStepError
from argos.init_wizard.env_file import load_env
from argos.init_wizard.steps import infra


@pytest.fixture(autouse=True)
def _force_noninteractive(monkeypatch):
    monkeypatch.setenv("ARGOS_INIT_NONINTERACTIVE", "1")


def _stub_runners(monkeypatch, *, installed_models=None, compose_calls=None):
    """Stub every runner so the test never hits real Docker/Ollama."""
    installed_models = installed_models or []
    compose_calls = compose_calls if compose_calls is not None else []
    pulled = []

    monkeypatch.setattr(infra.runners, "docker_compose_up", lambda repo: compose_calls.append(repo))
    monkeypatch.setattr(infra.runners, "wait_pg_ready", lambda h, p, **kw: None)
    monkeypatch.setattr(infra.runners, "alembic_upgrade_head", lambda repo: None)
    monkeypatch.setattr(infra.runners, "ollama_list", lambda host: list(installed_models))
    monkeypatch.setattr(infra.runners, "ollama_pull", lambda m: pulled.append(m))
    return pulled, compose_calls


def test_infra_uses_existing_env_defaults(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "POSTGRES_USER=existing_user\n"
        "POSTGRES_PASSWORD=existing_pw\n"
        "POSTGRES_DB=argos\n"
        "POSTGRES_HOST=localhost\n"
        "POSTGRES_PORT=5432\n",
    )
    _stub_runners(monkeypatch, installed_models=infra.REQUIRED_OLLAMA_MODELS)

    infra.run_infra_step(tmp_path, env_path=env_path)

    data = load_env(env_path)
    # Non-interactive mode keeps the existing values.
    assert data["POSTGRES_USER"] == "existing_user"
    assert data["POSTGRES_PASSWORD"] == "existing_pw"


def test_infra_skips_env_write_when_nothing_changes(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "POSTGRES_USER=argos\n"
        "POSTGRES_PASSWORD=argos_dev_password\n"
        "POSTGRES_DB=argos\n"
        "POSTGRES_HOST=localhost\n"
        "POSTGRES_PORT=5432\n",
    )
    original_mtime = env_path.stat().st_mtime_ns
    _stub_runners(monkeypatch, installed_models=infra.REQUIRED_OLLAMA_MODELS)

    infra.run_infra_step(tmp_path, env_path=env_path)
    # File wasn't rewritten — mtime unchanged.
    assert env_path.stat().st_mtime_ns == original_mtime


def test_infra_pulls_only_missing_models(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "POSTGRES_USER=argos\nPOSTGRES_PASSWORD=p\nPOSTGRES_DB=argos\n"
        "POSTGRES_HOST=localhost\nPOSTGRES_PORT=5432\n"
    )
    pulled, _ = _stub_runners(monkeypatch, installed_models=["qwen3:8b"])

    infra.run_infra_step(tmp_path, env_path=env_path)

    # qwen3:8b already present; qwen3:32b + nomic-embed-text must be pulled.
    assert set(pulled) == {"qwen3:32b", "nomic-embed-text"}


def test_infra_surfaces_pg_ready_timeout(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "POSTGRES_USER=argos\nPOSTGRES_PASSWORD=p\nPOSTGRES_DB=argos\n"
        "POSTGRES_HOST=localhost\nPOSTGRES_PORT=5432\n"
    )

    monkeypatch.setattr(infra.runners, "docker_compose_up", lambda repo: None)

    def boom(h, p, **kw):
        raise WizardStepError("pg not ready", hint="check docker compose ps")

    monkeypatch.setattr(infra.runners, "wait_pg_ready", boom)
    monkeypatch.setattr(infra.runners, "alembic_upgrade_head", lambda repo: None)
    monkeypatch.setattr(infra.runners, "ollama_list", lambda host: [])
    monkeypatch.setattr(infra.runners, "ollama_pull", lambda m: None)

    with pytest.raises(WizardStepError) as excinfo:
        infra.run_infra_step(tmp_path, env_path=env_path)
    assert "pg not ready" in str(excinfo.value)
    assert excinfo.value.hint and "docker compose ps" in excinfo.value.hint
