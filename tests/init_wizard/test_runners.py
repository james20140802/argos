from __future__ import annotations

import subprocess

import httpx
import pytest

from argos.init_wizard import WizardStepError, runners


# ---------------------------------------------------------------------------
# which()
# ---------------------------------------------------------------------------


def test_which_returns_shutil_result(monkeypatch):
    monkeypatch.setattr(runners.shutil, "which", lambda b: f"/fake/{b}")
    assert runners.which("docker") == "/fake/docker"


def test_which_returns_none_when_missing(monkeypatch):
    monkeypatch.setattr(runners.shutil, "which", lambda b: None)
    assert runners.which("nope") is None


# ---------------------------------------------------------------------------
# subprocess _run helper (via docker_compose_up / alembic_upgrade_head)
# ---------------------------------------------------------------------------


def _make_proc(returncode=0, stdout="", stderr=""):
    class P:
        pass

    p = P()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


def test_docker_compose_up_calls_subprocess(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return _make_proc(0)

    monkeypatch.setattr(runners.subprocess, "run", fake_run)
    runners.docker_compose_up(tmp_path)
    assert seen["cmd"] == ["docker", "compose", "up", "-d"]
    assert seen["cwd"] == str(tmp_path)


def test_docker_compose_up_raises_on_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runners.subprocess,
        "run",
        lambda *a, **kw: _make_proc(1, stderr="boom"),
    )
    with pytest.raises(WizardStepError) as excinfo:
        runners.docker_compose_up(tmp_path)
    assert "exited with code 1" in str(excinfo.value)


def test_docker_compose_up_raises_when_binary_missing(monkeypatch, tmp_path):
    def fake_run(*a, **kw):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(runners.subprocess, "run", fake_run)
    with pytest.raises(WizardStepError) as excinfo:
        runners.docker_compose_up(tmp_path)
    assert "command not found" in str(excinfo.value)
    assert excinfo.value.hint is not None


def test_docker_compose_up_raises_on_timeout(monkeypatch, tmp_path):
    def fake_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=1)

    monkeypatch.setattr(runners.subprocess, "run", fake_run)
    with pytest.raises(WizardStepError) as excinfo:
        runners.docker_compose_up(tmp_path)
    assert "timeout" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# wait_pg_ready
# ---------------------------------------------------------------------------


def test_wait_pg_ready_returns_on_first_success(monkeypatch):
    monkeypatch.setattr(
        runners.subprocess,
        "run",
        lambda *a, **kw: _make_proc(0),
    )
    # Should not raise, should return None
    assert runners.wait_pg_ready("localhost", 5432, timeout=1) is None


def test_wait_pg_ready_times_out(monkeypatch):
    monkeypatch.setattr(
        runners.subprocess,
        "run",
        lambda *a, **kw: _make_proc(1, stderr="no connection"),
    )
    # Use a tiny timeout to keep the test fast.
    monkeypatch.setattr(runners, "PG_READY_POLL_INTERVAL_SEC", 0.01)
    with pytest.raises(WizardStepError) as excinfo:
        runners.wait_pg_ready("localhost", 5432, timeout=0.05)
    msg = str(excinfo.value)
    assert "did not become ready" in msg


def test_wait_pg_ready_falls_back_to_socket_probe_when_pg_isready_missing(monkeypatch):
    """When pg_isready is not on PATH, wait_pg_ready should use a TCP probe."""

    def fake_run(*a, **kw):
        raise FileNotFoundError("pg_isready")

    probe_calls = {"count": 0}

    def fake_probe(host, port, *, timeout):
        probe_calls["count"] += 1
        # Succeed on the first probe so the function returns promptly.
        return True

    monkeypatch.setattr(runners.subprocess, "run", fake_run)
    monkeypatch.setattr(runners, "_socket_probe", fake_probe)
    monkeypatch.setattr(runners, "PG_READY_POLL_INTERVAL_SEC", 0.01)

    assert runners.wait_pg_ready("localhost", 5432, timeout=1) is None
    assert probe_calls["count"] == 1


def test_wait_pg_ready_socket_fallback_times_out_when_nothing_listening(monkeypatch):
    """pg_isready missing + no TCP listener → WizardStepError with helpful hint."""

    def fake_run(*a, **kw):
        raise FileNotFoundError("pg_isready")

    monkeypatch.setattr(runners.subprocess, "run", fake_run)
    monkeypatch.setattr(runners, "_socket_probe", lambda *a, **kw: False)
    monkeypatch.setattr(runners, "PG_READY_POLL_INTERVAL_SEC", 0.01)

    with pytest.raises(WizardStepError) as excinfo:
        runners.wait_pg_ready("localhost", 5432, timeout=0.05)
    assert "did not become ready" in str(excinfo.value)
    assert excinfo.value.hint is not None


def test_socket_probe_returns_true_when_listener_accepts(monkeypatch):
    class _Sock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(runners.socket, "create_connection", lambda *a, **kw: _Sock())
    assert runners._socket_probe("localhost", 5432, timeout=0.1) is True


def test_socket_probe_returns_false_on_oserror(monkeypatch):
    def boom(*a, **kw):
        raise OSError("refused")

    monkeypatch.setattr(runners.socket, "create_connection", boom)
    assert runners._socket_probe("localhost", 5432, timeout=0.1) is False


# ---------------------------------------------------------------------------
# ollama_list / ollama_pull
# ---------------------------------------------------------------------------


def test_ollama_list_returns_names(monkeypatch):
    def fake_get(url, timeout):
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(runners.httpx, "get", fake_get)
    assert runners.ollama_list() == ["qwen3:8b", "nomic-embed-text"]


def test_ollama_list_raises_on_http_error(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(runners.httpx, "get", fake_get)
    with pytest.raises(WizardStepError) as excinfo:
        runners.ollama_list()
    assert "could not reach Ollama" in str(excinfo.value)


def test_ollama_pull_invokes_subprocess(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _make_proc(0)

    monkeypatch.setattr(runners.subprocess, "run", fake_run)
    runners.ollama_pull("qwen3:8b")
    assert seen["cmd"] == ["ollama", "pull", "qwen3:8b"]


# ---------------------------------------------------------------------------
# slack_auth_test
# ---------------------------------------------------------------------------


def test_slack_auth_test_happy_path(monkeypatch):
    def fake_post(url, headers, timeout):
        return httpx.Response(
            200, json={"ok": True, "team": "test-team"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(runners.httpx, "post", fake_post)
    payload = runners.slack_auth_test("xoxb-good")
    assert payload["team"] == "test-team"


def test_slack_auth_test_rejects_empty_token():
    with pytest.raises(WizardStepError):
        runners.slack_auth_test("")


def test_slack_auth_test_raises_on_slack_error(monkeypatch):
    def fake_post(url, headers, timeout):
        return httpx.Response(
            200, json={"ok": False, "error": "invalid_auth"}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(runners.httpx, "post", fake_post)
    with pytest.raises(WizardStepError) as excinfo:
        runners.slack_auth_test("xoxb-bad")
    assert "invalid_auth" in str(excinfo.value)


def test_slack_auth_test_raises_on_network_error(monkeypatch):
    def fake_post(url, headers, timeout):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(runners.httpx, "post", fake_post)
    with pytest.raises(WizardStepError) as excinfo:
        runners.slack_auth_test("xoxb-good")
    assert "network error" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ollama_ping
# ---------------------------------------------------------------------------


def test_ollama_ping_ok(monkeypatch):
    def fake_get(url, timeout):
        return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr(runners.httpx, "get", fake_get)
    assert runners.ollama_ping() is None


def test_ollama_ping_raises_on_failure(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(runners.httpx, "get", fake_get)
    with pytest.raises(WizardStepError):
        runners.ollama_ping()


# ---------------------------------------------------------------------------
# Regression: Finding 1 — alembic_upgrade_head propagates env_path into subprocess env
# ---------------------------------------------------------------------------


def test_alembic_upgrade_head_passes_env_path_vars_to_subprocess(monkeypatch, tmp_path):
    """A custom env_path must inject its POSTGRES_* values into the subprocess env.

    Without this fix the Alembic subprocess inherits the parent process env and
    may migrate the wrong database when env_path differs from the CWD's .env.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_USER=argos\n"
        "POSTGRES_PASSWORD=argos_dev_password\n"
        "POSTGRES_DB=argos\n"
        "POSTGRES_HOST=localhost\n"
        "POSTGRES_PORT=9999\n"
    )

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _make_proc(0)

    monkeypatch.setattr(runners.subprocess, "run", fake_run)
    runners.alembic_upgrade_head(tmp_path, env_path=env_file)

    assert captured.get("env") is not None, "env kwarg was not passed to subprocess.run"
    assert captured["env"]["POSTGRES_PORT"] == "9999", (
        f"expected POSTGRES_PORT=9999 in subprocess env, got: {captured['env'].get('POSTGRES_PORT')!r}"
    )


def test_alembic_upgrade_head_without_env_path_passes_none_env(monkeypatch, tmp_path):
    """Without env_path the subprocess inherits the default (None) env."""
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _make_proc(0)

    monkeypatch.setattr(runners.subprocess, "run", fake_run)
    runners.alembic_upgrade_head(tmp_path)

    assert captured.get("env") is None, (
        f"expected env=None when no env_path given, got: {captured.get('env')!r}"
    )
