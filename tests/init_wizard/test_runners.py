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
