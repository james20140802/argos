from __future__ import annotations


from argos.init_wizard.steps import healthcheck


def _stub_all_ok(monkeypatch):
    monkeypatch.setattr(healthcheck.runners, "run_async", lambda coro: None)

    # Avoid creating the real coroutine (we never await it because run_async is stubbed).
    monkeypatch.setattr(healthcheck.runners, "db_ping", lambda: None)
    monkeypatch.setattr(healthcheck.runners, "ollama_ping", lambda host: None)
    monkeypatch.setattr(healthcheck.runners, "slack_auth_test", lambda t: {"ok": True})


def test_healthcheck_returns_zero_when_all_pass(tmp_path, monkeypatch, capsys):
    env_path = tmp_path / ".env"
    env_path.write_text("SLACK_BOT_TOKEN=xoxb-good\n")

    _stub_all_ok(monkeypatch)
    monkeypatch.setattr(healthcheck, "is_loaded", None)  # ARG-51 absent — SKIP rows

    failures = healthcheck.run_healthcheck_step(tmp_path, env_path=env_path)
    assert failures == 0
    out = capsys.readouterr().out
    assert "PostgreSQL" in out
    assert "Ollama" in out
    assert "Slack auth.test" in out


def test_healthcheck_returns_nonzero_on_db_failure(tmp_path, monkeypatch, capsys):
    env_path = tmp_path / ".env"
    env_path.write_text("SLACK_BOT_TOKEN=xoxb-good\n")

    def boom(coro):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(healthcheck.runners, "run_async", boom)
    monkeypatch.setattr(healthcheck.runners, "db_ping", lambda: None)
    monkeypatch.setattr(healthcheck.runners, "ollama_ping", lambda host: None)
    monkeypatch.setattr(healthcheck.runners, "slack_auth_test", lambda t: {"ok": True})
    monkeypatch.setattr(healthcheck, "is_loaded", None)

    failures = healthcheck.run_healthcheck_step(tmp_path, env_path=env_path)
    assert failures == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "connection refused" in out


def test_healthcheck_skips_slack_when_token_missing(tmp_path, monkeypatch, capsys):
    env_path = tmp_path / ".env"
    env_path.write_text("# empty\n")

    _stub_all_ok(monkeypatch)
    monkeypatch.setattr(healthcheck, "is_loaded", None)

    failures = healthcheck.run_healthcheck_step(tmp_path, env_path=env_path)
    assert failures == 0
    out = capsys.readouterr().out
    assert "SKIP" in out


def test_healthcheck_uses_is_loaded_when_arg51_present(tmp_path, monkeypatch, capsys):
    env_path = tmp_path / ".env"
    env_path.write_text("SLACK_BOT_TOKEN=xoxb-good\n")

    _stub_all_ok(monkeypatch)
    seen_labels = []

    def fake_is_loaded(label):
        seen_labels.append(label)
        return False  # fail both

    monkeypatch.setattr(healthcheck, "is_loaded", fake_is_loaded)

    failures = healthcheck.run_healthcheck_step(tmp_path, env_path=env_path)
    assert failures == 2  # both launchd labels failed
    assert set(seen_labels) == set(healthcheck.LAUNCHD_LABELS)
