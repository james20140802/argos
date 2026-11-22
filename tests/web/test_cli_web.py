"""Tests for the `argos web` CLI subcommand (ARG-133)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from argos.cli import main


@pytest.fixture(autouse=True)
def _no_real_cleanup(monkeypatch):
    """`argos web`은 시작 시 DB를 정리한다(ARG-338) — 이 파일의 테스트는 실제 DB를 건드리지 않는다."""

    async def _noop():
        return None

    monkeypatch.setattr("argos.cli._fail_interrupted_deep_analyses", _noop)


def test_web_invokes_uvicorn_with_config_defaults():
    """`argos web` runs uvicorn against build_web_app() with config host/port."""
    with patch("uvicorn.run") as mock_run:
        rc = main(["web"])
    assert rc == 0
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    # First positional is the FastAPI app instance.
    from fastapi import FastAPI

    assert isinstance(args[0], FastAPI)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8765


def test_web_cli_flags_override_config():
    """--host / --port flags override values from settings.user.web."""
    with patch("uvicorn.run") as mock_run:
        rc = main(["web", "--host", "0.0.0.0", "--port", "9000"])
    assert rc == 0
    _, kwargs = mock_run.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9000


def test_web_cli_empty_host_falls_back_to_config():
    """--host "" is falsy and must not leak through to uvicorn as a blank bind."""
    with patch("uvicorn.run") as mock_run:
        rc = main(["web", "--host", ""])
    assert rc == 0
    _, kwargs = mock_run.call_args
    assert kwargs["host"] == "127.0.0.1"


@pytest.mark.parametrize("port", ["0", "65536"])
def test_web_cli_rejects_out_of_range_port(port):
    """argparse rejects ports outside 1..65535 at both bounds."""
    with patch("uvicorn.run"):
        with pytest.raises(SystemExit):
            main(["web", "--port", port])


# ---------------------------------------------------------------------------
# ARG-195: login token bootstrap + `argos web token`
# ---------------------------------------------------------------------------


def _token_file(tmp_path, monkeypatch):
    from argos.web import auth

    path = tmp_path / "cfg" / "argos" / "web_token"
    monkeypatch.setenv(auth.TOKEN_FILE_ENV, str(path))
    return path


def test_web_creates_token_without_printing_it(tmp_path, monkeypatch, capsys):
    import os
    import stat

    path = _token_file(tmp_path, monkeypatch)
    with patch("uvicorn.run") as mock_run:
        assert main(["web"]) == 0
    token = path.read_text().strip()
    assert token
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    out = capsys.readouterr()
    # launchd captures stdout/stderr into ~/Library/Logs — never the token.
    assert token not in out.out + out.err
    assert str(path) in out.out + out.err
    app = mock_run.call_args[0][0]
    assert app.state.security_policy.enabled is True
    assert app.state.security_policy.token_path == path


def test_web_tightens_loose_token_file_and_warns(tmp_path, monkeypatch, capsys):
    import os
    import stat

    path = _token_file(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text("existing-token\n")
    os.chmod(path, 0o644)
    with patch("uvicorn.run"):
        assert main(["web"]) == 0
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    out = capsys.readouterr()
    assert "existing-token" not in out.out + out.err
    assert "0600" in out.err


def test_web_token_prints_token(tmp_path, monkeypatch, capsys):
    path = _token_file(tmp_path, monkeypatch)
    assert main(["web", "token"]) == 0
    out = capsys.readouterr().out
    assert path.read_text().strip() in out


def test_web_token_rotate_changes_token(tmp_path, monkeypatch, capsys):
    path = _token_file(tmp_path, monkeypatch)
    main(["web", "token"])
    old = path.read_text().strip()
    capsys.readouterr()
    assert main(["web", "token", "--rotate"]) == 0
    new = path.read_text().strip()
    assert new != old
    out = capsys.readouterr().out
    assert new in out and old not in out


def test_web_token_url_prints_login_link(tmp_path, monkeypatch, capsys):
    path = _token_file(tmp_path, monkeypatch)
    assert main(["web", "token", "--url", "https://mac.tail1.ts.net/"]) == 0
    token = path.read_text().strip()
    assert f"https://mac.tail1.ts.net/login?token={token}" in capsys.readouterr().out


def test_web_token_url_rejects_non_http(tmp_path, monkeypatch, capsys):
    _token_file(tmp_path, monkeypatch)
    assert main(["web", "token", "--url", "javascript:alert(1)"]) == 2


def test_cmd_web_fails_interrupted_runs_before_serving(monkeypatch):
    """ARG-338: 웹을 띄우기 전에 대기·진행 중이던 심층 분석을 '중단됨'으로 정리한다."""
    order: list[str] = []

    async def fake_cleanup():
        order.append("cleanup")

    monkeypatch.setattr("argos.cli._fail_interrupted_deep_analyses", fake_cleanup)
    with patch("uvicorn.run", side_effect=lambda *a, **k: order.append("serve")):
        assert main(["web"]) == 0
    assert order == ["cleanup", "serve"]


def test_cmd_web_serves_even_if_cleanup_fails(monkeypatch):
    async def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr("argos.cli._fail_interrupted_deep_analyses", boom)
    with patch("uvicorn.run") as mock_run:
        assert main(["web"]) == 0
    mock_run.assert_called_once()


class _FakeCtx:
    def __init__(self):
        pass

    async def __aenter__(self):
        return object()

    async def __aexit__(self, *a):
        return False


@pytest.mark.parametrize("raises", [False, True])
def test_cleanup_disposes_engine_even_on_failure(monkeypatch, raises):
    """루프가 닫히기 전에 공유 엔진을 비운다 — 정리가 실패해도."""
    import asyncio

    from argos import cli, database

    disposed: list[bool] = []

    class _Engine:
        async def dispose(self):
            disposed.append(True)

    async def fake_fail(session):
        if raises:
            raise RuntimeError("db down")
        return 0

    # autouse 스텁을 걷어 내고 진짜 함수를 부른다.
    monkeypatch.undo()
    monkeypatch.setattr(database, "engine", _Engine())
    monkeypatch.setattr(database, "AsyncSessionLocal", lambda: _FakeCtx())
    monkeypatch.setattr(
        "argos.web.services.deep_analysis.fail_interrupted_runs", fake_fail
    )
    if raises:
        with pytest.raises(RuntimeError):
            asyncio.run(cli._fail_interrupted_deep_analyses())
    else:
        asyncio.run(cli._fail_interrupted_deep_analyses())
    assert disposed == [True]
