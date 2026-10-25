"""Tests for the `argos web` CLI subcommand (ARG-133)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from argos.cli import main


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
