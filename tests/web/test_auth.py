"""ARG-195: web token storage + session cookie signing (no DB, no network)."""

from __future__ import annotations

import os
import stat

from argos.web import auth


def _mode(p) -> int:
    return stat.S_IMODE(os.stat(p).st_mode)


def test_default_token_path_honours_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(auth.TOKEN_FILE_ENV, str(tmp_path / "t"))
    assert auth.default_token_path() == tmp_path / "t"


def test_default_token_path_uses_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv(auth.TOKEN_FILE_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert auth.default_token_path() == tmp_path / "argos" / "web_token"


def test_ensure_token_creates_owner_only_file(tmp_path):
    path = tmp_path / "argos" / "web_token"
    token, created, _ = auth.ensure_token(path)
    assert created is True
    assert len(token) >= 40
    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700
    # idempotent: same token on second call
    again, created2, tightened = auth.ensure_token(path)
    assert again == token
    assert created2 is False
    assert tightened is False


def test_ensure_token_tightens_loose_file(tmp_path):
    path = tmp_path / "argos" / "web_token"
    path.parent.mkdir()
    path.write_text("abc\n")
    os.chmod(path, 0o644)
    os.chmod(path.parent, 0o755)
    token, created, tightened = auth.ensure_token(path)
    assert (token, created, tightened) == ("abc", False, True)
    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700


def test_rotate_token_invalidates_old_cookie(tmp_path):
    path = tmp_path / "web_token"
    old, _, _ = auth.ensure_token(path)
    cookie = auth.session_value(old)
    new = auth.rotate_token(path)
    assert new != old
    assert auth.read_token(path) == new
    assert auth.verify_session(cookie, old)
    assert not auth.verify_session(cookie, new)


def test_session_value_does_not_contain_token():
    token = auth.generate_token()
    assert token not in auth.session_value(token)


def test_verify_rejects_empty_and_wrong():
    assert not auth.verify_session(None, "t")
    assert not auth.verify_session("x", None)
    assert not auth.verify_session("x", "t")
    assert auth.verify_token(" t \n", "t")
    assert not auth.verify_token("u", "t")
    assert not auth.verify_token("", "t")
