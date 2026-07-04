"""Route + template tests for the 설정 screen (ARG-186).

The settings routes touch no database — they read/write ``config.toml`` via
config_store — so these run without Postgres (release.yml CI safe). The config
path is redirected to a ``tmp_path`` file by monkeypatching
``config_store.default_config_path``.
"""
from __future__ import annotations

import tomllib

import pytest
from starlette.testclient import TestClient

from argos import config_store
from argos.web.app import build_web_app


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setattr(config_store, "default_config_path", lambda: path)
    return path


def _load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_get_settings_renders_full_page(cfg):
    config_store.set_value(cfg, "slack.channel_id", "xoxb-secret")
    client = TestClient(build_web_app())

    resp = client.get("/settings")

    assert resp.status_code == 200
    body = resp.text
    assert "<!DOCTYPE html>" in body  # full page, not a partial
    assert 'name="interests.topics"' in body  # an editable field is rendered
    assert "xoxb-***" in body  # masked token in the read-only dump


def test_post_valid_redirects_and_persists(cfg):
    client = TestClient(build_web_app())

    resp = client.post(
        "/settings",
        # weekly_enabled is sent checked so the noop-on-absence uncheck doesn't
        # flip the default; we only mean to change briefing.time here.
        data={"briefing.time": "08:15", "briefing.weekly_enabled": "on"},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings?saved=1"
    assert _load_toml(cfg)["briefing"]["time"] == "08:15"


def test_get_after_save_shows_success_banner(cfg):
    client = TestClient(build_web_app())

    resp = client.get("/settings?saved=1")

    assert resp.status_code == 200
    assert "저장되었습니다" in resp.text


def test_post_invalid_rerenders_400_with_inline_error(cfg):
    client = TestClient(build_web_app(), raise_server_exceptions=False)

    resp = client.post(
        "/settings",
        data={"briefing.limit_per_category": "0"},  # ge=1 violation
        follow_redirects=False,
    )

    assert resp.status_code == 400
    body = resp.text
    assert "<!DOCTYPE html>" in body  # re-rendered page, not a redirect
    assert 'class="field-error"' in body
    assert 'value="0"' in body  # the user's submitted value is preserved


def test_post_never_writes_secret_key(cfg):
    client = TestClient(build_web_app())

    resp = client.post(
        "/settings",
        data={"slack.bot_token": "xoxb-evil", "briefing.time": "08:15"},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    data = _load_toml(cfg)
    assert "bot_token" not in data.get("slack", {})


def test_nav_link_present_on_settings_page(cfg):
    client = TestClient(build_web_app())

    resp = client.get("/settings")

    assert resp.status_code == 200
    assert 'href="/settings"' in resp.text
    assert 'aria-current="page"' in resp.text  # settings tab marked active
