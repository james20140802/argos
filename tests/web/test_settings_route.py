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
        data={"briefing.time": "08:15"},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings?saved=1"
    assert _load_toml(cfg)["briefing"]["time"] == "08:15"


def test_explicit_config_path_wins_over_default(tmp_path, monkeypatch):
    # The daemon started with `argos web --config <active>` must have its
    # settings page read/write <active>, not the default path.
    default_path = tmp_path / "default.toml"
    active_path = tmp_path / "active.toml"
    monkeypatch.setattr(config_store, "default_config_path", lambda: default_path)
    client = TestClient(build_web_app(config_path=active_path))

    resp = client.post(
        "/settings", data={"briefing.time": "08:15"}, follow_redirects=False
    )

    assert resp.status_code == 303
    assert _load_toml(active_path)["briefing"]["time"] == "08:15"
    assert not default_path.exists()  # default file untouched


def test_checkbox_uncheck_with_marker_disables_bool(cfg):
    # weekly_enabled defaults to True; a full-form POST whose checkbox is absent
    # but whose hidden marker is present is an intentional uncheck.
    client = TestClient(build_web_app())

    resp = client.post(
        "/settings",
        data={"briefing.weekly_enabled__present": "1"},  # marker, checkbox off
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert config_store.get_value(cfg, "briefing.weekly_enabled") is False


def test_partial_post_without_marker_leaves_bool_untouched(cfg):
    # A partial / non-browser POST that omits the checkbox marker must not flip
    # weekly_enabled off as a side effect of changing an unrelated field.
    config_store.set_value(cfg, "briefing.weekly_enabled", "true")
    client = TestClient(build_web_app())

    resp = client.post(
        "/settings", data={"briefing.time": "08:15"}, follow_redirects=False
    )

    assert resp.status_code == 303
    assert config_store.get_value(cfg, "briefing.weekly_enabled") is True


def test_post_weekdays_multi_checkbox_joins_to_list(cfg):
    # The weekday toggle group posts one entry per checked day; the server joins
    # them into the list[str] the model expects.
    client = TestClient(build_web_app())

    resp = client.post(
        "/settings",
        data={
            "briefing.weekdays__present": "1",
            "briefing.weekdays": ["Mon", "Wed", "Fri"],
        },
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert _load_toml(cfg)["briefing"]["weekdays"] == ["Mon", "Wed", "Fri"]


def test_post_weekdays_all_off_is_validation_error(cfg):
    # All days unchecked (marker present, no values) is an intentional empty
    # list → BriefingConfig.weekdays min_length=1 rejects it with a 400.
    config_store.set_value(cfg, "briefing.weekdays", "Mon,Tue")
    client = TestClient(build_web_app(), raise_server_exceptions=False)

    resp = client.post(
        "/settings",
        data={"briefing.weekdays__present": "1"},  # marker only, all days off
        follow_redirects=False,
    )

    assert resp.status_code == 400
    assert 'class="field-error"' in resp.text
    # The rejected value never reaches disk.
    assert _load_toml(cfg)["briefing"]["weekdays"] == ["Mon", "Tue"]


def test_post_partial_without_weekdays_marker_leaves_days_untouched(cfg):
    # A partial POST that never carried the weekday group must not blank it.
    config_store.set_value(cfg, "briefing.weekdays", "Mon,Tue")
    client = TestClient(build_web_app())

    resp = client.post(
        "/settings", data={"briefing.time": "08:15"}, follow_redirects=False
    )

    assert resp.status_code == 303
    assert _load_toml(cfg)["briefing"]["weekdays"] == ["Mon", "Tue"]


def test_get_renders_richer_controls(cfg):
    # The form uses purpose-built controls, not a text box for everything.
    client = TestClient(build_web_app())

    body = client.get("/settings").text

    assert 'type="time"' in body  # briefing.time / run.time
    assert 'class="daypicker"' in body  # briefing.weekdays toggle group
    assert 'name="briefing.weekly_weekday"' in body and "<select" in body


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
