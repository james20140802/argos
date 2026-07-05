from __future__ import annotations
from pathlib import Path
import argos.web

PKG = Path(argos.web.__file__).parent
SW = PKG / "assets" / "sw.js"
REFRESH_JS = PKG / "static" / "js" / "refresh.js"
BASE = PKG / "templates" / "base.html"
FEED = PKG / "templates" / "feed.html"
PORTFOLIO = PKG / "templates" / "portfolio.html"


def test_refresh_js_exists_and_exposes_global():
    body = REFRESH_JS.read_text(encoding="utf-8")
    assert "ArgosRefresh" in body            # 전역 노출 (T2 재사용)
    assert "cache" in body and "reload" in body  # 캐시 우회 재요청


def test_refresh_js_bypasses_cache_and_notifies_sw():
    body = REFRESH_JS.read_text(encoding="utf-8")
    assert "argos-shell-refresh" in body     # SW 캐시 반영 메시지
    assert "postMessage" in body


def test_base_loads_refresh_script():
    assert "/static/js/refresh.js" in BASE.read_text(encoding="utf-8")


def test_feed_and_portfolio_have_refresh_button():
    assert 'data-refresh="feed"' in FEED.read_text(encoding="utf-8")
    assert 'data-refresh="portfolio"' in PORTFOLIO.read_text(encoding="utf-8")


def test_portfolio_refresh_scoped_to_dedicated_container():
    # CRITICAL fix: portfolio refresh must NOT swap the shared main.page shell
    # (which also contains the filter nav + refresh button itself). It must
    # swap a dedicated #portfolio-list container, mirroring feed's #feed-list.
    portfolio_body = PORTFOLIO.read_text(encoding="utf-8")
    assert 'id="portfolio-list"' in portfolio_body

    refresh_body = REFRESH_JS.read_text(encoding="utf-8")
    assert "#portfolio-list" in refresh_body
    assert "main.page" not in refresh_body


def test_sw_precaches_refresh_js_and_bumps_version():
    body = SW.read_text(encoding="utf-8")
    assert "/static/js/refresh.js" in body
    assert "argos-v11" in body               # 최신+1 범프
    assert "argos-shell-refresh" in body     # message 리스너


def test_sw_message_listener_writes_shell_cache():
    body = SW.read_text(encoding="utf-8")
    assert "addEventListener('message'" in body or 'addEventListener("message"' in body
    assert "cache.put" in body or ".put(" in body
