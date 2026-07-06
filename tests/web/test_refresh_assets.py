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


def test_refresh_reprocesses_swapped_list_for_htmx():
    # P2 fix (Codex review): the fresh list comes from DOMParser, so HTMX's
    # initial document scan never saw it. Without htmx.process(freshEl) the
    # hx-post/hx-get controls inside (#feed-list Keep/Pass + load-more,
    # #portfolio-list Untrack) go inert after any refresh.
    body = REFRESH_JS.read_text(encoding="utf-8")
    assert "htmx.process" in body


def test_sw_precaches_refresh_js_and_bumps_version():
    body = SW.read_text(encoding="utf-8")
    assert "/static/js/refresh.js" in body
    assert "argos-v12" in body               # 최신+1 범프 (T2가 v11 → v12로 재범프)
    assert "argos-shell-refresh" in body     # message 리스너


def test_sw_message_listener_writes_shell_cache():
    body = SW.read_text(encoding="utf-8")
    assert "addEventListener('message'" in body or 'addEventListener("message"' in body
    assert "cache.put" in body or ".put(" in body


def test_sw_message_listener_validates_origin_and_app_shell_routes():
    # IMPORTANT fix: the message handler must mirror the fetch handler's
    # gate before writing arbitrary posted url/html into the shell cache —
    # same-origin check + APP_SHELL_ROUTES allowlist — instead of trusting
    # any posted data.url. Assert the allowlist check now appears at least
    # twice (fetch gate + message gate) and the message handler resolves the
    # posted url against self.location.origin.
    body = SW.read_text(encoding="utf-8")
    assert "argos-shell-refresh" in body
    assert body.count("APP_SHELL_ROUTES.includes") >= 2
    assert "self.location.origin" in body


# --------------------------------------------------------------------- #
# New-items poll pill (ARG-203)
# --------------------------------------------------------------------- #

FEED_POLL_JS = PKG / "static" / "js" / "feed-poll.js"


def test_feed_poll_js_exists_and_polls():
    body = FEED_POLL_JS.read_text(encoding="utf-8")
    assert "/feed/poll" in body
    assert "visibilitychange" in body or "visibilityState" in body
    assert "new_count" in body


def test_feed_template_has_pill_and_latest_cursor():
    body = FEED.read_text(encoding="utf-8")
    assert "data-latest-cursor" in body
    assert "new-items-pill" in body


def test_base_loads_feed_poll_script():
    assert "/static/js/feed-poll.js" in BASE.read_text(encoding="utf-8")


def test_sw_precaches_feed_poll_js():
    body = SW.read_text(encoding="utf-8")
    assert "/static/js/feed-poll.js" in body


def test_feed_poll_pill_tap_gates_hide_and_scroll_on_refresh_result():
    # CRITICAL fix: ArgosRefresh.refresh() never rejects — it resolves false
    # on a failed fetch (offline/5xx). An unconditional .then() would hide
    # the pill and scroll to top over stale content. The success callback
    # must inspect the resolved value and only hide+scroll when truthy.
    body = FEED_POLL_JS.read_text(encoding="utf-8")
    assert "function (ok)" in body or "function(ok)" in body
    assert "if (ok)" in body or "if (result)" in body


def test_feed_poll_hides_pill_when_no_newer_items():
    # P3 fix (Codex review): if the user refreshes via the header button or
    # pull-to-refresh while the pill is up, the next poll returns
    # new_count: 0 — the pill must be hidden then, not left stale.
    body = FEED_POLL_JS.read_text(encoding="utf-8")
    assert "hidePill()" in body
    poll_fn = body.split("function poll()")[1].split("function start()")[0]
    assert "hidePill()" in poll_fn


def test_feed_poll_reads_cursor_live_not_from_stale_captured_node():
    # CRITICAL fix: refresh.js's currentEl.replaceWith(freshEl) detaches the
    # original #feed-list node on every refresh (pill tap / header button /
    # pull-to-refresh), and the replacement carries an updated
    # data-latest-cursor. A cursor read that closes over a load-time
    # querySelector result keeps reading the detached node's stale cursor
    # forever, so count_new_since never advances and the pill re-appears for
    # items already on screen. The poll path must re-query #feed-list live on
    # every tick instead of reusing a single captured reference.
    body = FEED_POLL_JS.read_text(encoding="utf-8")
    assert "getAttribute(\"data-latest-cursor\")" in body or "getAttribute('data-latest-cursor')" in body
    # The live re-query must happen inside a helper/poll function, not just
    # once at top-level load time — i.e. #feed-list must be queried at least
    # twice (once for the load-time "is this a feed page" bail, again per poll).
    assert body.count("#feed-list") >= 2
    assert "currentCursor" in body
