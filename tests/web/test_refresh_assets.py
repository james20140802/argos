from __future__ import annotations
from pathlib import Path
import argos.web

PKG = Path(argos.web.__file__).parent
SW = PKG / "assets" / "sw.js"
REFRESH_JS = PKG / "static" / "js" / "refresh.js"
FEED_EVENTS_JS = PKG / "static" / "js" / "feed-events.js"
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


def test_refresh_button_uses_svg_icon_not_text_glyph():
    # UX feedback (2026-07-06): the ⟳ text glyph read as cheap — the button
    # carries a stroke SVG icon instead, on both pages.
    for tpl in (FEED, PORTFOLIO):
        body = tpl.read_text(encoding="utf-8")
        assert "<svg" in body
        assert "⟳" not in body


def test_refresh_progress_is_a_rotating_ring_not_glyph():
    # UX feedback: in-flight progress is a border-ring spinner (button ::after
    # + .refresh-spinner::before), not a spinning text glyph.
    css = (PKG / "static" / "css" / "argos.css").read_text(encoding="utf-8")
    assert ".refresh-btn.is-refreshing::after" in css
    assert "⟳" not in css


def test_refresh_button_floats_bottom_right_desktop_only():
    # UX feedback: the desktop button is a floating action at the viewport's
    # bottom-right, and hidden on mobile where pull-to-refresh covers it.
    css = (PKG / "static" / "css" / "argos.css").read_text(encoding="utf-8")
    btn_block = css.split(".refresh-btn")[1]
    assert "display: none" in btn_block          # mobile default
    fab = css.split("Desktop refresh button")[1].split(".refresh-spinner")[0]
    assert "position: fixed" in fab
    assert "bottom" in fab and "right" in fab


def test_ptr_holds_pulled_gap_while_refreshing():
    # UX feedback: releasing past the threshold must KEEP the pulled-down gap
    # open (content held at PTR_HOLD via translateY) until the refresh
    # resolves, instead of snapping back immediately.
    body = REFRESH_JS.read_text(encoding="utf-8")
    assert "PTR_HOLD" in body
    assert "translateY" in body
    assert "touchcancel" in body
    finish = body.split("function onTouchFinish()")[1]
    assert "PTR_HOLD" in finish  # the hold is applied at gesture end...
    assert "refresh(kind)" in finish  # ...and released only after refresh()


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
    assert "argos-v17" in body               # bumped to v17 (ARG-201/213 feed ranking: argos.css + refresh.js changed)
    assert "argos-shell-refresh" in body     # message 리스너


def test_dwell_segment_gated_on_initial_visibility():
    # P2 fix (Codex review): a detail page opened in a background tab
    # (middle-click / "open in new tab" / PWA prefetch) starts hidden. The
    # dwell segment must NOT open until the page is first visible, else
    # background time gets logged as Dwell. Assert the initial segment state
    # is derived from visibilityState rather than hard-coded open.
    body = FEED_EVENTS_JS.read_text(encoding="utf-8")
    assert 'segmentOpen = document.visibilityState === "visible"' in body
    # And a resume path re-opens a fresh segment on the first visible transition.
    assert '"visible" && !segmentOpen' in body


def test_impression_timers_cancelled_on_hide():
    # P2 fix (Codex review): an armed 1s impression timer must be cleared when
    # the tab is backgrounded, else it fires while hidden / on resume and logs
    # an Impression for a card that wasn't continuously half-visible for a
    # second (and marks it seen, blocking a later real impression).
    body = FEED_EVENTS_JS.read_text(encoding="utf-8")
    assert "cancelPendingTimers" in body
    assert "timers.clear()" in body
    # Cleared from BOTH the impression visibility handler's hidden branch and
    # pagehide.
    imp = body.split("function initImpressions")[1]
    handler = imp.split('addEventListener("visibilitychange"')[1]
    assert 'visibilityState === "hidden"' in handler
    assert "cancelPendingTimers()" in handler
    assert 'addEventListener("pagehide", cancelPendingTimers)' in body


def test_impression_timer_checks_target_still_connected():
    # P2 fix (Codex review): a refresh (pill / header / pull-to-refresh) or a
    # Keep/Pass HTMX outerHTML swap can detach a feed card while its 1s
    # impression timer is armed. The timeout callback must re-check the
    # observed node is still in the document (target.isConnected) before
    # enqueueing — else it logs an Impression for a card that did not stay
    # continuously half-visible for the second (and marks it seen, blocking the
    # replacement card's genuine impression).
    body = FEED_EVENTS_JS.read_text(encoding="utf-8")
    # The observed node is passed into armTimer and closed over as `target`.
    assert "armTimer(itemId, entry.target)" in body
    assert "target.isConnected" in body
    # The connectivity guard lives inside the setTimeout callback (before the
    # enqueue), not merely somewhere in the file.
    cb = body.split("setTimeout(function ()")[1].split("IMPRESSION_DWELL_MS")[0]
    assert "if (!target.isConnected) return" in cb


def test_impression_timer_rearms_replacement_after_stale_node():
    # P2 fix (Codex review): when a Keep/Pass HTMX swap or refresh replaces a
    # card during the 1s impression window and the same item stays visible, the
    # replacement's observer callback must not bail on an occupied timer slot
    # whose node has been detached — that stale timer self-cancels via
    # isConnected and the still-visible item never records. armTimer must store
    # the node alongside the timer and, when the existing timer's node is no
    # longer connected, clear it and re-arm against the live replacement.
    body = FEED_EVENTS_JS.read_text(encoding="utf-8")
    imp = body.split("function initImpressions")[1]
    arm = imp.split("function armTimer")[1].split("var observer")[0]
    # The timer is stored with its node so a replacement can be detected.
    assert "timers.set(itemId, { timerId: timerId, target: target })" in arm
    # Same node or a still-connected node → leave the pending timer alone.
    assert "existing.target === target || existing.target.isConnected" in arm
    # A detached (stale) node's timer is cleared so arming falls through.
    assert "clearTimeout(existing.timerId)" in arm
    # The blanket `timers.has(itemId)` early-return is gone (it blocked re-arm).
    assert "timers.has(itemId)" not in arm


def test_impression_timer_gated_on_foreground_visibility():
    # P2 fix (Codex review): /feed opened or restored in a BACKGROUND tab
    # (middle-click / "open in new tab" / PWA prefetch) starts hidden, so the
    # IntersectionObserver can arm the 1s impression timer with no initial
    # visibilitychange to clear it — logging an Impression for a card the user
    # never saw. The timer must not be armed while hidden, and must be armed on
    # the first foreground for cards still on screen (mirroring the Dwell
    # segment's visibility gate). Assert: (1) arming bails unless visible, and
    # (2) the visibilitychange handler re-arms from the live intersecting set on
    # the transition to visible.
    body = FEED_EVENTS_JS.read_text(encoding="utf-8")
    imp = body.split("function initImpressions")[1]
    # armTimer refuses to start the count unless the page is foreground.
    assert 'document.visibilityState !== "visible"' in imp
    # A live map of currently-intersecting nodes exists to re-arm on foreground.
    assert "intersecting" in imp
    # The else-branch of the impression visibility handler (became visible)
    # re-arms timers from the intersecting set.
    handler = imp.split('addEventListener("visibilitychange"')[1]
    assert "intersecting.forEach" in handler
    assert "armTimer" in handler


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


def test_sw_precaches_portfolio_timeline_js():
    # ARG-205: base.html loads portfolio-timeline.js on the /portfolio shell, so
    # it belongs in the precached app shell (like feed-poll.js/refresh.js) —
    # otherwise the accordion is broken on a cached-shell / offline visit.
    body = SW.read_text(encoding="utf-8")
    assert "/static/js/portfolio-timeline.js" in body


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
