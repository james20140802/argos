// Feed interaction collection: Impression + Dwell (ARG-207).
//
// Click is recorded server-side (GET /item/{id} inserts a Click feed_event
// directly — see argos.web.app.item_detail); this script covers the two
// events only the client can observe:
//   - Impression: a feed card that stays >=50% visible for >=1s, recorded
//     once per page load (a Set de-dupes repeat intersections from scroll
//     jitter — leaving and re-entering the viewport doesn't re-fire it).
//   - Dwell: seconds spent on the item detail page, sent via sendBeacon on
//     tab-hide / navigation-away so it survives the page unloading.
//
// Both funnel into POST /events/batch (added alongside this script).
// Impressions are queued and flushed on a debounce so a fast scroll through
// the feed doesn't fire one request per card.
(function () {
  "use strict";

  var BATCH_DEBOUNCE_MS = 2000;
  var IMPRESSION_DWELL_MS = 1000;
  var IMPRESSION_THRESHOLD = 0.5;

  var queue = [];
  var flushTimer = null;

  function scheduleFlush() {
    if (flushTimer !== null) return;
    flushTimer = setTimeout(flush, BATCH_DEBOUNCE_MS);
  }

  function flush() {
    flushTimer = null;
    if (queue.length === 0) return;
    var events = queue;
    queue = [];

    fetch("/events/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events: events }),
      keepalive: true,
    }).catch(function (err) {
      if (window.console && console.warn) {
        console.warn("[argos] feed-events batch send failed:", err);
      }
    });
  }

  function enqueue(evt) {
    queue.push(evt);
    scheduleFlush();
  }

  // Flush pending events immediately, preferring sendBeacon so an in-flight
  // page unload can't abort the request. The 2s debounce means a card can
  // become visible → be enqueued → the user clicks it and navigates away
  // before flush() ever fires; without this the Impression is silently lost.
  function flushBeacon() {
    if (flushTimer !== null) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
    if (queue.length === 0) return;
    var events = queue;
    queue = [];
    var payload = JSON.stringify({ events: events });
    if (navigator.sendBeacon) {
      var blob = new Blob([payload], { type: "application/json" });
      navigator.sendBeacon("/events/batch", blob);
    } else {
      fetch("/events/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
        keepalive: true,
      }).catch(function () {});
    }
  }

  // Feed page (and every other page): drain the queue on the way out.
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") flushBeacon();
  });
  window.addEventListener("pagehide", flushBeacon);

  // --- Impression: IntersectionObserver over feed cards ------------------ //

  function initImpressions() {
    if (typeof IntersectionObserver === "undefined") return;

    var seen = new Set();
    // itemId -> { timerId, target }. The observed node is stored alongside its
    // pending timer so a card REPLACED mid-countdown (a Keep/Pass HTMX swap or a
    // refresh) can be detected and its now-stale timer re-armed against the live
    // replacement node — see armTimer. Keying on itemId alone (without the node)
    // let a replacement bail on `timers.has(itemId)` while the old node's timer
    // self-cancelled via isConnected, so the still-visible item never recorded.
    var timers = new Map();
    // itemId -> node for every card currently ≥50% intersecting. Kept live so
    // that when the tab is first foregrounded we can arm timers for cards that
    // were already on screen during a hidden background load (see below) — the
    // IntersectionObserver does NOT re-fire just because visibility changed.
    var intersecting = new Map();

    // Arm the 1s dwell timer for a visible card, but ONLY while the page is
    // actually in the foreground. If `/feed` is opened or restored in a
    // background tab (middle-click / "open in new tab" / PWA prefetch) it
    // starts `hidden`, so there is no initial `visibilitychange` to clear a
    // timer armed here — mirroring the Dwell segment, we must not start the
    // count until the tab is visible. On the first foreground the
    // visibilitychange handler re-arms these from `intersecting`, so a
    // background-loaded card still counts once actually seen.
    function armTimer(itemId, target) {
      if (seen.has(itemId)) return;
      var existing = timers.get(itemId);
      if (existing !== undefined) {
        // A timer is already counting this item. Leave it if it belongs to the
        // same node (a re-observe or a foreground re-arm), or to a different
        // node that is still connected (the same card genuinely still on
        // screen). But if its node has been DETACHED — a Keep/Pass HTMX swap or
        // a refresh replaced the card during the 1s window — that pending timer
        // will self-cancel via its isConnected guard and never record. Drop it
        // and fall through to re-arm against the live replacement, so a
        // still-visible item still earns its Impression without the user having
        // to scroll it out of the viewport and back in.
        if (existing.target === target || existing.target.isConnected) return;
        clearTimeout(existing.timerId);
        timers.delete(itemId);
      }
      if (document.visibilityState !== "visible") return;
      var timerId = setTimeout(function () {
        timers.delete(itemId);
        if (seen.has(itemId)) return;
        // The observed card must still be in the live document after the full
        // second. A refresh (pill / header / pull-to-refresh replaces
        // #feed-list via refresh.js's replaceWith) or a Keep/Pass HTMX
        // outerHTML swap can DETACH this card mid-countdown; its armed timer
        // survives in `timers` and would otherwise enqueue an Impression for a
        // card that did not stay continuously ≥50% visible for the second (and
        // mark it seen, blocking the fresh card's genuine impression).
        // isConnected is false once the node leaves the document, so a
        // swapped-out card is correctly dropped and the replacement
        // re-accumulates its own second via observeAll.
        if (!target.isConnected) return;
        seen.add(itemId);
        enqueue({ type: "Impression", item_id: itemId });
      }, IMPRESSION_DWELL_MS);
      timers.set(itemId, { timerId: timerId, target: target });
    }

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          var itemId = entry.target.getAttribute("data-item-id");
          if (!itemId) return;

          // Gate on the ratio, not just isIntersecting: an initial observe
          // callback (or a card dropping 60%->40%) can report isIntersecting
          // while intersectionRatio is below the 50% threshold. Starting the
          // dwell timer there would log an Impression for a card that never
          // stayed half-visible, corrupting the ARG-207 training events.
          if (entry.isIntersecting && entry.intersectionRatio >= IMPRESSION_THRESHOLD) {
            intersecting.set(itemId, entry.target);
            armTimer(itemId, entry.target);
          } else {
            intersecting.delete(itemId);
            var pending = timers.get(itemId);
            if (pending !== undefined) {
              clearTimeout(pending.timerId);
              timers.delete(itemId);
            }
          }
        });
      },
      { threshold: IMPRESSION_THRESHOLD }
    );

    function observeAll() {
      document.querySelectorAll(".card[data-item-id]").forEach(function (el) {
        observer.observe(el);
      });
    }

    // Cancel any armed impression timers when the tab is backgrounded. A card
    // can cross the 50% threshold, arm its 1s timer, then have the tab hidden
    // before the second elapses — and IntersectionObserver does NOT reliably
    // emit a below-threshold entry just because the document went hidden, so
    // the timer would otherwise fire while hidden (or on resume) and log an
    // Impression for a card that was never continuously half-visible for a
    // second. Worse, it marks the item seen, blocking a later genuine
    // impression. Clearing the pending timers (without marking seen) means the
    // card must re-accumulate a full foreground second to count.
    function cancelPendingTimers() {
      timers.forEach(function (entry) {
        clearTimeout(entry.timerId);
      });
      timers.clear();
    }
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "hidden") {
        cancelPendingTimers();
      } else {
        // First foreground (or resume after a hide): arm a fresh 1s timer for
        // every card still ≥50% visible. This is the ONLY path that starts the
        // count for a card that was already on screen when the tab loaded
        // hidden — armTimer bailed then, and the observer won't fire again for
        // an unchanged intersection.
        intersecting.forEach(function (node, itemId) {
          armTimer(itemId, node);
        });
      }
    });
    window.addEventListener("pagehide", cancelPendingTimers);

    observeAll();

    // Keep/Pass (hx-swap="outerHTML") and load-more go through HTMX's own swap
    // lifecycle, which fires htmx:afterSwap/afterSettle on the inserted content.
    document.body.addEventListener("htmx:afterSwap", observeAll);
    document.body.addEventListener("htmx:afterSettle", observeAll);
    // Pull-to-refresh / new-items-pill / header refresh replace #feed-list via
    // refresh.js's currentEl.replaceWith() + htmx.process(), which does NOT emit
    // the htmx events above. refresh.js dispatches argos:refreshed instead so
    // the freshly inserted cards still get observed.
    document.body.addEventListener("argos:refreshed", observeAll);
  }

  // --- Dwell: item detail page only --------------------------------------- //

  function initDwell() {
    var article = document.querySelector(".detail[data-item-id]");
    if (!article) return;

    var itemId = article.getAttribute("data-item-id");
    if (!itemId) return;

    // Dwell is measured per *visible segment*, not once for the whole page
    // lifetime. A detail page can be hidden (tab switch, phone lock, PWA
    // backgrounded) and later resumed; latching a single one-shot flag on the
    // first hide would drop every second of resumed reading. Each segment is
    // flushed as its own additive Dwell event — feed_events rows are summed
    // per item downstream — so leaving and returning is counted in full.
    // If the detail page loads in a background tab (opened via middle-click /
    // "open in new tab", or a PWA prefetch) its visibilityState is already
    // "hidden" — nobody is reading it yet. Start the segment CLOSED in that
    // case so background time isn't logged as Dwell; the visibilitychange
    // handler below opens a fresh segment on the first transition to "visible".
    var segmentOpen = document.visibilityState === "visible";
    var segmentStart = segmentOpen ? Date.now() : 0;

    function flushSegment() {
      if (!segmentOpen) return; // this segment was already flushed
      segmentOpen = false;
      var seconds = (Date.now() - segmentStart) / 1000;
      if (seconds <= 0) return;

      var payload = JSON.stringify({
        events: [{ type: "Dwell", item_id: itemId, value: seconds }],
      });

      if (navigator.sendBeacon) {
        var blob = new Blob([payload], { type: "application/json" });
        navigator.sendBeacon("/events/batch", blob);
      } else {
        // Fallback for browsers without sendBeacon — best-effort only, may be
        // aborted mid-flight by the very navigation this listener fires on.
        fetch("/events/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
          keepalive: true,
        }).catch(function () {});
      }
    }

    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "hidden") {
        flushSegment();
      } else if (document.visibilityState === "visible" && !segmentOpen) {
        // Resumed after a hide — open a fresh segment so the additional
        // visible time is measured and reported on the next flush.
        segmentStart = Date.now();
        segmentOpen = true;
      }
    });
    // Final segment on unload. Idempotent with the visibilitychange flush:
    // when a tab close fires hidden→pagehide, the segment is already closed,
    // so this is a no-op and never double-counts.
    window.addEventListener("pagehide", flushSegment);
  }

  initImpressions();
  initDwell();
})();
