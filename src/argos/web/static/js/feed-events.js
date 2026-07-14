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
    var timers = new Map();

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
            if (seen.has(itemId) || timers.has(itemId)) return;
            var timerId = setTimeout(function () {
              timers.delete(itemId);
              if (seen.has(itemId)) return;
              seen.add(itemId);
              enqueue({ type: "Impression", item_id: itemId });
            }, IMPRESSION_DWELL_MS);
            timers.set(itemId, timerId);
          } else {
            var pending = timers.get(itemId);
            if (pending !== undefined) {
              clearTimeout(pending);
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
      timers.forEach(function (timerId) {
        clearTimeout(timerId);
      });
      timers.clear();
    }
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "hidden") cancelPendingTimers();
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
