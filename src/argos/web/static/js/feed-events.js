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

          if (entry.isIntersecting) {
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

    observeAll();

    // Keep/Pass (hx-swap="outerHTML") and pull-to-refresh / new-items-pill
    // (whole #feed-list replacement) both detach observed nodes and insert
    // fresh ones the observer hasn't seen yet. htmx fires these events on the
    // swapped-in content in both cases.
    document.body.addEventListener("htmx:afterSwap", observeAll);
    document.body.addEventListener("htmx:afterSettle", observeAll);
  }

  // --- Dwell: item detail page only --------------------------------------- //

  function initDwell() {
    var article = document.querySelector(".detail[data-item-id]");
    if (!article) return;

    var itemId = article.getAttribute("data-item-id");
    if (!itemId) return;

    var enteredAt = Date.now();
    var sent = false;

    function sendDwell() {
      if (sent) return;
      var seconds = (Date.now() - enteredAt) / 1000;
      if (seconds <= 0) return;
      sent = true;

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
      if (document.visibilityState === "hidden") sendDwell();
    });
    window.addEventListener("pagehide", sendDwell);
  }

  initImpressions();
  initDwell();
})();
