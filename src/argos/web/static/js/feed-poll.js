// New-items polling for /feed (ARG-203).
//
// Every 60s, asks the server how many items sort newer than the cursor of
// the first page's top item (data-latest-cursor on #feed-list). If any are
// found, shows a "↑ 새 항목 N개" pill above the list — it never mutates the
// feed DOM itself. Only a tap on the pill triggers an actual refresh, reusing
// Task 1's window.ArgosRefresh.refresh('feed') (falling back to a full
// location.reload() if that global is absent).
//
// Polling pauses while the tab is hidden (document.visibilityState) and
// resumes on visibilitychange, so a backgrounded tab never wastes requests.
(function () {
  "use strict";

  var POLL_INTERVAL_MS = 60000;

  var list = document.querySelector("#feed-list[data-latest-cursor]");
  if (!list) return;

  var pill = document.querySelector("[data-new-items-pill]");
  var countEl = pill ? pill.querySelector("[data-new-items-count]") : null;

  var timerId = null;

  function currentCategory() {
    var params = new URLSearchParams(location.search);
    return params.get("category");
  }

  function showPill(n) {
    if (!pill) return;
    if (countEl) countEl.textContent = String(n);
    pill.hidden = false;
  }

  function hidePill() {
    if (!pill) return;
    pill.hidden = true;
  }

  function poll() {
    var cursor = list.getAttribute("data-latest-cursor");
    if (!cursor) return;

    var url = "/feed/poll?cursor=" + encodeURIComponent(cursor);
    var category = currentCategory();
    if (category) {
      url += "&category=" + encodeURIComponent(category);
    }

    fetch(url, { headers: { "HX-Request": "false" } })
      .then(function (res) {
        if (!res || !res.ok) return null;
        return res.json();
      })
      .then(function (data) {
        if (!data) return;
        var newCount = data.new_count || 0;
        if (newCount > 0) {
          showPill(newCount);
        }
      })
      .catch(function (err) {
        if (window.console && console.warn) {
          console.warn("[argos] feed poll failed:", err);
        }
      });
  }

  function start() {
    if (timerId !== null) return;
    timerId = setInterval(poll, POLL_INTERVAL_MS);
  }

  function stop() {
    if (timerId === null) return;
    clearInterval(timerId);
    timerId = null;
  }

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") {
      stop();
    } else {
      start();
    }
  });

  if (pill) {
    pill.addEventListener("click", function () {
      var refresh = window.ArgosRefresh && window.ArgosRefresh.refresh
        ? window.ArgosRefresh.refresh("feed")
        : null;

      if (refresh && typeof refresh.then === "function") {
        refresh.then(function (ok) {
          if (ok) {
            hidePill();
            window.scrollTo({ top: 0 });
          }
          // ok === false → refresh failed silently; leave the pill up so
          // the user can retry instead of hiding it over stale content.
        });
      } else {
        location.reload();
      }
    });
  }

  if (document.visibilityState !== "hidden") {
    start();
  }
})();
