// Pull-to-refresh + desktop refresh button (ARG-202).
//
// The service worker (sw.js) serves /feed and /portfolio stale-while-revalidate,
// so a manual "pull down" gesture would otherwise still show the cached shell
// first. This script re-requests the current route bypassing the HTTP cache,
// swaps the list container in place, and pushes the fresh HTML back into the
// SW's cache (via postMessage) so a later revisit gets the updated shell
// instead of the stale one.
//
// Exposes window.ArgosRefresh.refresh(kind) — kind is 'feed' or 'portfolio'.
// Task 2 (ARG-203?) depends on this exact global + message protocol, so keep
// the names stable.
(function () {
  "use strict";

  var CONTAINER_SELECTORS = {
    feed: "#feed-list",
    portfolio: "#portfolio-list",
  };

  function getContainerSelector(kind) {
    return CONTAINER_SELECTORS[kind] || null;
  }

  function notifyServiceWorker(url, html) {
    try {
      if (
        "serviceWorker" in navigator &&
        navigator.serviceWorker &&
        navigator.serviceWorker.controller
      ) {
        navigator.serviceWorker.controller.postMessage({
          type: "argos-shell-refresh",
          url: url,
          html: html,
        });
      }
    } catch (err) {
      // Never let SW sync failures break the visible refresh.
      if (window.console && console.warn) {
        console.warn("[argos] failed to notify service worker:", err);
      }
    }
  }

  function refresh(kind) {
    try {
      var selector = getContainerSelector(kind);
      if (!selector || typeof fetch !== "function" || typeof DOMParser === "undefined") {
        return Promise.resolve(false);
      }

      var url = location.pathname + location.search;

      return fetch(url, {
        cache: "reload",
        headers: { "HX-Request": "false" },
      })
        .then(function (res) {
          if (!res || !res.ok) return false;
          return res.text().then(function (html) {
            var doc = new DOMParser().parseFromString(html, "text/html");
            var freshEl = doc.querySelector(selector);
            var currentEl = document.querySelector(selector);
            if (freshEl && currentEl) {
              currentEl.replaceWith(freshEl);
              // freshEl came from DOMParser, so HTMX never scanned it: the
              // hx-post/hx-get controls inside (Keep/Pass, load-more,
              // Untrack) stay inert until explicitly processed.
              if (window.htmx && typeof window.htmx.process === "function") {
                window.htmx.process(freshEl);
              }
            }
            notifyServiceWorker(url, html);
            return true;
          });
        })
        .catch(function (err) {
          if (window.console && console.warn) {
            console.warn("[argos] refresh failed:", err);
          }
          return false;
        });
    } catch (err) {
      // Defensive: never throw in unsupported environments.
      return Promise.resolve(false);
    }
  }

  // --- Desktop refresh button: delegated click on [data-refresh]. ---
  document.addEventListener("click", function (event) {
    var target = event.target && event.target.closest
      ? event.target.closest("[data-refresh]")
      : null;
    if (!target) return;
    event.preventDefault();
    target.classList.add("is-refreshing");
    refresh(target.getAttribute("data-refresh")).then(function () {
      target.classList.remove("is-refreshing");
    });
  });

  // --- Pull-to-refresh gesture (mobile). ---
  // Only arms when the touch starts at the very top of the scroll container,
  // so it never fights the page's normal scroll and coexists with the
  // browser's native overscroll-refresh.
  var PTR_THRESHOLD = 70;
  var touchStartY = null;
  var ptrArmed = false;
  var ptrTriggered = false;

  function currentKind() {
    var path = location.pathname;
    if (path === "/feed" || path.indexOf("/feed/") === 0) return "feed";
    if (path === "/portfolio" || path.indexOf("/portfolio/") === 0) return "portfolio";
    return null;
  }

  function showSpinner() {
    var el = document.querySelector(".refresh-spinner");
    if (el) el.classList.add("is-active");
  }

  function hideSpinner() {
    var el = document.querySelector(".refresh-spinner");
    if (el) el.classList.remove("is-active");
  }

  document.addEventListener(
    "touchstart",
    function (event) {
      if (!event.touches || event.touches.length !== 1) return;
      var scroller = document.scrollingElement || document.documentElement;
      ptrArmed = scroller && scroller.scrollTop === 0;
      ptrTriggered = false;
      touchStartY = ptrArmed ? event.touches[0].clientY : null;
    },
    { passive: true }
  );

  document.addEventListener(
    "touchmove",
    function (event) {
      if (!ptrArmed || ptrTriggered || touchStartY === null) return;
      if (!event.touches || event.touches.length !== 1) return;
      var delta = event.touches[0].clientY - touchStartY;
      if (delta > PTR_THRESHOLD) {
        var kind = currentKind();
        if (!kind) return;
        ptrTriggered = true;
        showSpinner();
        refresh(kind).then(hideSpinner);
      }
    },
    { passive: true }
  );

  document.addEventListener(
    "touchend",
    function () {
      ptrArmed = false;
      touchStartY = null;
    },
    { passive: true }
  );

  window.ArgosRefresh = { refresh: refresh };
})();
