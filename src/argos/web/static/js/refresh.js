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
              // htmx.process() activates hx-* controls but — unlike an
              // HTMX-driven swap — emits NO htmx:afterSwap/afterSettle. Fire an
              // explicit event so listeners that only re-scan on those HTMX
              // events (feed-events.js impression tracking) observe the freshly
              // inserted cards. Covers every manual path (pill / header button /
              // pull-to-refresh) since they all funnel through refresh().
              try {
                document.body.dispatchEvent(
                  new CustomEvent("argos:refreshed", {
                    bubbles: true,
                    detail: { kind: kind, root: freshEl },
                  })
                );
              } catch (err) {
                // CustomEvent unsupported — degrade silently.
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
  // so it never fights the page's normal scroll. The content column follows
  // the finger down (damped); releasing past the threshold KEEPS the pulled
  // gap open (held at PTR_HOLD) with the ring spinning inside it until the
  // refresh settles, then the content eases back up. Releasing short of the
  // threshold springs straight back with no request.
  var PTR_THRESHOLD = 70; // finger travel (px) required to trigger a refresh
  var PTR_HOLD = 56;      // gap (px) held open while the refresh is in flight
  var PTR_MAX = 96;       // cap on the damped pull distance
  var PTR_DAMPING = 0.5;
  var SPINNER_SIZE = 30;  // keep in sync with .refresh-spinner width/height

  var touchStartY = null;
  var ptrArmed = false;
  var ptrRefreshing = false;
  var ptrPastThreshold = false;
  var ptrBaseTop = 0;

  function currentKind() {
    var path = location.pathname;
    if (path === "/feed" || path.indexOf("/feed/") === 0) return "feed";
    if (path === "/portfolio" || path.indexOf("/portfolio/") === 0) return "portfolio";
    return null;
  }

  function ptrContent() {
    return document.querySelector("main");
  }

  function setPull(px, animate) {
    var el = ptrContent();
    if (!el) return;
    el.style.transition = animate ? "transform 0.25s ease" : "none";
    el.style.transform = px > 0 ? "translateY(" + px + "px)" : "translateY(0)";
  }

  function clearPull() {
    var el = ptrContent();
    if (!el) return;
    el.style.transition = "";
    el.style.transform = "";
  }

  function showSpinner() {
    var el = document.querySelector(".refresh-spinner");
    if (!el) return;
    if (ptrBaseTop > 0) {
      // Center the ring inside the PTR_HOLD-tall gap that the pulled-down
      // content opens beneath its resting top (captured at touchstart,
      // before any transform skews getBoundingClientRect).
      el.style.top = Math.round(ptrBaseTop + (PTR_HOLD - SPINNER_SIZE) / 2) + "px";
    }
    el.classList.add("is-active");
  }

  function hideSpinner() {
    var el = document.querySelector(".refresh-spinner");
    if (el) el.classList.remove("is-active");
  }

  function settlePull() {
    setPull(0, true);
    hideSpinner();
    window.setTimeout(clearPull, 300);
  }

  document.addEventListener(
    "touchstart",
    function (event) {
      if (!event.touches || event.touches.length !== 1) return;
      if (ptrRefreshing) {
        ptrArmed = false;
        return;
      }
      var scroller = document.scrollingElement || document.documentElement;
      ptrArmed = !!(scroller && scroller.scrollTop === 0 && currentKind());
      ptrPastThreshold = false;
      touchStartY = ptrArmed ? event.touches[0].clientY : null;
      if (ptrArmed) {
        var content = ptrContent();
        ptrBaseTop = content ? content.getBoundingClientRect().top : 0;
      }
    },
    { passive: true }
  );

  document.addEventListener(
    "touchmove",
    function (event) {
      if (!ptrArmed || ptrRefreshing || touchStartY === null) return;
      if (!event.touches || event.touches.length !== 1) return;
      var scroller = document.scrollingElement || document.documentElement;
      if (scroller && scroller.scrollTop > 0) {
        // The gesture turned into a normal scroll — stand down.
        ptrPastThreshold = false;
        setPull(0, false);
        return;
      }
      var delta = event.touches[0].clientY - touchStartY;
      if (delta <= 0) {
        ptrPastThreshold = false;
        setPull(0, false);
        hideSpinner();
        return;
      }
      var pulled = Math.min(PTR_MAX, delta * PTR_DAMPING);
      setPull(pulled, false);
      ptrPastThreshold = delta > PTR_THRESHOLD;
      if (pulled > 8) {
        showSpinner();
      } else {
        hideSpinner();
      }
    },
    { passive: true }
  );

  function onTouchFinish() {
    if (!ptrArmed) return;
    ptrArmed = false;
    touchStartY = null;
    if (ptrRefreshing) return;
    var kind = currentKind();
    if (ptrPastThreshold && kind) {
      ptrRefreshing = true;
      setPull(PTR_HOLD, true);
      showSpinner();
      refresh(kind).then(function () {
        ptrRefreshing = false;
        settlePull();
      });
    } else {
      settlePull();
    }
    ptrPastThreshold = false;
  }

  document.addEventListener("touchend", onTouchFinish, { passive: true });
  document.addEventListener("touchcancel", onTouchFinish, { passive: true });

  window.ArgosRefresh = { refresh: refresh };
})();
