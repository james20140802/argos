// Card ⇄ detail morph for cross-document view transitions (ARG-243).
//
// The detail page's hero is always named `c<id>`. On the list side, a cover
// only gets that same name for the one navigation that involves it — forward
// (tap a card → its detail page) or back (detail → the list it came from).
// Covers are never named up front: on list → list navigations (category /
// sort changes) every named cover would fly to its new slot on its own while
// the rest of its card only crossfaded, so the thumbnails looked detached.
//
// Browsers without cross-document view transitions never fire these events;
// the page simply navigates as before.
(function () {
  "use strict";

  var DETAIL = /^\/(?:event|item)\/([0-9a-f-]{36})\/?$/i;

  function detailId(url) {
    if (!url) return null;
    try {
      var m = new URL(url, location.href).pathname.match(DETAIL);
      return m ? m[1].toLowerCase() : null;
    } catch (err) {
      return null;
    }
  }

  function isVisible(el) {
    var r = el.getBoundingClientRect();
    return r.bottom > 0 && r.top < window.innerHeight && r.width > 0;
  }

  // Name the cover for detail `id` (if it's on screen) for the duration of
  // one transition, then clear it so bfcache restores an unnamed page.
  function nameCover(id, transition) {
    var cover = document.querySelector('[data-vt="c' + id.replace(/-/g, "") + '"]');
    if (!cover || !isVisible(cover)) return;
    cover.style.viewTransitionName = cover.getAttribute("data-vt");
    transition.finished.finally(function () {
      cover.style.viewTransitionName = "";
    });
  }

  // Leaving a list for a detail page.
  window.addEventListener("pageswap", function (event) {
    if (!event.viewTransition || !event.activation) return;
    var id = detailId(event.activation.entry && event.activation.entry.url);
    if (id) nameCover(id, event.viewTransition);
  });

  // Arriving back on a list from a detail page.
  window.addEventListener("pagereveal", function (event) {
    if (!event.viewTransition || !window.navigation || !navigation.activation) return;
    var from = navigation.activation.from;
    var id = detailId(from && from.url);
    if (!id || detailId(location.href)) return;
    // This script sits at the end of <body>, so the cards are normally parsed
    // already; the DOMContentLoaded branch is a safety net.
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () {
        nameCover(id, event.viewTransition);
      }, { once: true });
    } else {
      nameCover(id, event.viewTransition);
    }
  });
})();
