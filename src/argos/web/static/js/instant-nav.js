// Instant response for filter chips and sort segments (ARG-243).
//
// Those links are htmx-boosted (see feed.html / portfolio.html): instead of a
// full page load — where the old page freezes at the tap until the network
// answers — htmx fetches the page and swaps <main> in place. This script makes
// the tap itself respond on the same frame: the chip / segment thumb moves to
// its new state and the current list dims, before any byte comes back.
(function () {
  "use strict";

  function listEl() {
    return document.querySelector("#feed-list, #portfolio-list");
  }

  document.addEventListener("click", function (event) {
    // Not gated on defaultPrevented: htmx's boost handler has already called
    // preventDefault() by the time the click bubbles here — that is the swap.
    if (event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    var link = event.target && event.target.closest
      ? event.target.closest("[data-instant-nav] a.chip")
      : null;
    if (!link || link.classList.contains("is-active")) return;

    var nav = link.closest("[data-instant-nav]");
    var chips = nav.querySelectorAll("a.chip");
    for (var i = 0; i < chips.length; i++) {
      var on = chips[i] === link;
      chips[i].classList.toggle("is-active", on);
      if (on) chips[i].setAttribute("aria-current", "true");
      else chips[i].removeAttribute("aria-current");
    }
    if (nav.classList.contains("segmented")) {
      nav.classList.toggle("segmented--second", link === chips[chips.length - 1]);
    }
    var list = listEl();
    if (list) list.classList.add("is-pending");
  });

  // If the swap fails, fall back to a real navigation rather than leaving a
  // dimmed list behind.
  function fail(event) {
    var elt = event.detail && event.detail.elt;
    if (!elt || !elt.closest || !elt.closest("[data-instant-nav]")) return;
    if (elt.href) location.href = elt.href;
  }
  document.addEventListener("htmx:responseError", fail);
  document.addEventListener("htmx:sendError", fail);
})();
