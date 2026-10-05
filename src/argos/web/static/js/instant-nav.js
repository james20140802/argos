// Instant response for filter chips, sort segments and Keep/Pass (ARG-243).
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

  // Keep / Pass: show the decision on the tap itself. The server's answer
  // (only the button row is swapped) arrives a round trip later carrying the
  // same state, so nothing visibly changes again when it lands.
  document.addEventListener("click", function (event) {
    var btn = event.target && event.target.closest
      ? event.target.closest(".card-action--keep, .card-action--pass")
      : null;
    if (!btn || !btn.closest(".card-actions[id^='actions-']")) return;
    var row = btn.parentNode;
    var turningOn = !btn.classList.contains("is-active");
    var all = row.querySelectorAll(".card-action--keep, .card-action--pass");
    for (var i = 0; i < all.length; i++) {
      var b = all[i];
      var on = b === btn && turningOn;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
      var label = b.classList.contains("card-action--keep") ? "Keep" : "Pass";
      b.textContent = on ? "✓ " + label : label;
    }
    if (turningOn) {
      btn.classList.remove("just-committed");
      void btn.offsetWidth; // restart the pop
      btn.classList.add("just-committed");
    }
  }, true);

  // If the swap fails, fall back to a real navigation rather than leaving a
  // dimmed list behind.
  function fail(event) {
    var elt = event.detail && event.detail.elt;
    if (!elt || !elt.closest) return;
    if (elt.closest("[data-instant-nav]")) {
      if (elt.href) location.href = elt.href;
      return;
    }
    // A Keep/Pass that didn't land: the optimistic state is a lie — reload
    // the page so the card shows what the server actually holds.
    if (elt.closest(".card-actions[id^='actions-']")) location.reload();
  }
  document.addEventListener("htmx:responseError", fail);
  document.addEventListener("htmx:sendError", fail);
})();
