// Portfolio card timeline accordion (ARG-205).
//
// Each card has one toggle button + one empty target container
// (#timeline-{asset_id}). Tapping the button the first time fetches the
// asset's recent-events fragment via htmx.ajax() and marks the container
// open; tapping again while open just clears the container and closes it —
// no second network round-trip for a collapse.
//
// A single delegated listener on `document` (rather than one per button)
// means cards loaded later via the /portfolio/items "더 보기" HTMX fragment
// get the same behavior without any re-binding step.
(function () {
  "use strict";

  document.addEventListener("click", function (event) {
    var btn = event.target.closest("[data-timeline-toggle]");
    if (!btn) return;

    var assetId = btn.getAttribute("data-timeline-toggle");
    var container = document.getElementById("timeline-" + assetId);
    if (!container) return;

    if (container.classList.contains("is-open")) {
      container.classList.remove("is-open");
      container.innerHTML = "";
      btn.setAttribute("aria-expanded", "false");
      return;
    }

    container.classList.add("is-open");
    btn.setAttribute("aria-expanded", "true");
    window.htmx.ajax("GET", "/portfolio/" + assetId + "/timeline", {
      target: container,
      swap: "innerHTML",
    });
  });
})();
