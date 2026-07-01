// Graceful cover-image fallback.
//
// og:image URLs are crawled from third-party pages and can rot — when one
// 404s or fails to decode the browser would otherwise paint a broken-image
// icon inside the cover. This listener catches that error (capture phase, since
// `error` does not bubble) and degrades the cover to the same glyph fallback a
// null image_url renders, so a dead URL looks intentional rather than broken.
(function () {
  "use strict";

  function toGlyph(container) {
    if (!container || container.dataset.fallbackApplied === "1") return;
    container.dataset.fallbackApplied = "1";
    var isHero = container.classList.contains("detail-hero");
    container.classList.add(isHero ? "detail-hero--fallback" : "cover--fallback");
    var glyph = document.createElement("span");
    glyph.className = isHero ? "detail-hero__glyph" : "cover__glyph";
    glyph.setAttribute("aria-hidden", "true");
    glyph.textContent = "◎";
    container.appendChild(glyph);
  }

  document.addEventListener(
    "error",
    function (event) {
      var el = event.target;
      if (!el || el.tagName !== "IMG") return;
      if (el.classList.contains("cover__img")) {
        var cover = el.closest(".cover");
        el.remove();
        toGlyph(cover);
      } else if (el.classList.contains("detail-hero__img")) {
        var hero = el.closest(".detail-hero");
        el.remove();
        toGlyph(hero);
      }
    },
    true
  );
})();
