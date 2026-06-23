/* Argos service worker registration (ARG-140).
 *
 * Service workers require a secure context (HTTPS or localhost). The Argos
 * web layer is served over both:
 *   - HTTPS via `tailscale serve` (PWA installable, SW registers)
 *   - Plain HTTP on the tailnet IP (core views still render; SW skipped)
 *
 * This script must therefore be a no-op outside of a secure context — any
 * attempt to call serviceWorker.register over HTTP throws a SecurityError
 * and would surface as a visible console error on the read-only HTTP path.
 */
(function () {
  if (typeof window === 'undefined') return;
  if (!window.isSecureContext) return;
  if (!('serviceWorker' in navigator)) return;

  window.addEventListener('load', function () {
    navigator.serviceWorker
      .register('/sw.js', { scope: '/' })
      .catch(function (err) {
        // Swallow — failing to register must not break the page.
        if (window.console && console.warn) {
          console.warn('[argos] service worker registration failed:', err);
        }
      });
  });
})();
