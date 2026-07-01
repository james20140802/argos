/* Argos service worker (ARG-140).
 *
 * App-shell cache for the read views (feed + portfolio) and the small set of
 * static assets they need. Stale-while-revalidate for navigations, cache-first
 * for versioned static assets.
 *
 * Bump CACHE_VERSION whenever any precached asset changes — its path list OR
 * its *contents* (e.g. an argos.css edit). Static assets are served cache-first
 * with no content hash in their filenames, so a same-URL content change is NOT
 * picked up until the cache name changes and `activate` evicts the old one.
 * The version is part of the cache name so multiple generations can coexist
 * briefly during activation.
 *
 * v2: ARG-171 magazine-grid CSS + /feed shell changes.
 * v3: observation-log redesign — editorial layout, light/dark theming, left
 *     rail nav (argos.css rewrite) + new img-fallback.js. Bumped so clients on
 *     the v2 cache actually receive the new CSS/JS instead of the stale shell.
 * v4: signal ticker + ultra-wide layout (argos.css). Bumped so v3 clients pick
 *     up the widened grid / ticker CSS instead of the cached v3 stylesheet.
 * v5: active (pressed) Keep/Pass button state (argos.css + _feed_card.html).
 */
const CACHE_VERSION = 'argos-v5';
// Navigations we treat as the cacheable app shell. Everything else (e.g.
// /item/{id} detail pages) carries changing per-item state and must never be
// served from a stale cache, so it stays network-only.
const APP_SHELL_ROUTES = ['/feed', '/portfolio'];
const APP_SHELL = [
  ...APP_SHELL_ROUTES,
  '/static/css/argos.css',
  '/static/img/logo.svg',
  '/static/img/icons/icon-192.png',
  '/static/img/icons/icon-512.png',
  '/static/js/htmx.min.js',
  '/static/js/img-fallback.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(APP_SHELL)),
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)),
      ),
    ),
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // Don't cache action POSTs, HTMX fragment endpoints, or item detail pages —
  // those are user-state-sensitive and stale shells would mislead. Action
  // routes are POSTs (skipped above); only the /feed and /portfolio entry HTML
  // is a navigable shell we serve from (and refresh into) cache. Any other
  // navigation (e.g. /item/{id}) falls through to a plain network fetch so it
  // is never cached and can't go stale.
  if (req.mode === 'navigate' && APP_SHELL_ROUTES.includes(url.pathname)) {
    // Stale-while-revalidate: serve cached shell instantly, refresh in bg.
    event.respondWith(
      caches.match(req).then((cached) => {
        const network = fetch(req)
          .then((res) => {
            if (res && res.ok) {
              const copy = res.clone();
              caches.open(CACHE_VERSION).then((c) => c.put(req, copy));
            }
            return res;
          })
          .catch(() => cached);
        return cached || network;
      }),
    );
    return;
  }

  // Static assets — cache-first.
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(req).then((cached) => cached || fetch(req)),
    );
  }
});
