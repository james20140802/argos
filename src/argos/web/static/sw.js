/* Argos service worker (ARG-140).
 *
 * App-shell cache for the read views (feed + portfolio) and the small set of
 * static assets they need. Stale-while-revalidate for navigations, cache-first
 * for versioned static assets.
 *
 * Bump CACHE_VERSION whenever the precache list changes so old shells get
 * evicted on next install. The version is part of the cache name so multiple
 * generations can coexist briefly during activation.
 */
const CACHE_VERSION = 'argos-v1';
const APP_SHELL = [
  '/feed',
  '/portfolio',
  '/static/css/argos.css',
  '/static/img/logo.svg',
  '/static/img/icons/icon-192.png',
  '/static/img/icons/icon-512.png',
  '/static/js/htmx.min.js',
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
  // routes are POSTs (skipped above); the feed/portfolio entry HTML is the
  // navigable shell and is the only navigation request we serve from cache.
  if (req.mode === 'navigate') {
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
