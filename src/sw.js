// Offline support: keeps a copy of the whole app (about 1 MB, model included) so it works at the
// table without a signal. The build fills in VERSION (a hash of the app's files) and FILES; a new
// build gives a new sw.js, which the browser picks up on the next visit and swaps in.
// Registered only over https (see main.ts), so the dev server is never cached.
const VERSION = "__VERSION__";
const FILES = __FILES__;
const PREFIX = "scorer-for-tacta-";
const CACHE = PREFIX + VERSION;

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(FILES.map((f) => new Request(f, { cache: "reload" }))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith(PREFIX) && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith(caches.open(CACHE).then((c) => c.match(req, { ignoreSearch: true })).then((hit) => hit ?? fetch(req)));
});
