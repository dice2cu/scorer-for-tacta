// Minimal headless-browser driver over the Chrome DevTools Protocol (Edge or Chrome), using only
// Node built-ins (global WebSocket, Node >= 22). Serves dist/ (the production build) plus repo test
// data, or (serveDev) Vite's dev server for test pages that import the TypeScript sources.
//
// Isolation: every hostname except localhost fails to resolve (--host-resolver-rules), background
// networking is off, and the browser profile is a fresh throwaway directory.
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, normalize, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { build, createServer as createViteServer } from "vite";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const BROWSERS = [
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
];
const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".css": "text/css", ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json", ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp",
};

/** Test pages post their results to /__report; they end up in `reports`. */
function reportHandler(reports) {
  return (req, res, next) => {
    if (req.method !== "POST" || !req.url.startsWith("/__report")) return next();
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => { res.end("ok"); reports.push(JSON.parse(body)); });
  };
}

/** Builds the app (vite build), then serves dist/, plus /tests/* and /training-data/* from the repo. */
export async function serve() {
  await build({ configFile: join(ROOT, "vite.config.mjs"), logLevel: "warn" });
  const reports = [];
  const onReport = reportHandler(reports);
  const server = createServer((req, res) => onReport(req, res, () => {
    const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    const base = path.startsWith("/tests/") || path.startsWith("/training-data/") ? ROOT : join(ROOT, "dist");
    const file = normalize(join(base, path.endsWith("/") ? path + "index.html" : path));
    if (!file.startsWith(base) || !existsSync(file) || !statSync(file).isFile()) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { "Content-Type": TYPES[extname(file)] ?? "application/octet-stream" });
    res.end(readFileSync(file));
  }));
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  return { server, reports, origin: `http://localhost:${server.address().port}` };
}

/** Vite's dev server over the whole repo, for test pages that import src/ TypeScript directly
 *  (e.g. /tests/browser/engine.html). */
export async function serveDev() {
  const reports = [];
  const vite = await createViteServer({
    configFile: false, root: ROOT, logLevel: "warn", appType: "mpa",
    server: { host: "127.0.0.1", port: 0, hmr: false, watch: null },
    optimizeDeps: { noDiscovery: true, include: [] },
    plugins: [{ name: "report", configureServer: (s) => { s.middlewares.use(reportHandler(reports)); } }],
  });
  await vite.listen();
  const server = { close: () => vite.close() };
  return { server, reports, origin: `http://localhost:${vite.httpServer.address().port}` };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function launch({ gpu = "default" } = {}) {
  const exe = BROWSERS.find((b) => existsSync(b));
  if (!exe) throw new Error("no Edge/Chrome found");
  const profile = mkdtempSync(join(tmpdir(), "tacta-browser-test-"));
  const proc = spawn(exe, [
    "--headless=new", "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
    "--disable-component-update", "--disable-sync", "--disable-extensions", "--disable-default-apps",
    "--disable-features=Translate,OptimizationHints,MediaRouter", "--metrics-recording-only",
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost", `--user-data-dir=${profile}`,
    "--remote-debugging-port=0",
    // CI runners (GitHub sets CI=true) can't always give Chrome its sandbox
    ...(process.env.CI === "true" ? ["--no-sandbox"] : []),
    ...(gpu === "swiftshader" ? ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] : ["--enable-gpu", "--ignore-gpu-blocklist"]),
    "about:blank",
  ], { stdio: "ignore" });
  const portFile = join(profile, "DevToolsActivePort");
  for (let i = 0; i < 100 && !existsSync(portFile); i++) await sleep(100);
  const port = readFileSync(portFile, "utf8").split("\n")[0].trim();
  let pages = [];
  for (let i = 0; i < 50 && !pages.length; i++) {
    pages = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).filter((t) => t.type === "page");
    if (!pages.length) await sleep(100);
  }
  const ws = new WebSocket(pages[0].webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const pending = new Map();
  const listeners = [];
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
    } else if (msg.method) for (const l of listeners) l(msg);
  };
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const i = ++id;
    pending.set(i, { resolve, reject });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
  const logs = [];
  listeners.push((m) => {
    if (m.method === "Runtime.consoleAPICalled") logs.push(m.params.args.map((a) => a.value ?? a.description).join(" "));
    if (m.method === "Runtime.exceptionThrown") logs.push("EXCEPTION " + (m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text));
  });
  await send("Runtime.enable");
  await send("Page.enable");
  return {
    send, logs,
    async goto(url) {
      const loaded = new Promise((r) => listeners.push((m) => m.method === "Page.loadEventFired" && r()));
      await send("Page.navigate", { url });
      await loaded;
    },
    /** Evaluate an expression (awaits promises) and return its JSON value. */
    async eval(expression) {
      const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
      return r.result.value;
    },
    async waitFor(expression, timeoutMs = 60000) {
      const t0 = Date.now();
      while (Date.now() - t0 < timeoutMs) {
        if (await this.eval(expression)) return;
        await sleep(100);
      }
      throw new Error(`timed out waiting for ${expression}`);
    },
    async phone(width = 390, height = 844, dpr = 3) {
      await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: dpr, mobile: true });
    },
    async screenshot() {
      const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
      return Buffer.from(r.data, "base64");
    },
    async close() {
      try { await send("Browser.close"); } catch {}
      ws.close();
      proc.kill();
      await sleep(500);
      try { rmSync(profile, { recursive: true, force: true }); } catch {}
    },
  };
}
