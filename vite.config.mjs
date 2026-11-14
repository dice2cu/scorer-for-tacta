// Vite build: src/index.html -> dist/ (one JS bundle, one stylesheet, content-hashed assets).
// Files in src/public/ (manifest, icons) are copied as-is. base "./" keeps every path relative, so
// the site works from any sub-folder (GitHub Pages serves it under /<repo>/).
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const src = fileURLToPath(new URL("src", import.meta.url));
const dist = fileURLToPath(new URL("dist", import.meta.url));

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else out.push(p);
  }
  return out;
}

/** Writes dist/sw.js from src/sw.js once the build is done: the list of files to keep offline and
 *  a version (hash of all of them), so every deploy that changes anything gets a new worker. */
function serviceWorker() {
  return {
    name: "tacta-service-worker",
    apply: "build",
    closeBundle() {
      const files = walk(dist).map((f) => relative(dist, f).split(sep).join("/")).filter((f) => f !== "sw.js").sort();
      const hash = createHash("sha256");
      for (const f of files) hash.update(f).update(readFileSync(join(dist, f)));
      const sw = readFileSync(join(src, "sw.js"), "utf8")
        .replace('"__VERSION__"', JSON.stringify(hash.digest("hex").slice(0, 12)))
        .replace("__FILES__", JSON.stringify(["./", ...files]));
      writeFileSync(join(dist, "sw.js"), sw);
    },
  };
}

export default defineConfig({
  root: src,
  base: "./",
  build: {
    outDir: dist,
    emptyOutDir: true,
    target: "es2022",
    assetsInlineLimit: 0, // keep the model weights a separate, cacheable file
  },
  server: { port: 8123 },
  preview: { port: 8124 },
  plugins: [serviceWorker()],
});
