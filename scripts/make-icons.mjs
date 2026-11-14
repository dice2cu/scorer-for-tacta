// One-off: renders the home-screen icons (src/public/icon-192.png, icon-512.png, apple-touch-icon.png)
// from src/public/icon.svg in headless Edge/Chrome. Re-run only when the icon changes:
//   node scripts/make-icons.mjs
// Home-screen icons are full-bleed squares (phones crop them to their own shape), so the cards
// sit on a solid background inside the central "safe zone".
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { launch, serve, ROOT } from "../tests/browser/browser.mjs";

const BG = "#f4f2ee"; // the app's light background
const inner = readFileSync(join(ROOT, "src", "public", "icon.svg"), "utf8").replace(/^[\s\S]*?<svg[^>]*>|<\/svg>\s*$/g, "");
// icon.svg's drawing spans about 4..30 x 2..31 in its 32-unit box; scale it to ~60% of the square
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" fill="${BG}"/>
  <g transform="translate(256 256) scale(10.5) translate(-17 -16.5)">${inner}</g>
</svg>`;

const OUT = { "icon-512.png": 512, "icon-192.png": 192, "apple-touch-icon.png": 180 };
const { server, origin } = await serve();
const b = await launch();
try {
  await b.goto(`${origin}/`);
  for (const [name, size] of Object.entries(OUT)) {
    const url = await b.eval(`(async () => {
      const img = new Image();
      img.src = "data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}";
      await img.decode();
      const c = document.createElement("canvas");
      c.width = c.height = ${size};
      const ctx = c.getContext("2d");
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, 0, 0, ${size}, ${size});
      return c.toDataURL("image/png");
    })()`);
    writeFileSync(join(ROOT, "src", "public", name), Buffer.from(url.split(",")[1], "base64"));
    console.log("wrote src/public/" + name);
  }
} finally {
  await b.close();
  server.close();
}
