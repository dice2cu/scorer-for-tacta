// Draws the photo with the app's work on top: card outlines, counted dots (ring in the card's
// colour), partly covered dots (dashed ring in the dot's colour), dots shown for review (dashed
// white ring with "?"), ones the player turned off (grey with a cross).
import type { Board } from "./board.ts";
import { DECK_INFO } from "./decks.ts";

const MAX_SIDE = 3200; // device px of the drawn image (sharp enough to pinch-zoom into)

export function drawBoard(canvas: HTMLCanvasElement, bitmap: ImageBitmap, board: Board, showMarks: boolean) {
  const k = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const W = Math.round(bitmap.width * k), H = Math.round(bitmap.height * k);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, W, H);
  if (!showMarks) return;
  ctx.fillStyle = "rgba(0,0,0,0.25)"; // dim the photo a little so marks stand out
  ctx.fillRect(0, 0, W, H);

  const line = Math.max(1.5, W / 900); // stroke width in device px, ~proportional to display size
  const r = Math.max(board.dotRadius * k * 1.45, 4 * line);
  ctx.lineJoin = "round";
  for (const c of board.cards) {
    ctx.beginPath();
    c.corners.forEach(([x, y], i) => (i ? ctx.lineTo(x * k, y * k) : ctx.moveTo(x * k, y * k)));
    ctx.closePath();
    ctx.strokeStyle = DECK_INFO[c.deck].color;
    ctx.lineWidth = line;
    ctx.stroke();
  }
  for (const d of board.dots) {
    const x = d.x * k, y = d.y * k;
    ctx.setLineDash([]);
    if (d.counted) {
      ring(ctx, x, y, r, "rgba(0,0,0,0.85)", 3.2 * line);
      ring(ctx, x, y, r, DECK_INFO[d.deck].color, 2 * line);
      if (d.source === "manual") plus(ctx, x, y, r * 0.55, DECK_INFO[d.deck].color, 1.5 * line);
    } else if (d.source === "partial") {
      ctx.setLineDash([2.2 * line, 2 * line]);
      ring(ctx, x, y, r, "rgba(0,0,0,0.85)", 3.2 * line);
      ring(ctx, x, y, r, DECK_INFO[d.deck].color, 2 * line);
      ctx.setLineDash([]);
    } else if (d.source === "review") {
      ctx.setLineDash([3 * line, 2.5 * line]);
      ring(ctx, x, y, r * 1.15, "#ffffff", 2 * line);
      ctx.setLineDash([]);
      ctx.font = `bold ${Math.round(r * 1.4)}px system-ui, sans-serif`;
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("?", x + r * 1.9, y - r * 1.3);
    } else {
      ring(ctx, x, y, r, "rgba(200,200,200,0.9)", 1.5 * line);
      cross(ctx, x, y, r * 0.6, "rgba(230,230,230,0.95)", 1.5 * line);
    }
  }
  ctx.setLineDash([]);
}

function ring(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, width: number) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function cross(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, color: string, width: number) {
  ctx.beginPath();
  ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
  ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function plus(ctx: CanvasRenderingContext2D, x: number, y: number, s: number, color: string, width: number) {
  ctx.beginPath();
  ctx.moveTo(x - s, y); ctx.lineTo(x + s, y);
  ctx.moveTo(x, y - s); ctx.lineTo(x, y + s);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}
