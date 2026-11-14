// GPU implementation of the detector with plain WebGL2 fragment shaders (no libraries).
//
// Layout: a CxHxW tensor is a float RGBA texture; channels 4g..4g+3 form "group" g, an HxW tile.
// Tiles are laid out in a grid (cols x rows) to respect the max texture size.
// Weights per conv op live in their own float texture (see packWeights).
import type { ConvOp, Model, Tensor } from "./model.ts";
import { lastUse, shapes } from "./model.ts";

interface Layout {
  c: number;
  h: number;
  w: number;
  groups: number;
  cols: number;
  rows: number;
}

interface GpuTensor extends Layout {
  tex: WebGLTexture;
  fb: WebGLFramebuffer;
}

const VERT = `#version 300 es
in vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

const COMMON = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out vec4 o;
struct T { ivec2 size; int cols; };
vec4 fetch(sampler2D s, T t, int g, int x, int y) {
  if (x < 0 || y < 0 || x >= t.size.x || y >= t.size.y) return vec4(0.0);
  return texelFetch(s, ivec2((g % t.cols) * t.size.x + x, (g / t.cols) * t.size.y + y), 0);
}
uniform T u_out;
uniform int u_groups;
// this fragment's output group and position
bool locate(out int g, out int x, out int y) {
  ivec2 f = ivec2(gl_FragCoord.xy);
  int col = f.x / u_out.size.x, row = f.y / u_out.size.y;
  x = f.x - col * u_out.size.x;
  y = f.y - row * u_out.size.y;
  g = row * u_out.cols + col;
  return g < u_groups;
}
`;

function convShader(op: ConvOp): string {
  const head = `#define K ${op.k}\n#define S ${op.stride}\n#define P ${op.pad}\n#define D ${op.dil}\n`;
  const tail = `
uniform sampler2D u_src; uniform T u_in;
uniform sampler2D u_w;
uniform sampler2D u_res; uniform int u_hasRes;
uniform int u_relu;
uniform int u_gin;
void main() {
  int g, x, y;
  if (!locate(g, x, y)) { o = vec4(0.0); return; }
  ${op.depthwise ? `
  vec4 acc = texelFetch(u_w, ivec2(K * K, g), 0);
  for (int ky = 0; ky < K; ky++) for (int kx = 0; kx < K; kx++)
    acc += texelFetch(u_w, ivec2(ky * K + kx, g), 0) * fetch(u_src, u_in, g, x * S - P + kx * D, y * S - P + ky * D);
  ` : `
  vec4 acc = texelFetch(u_w, ivec2(4 * u_gin * K * K, g), 0);
  for (int gi = 0; gi < u_gin; gi++) for (int ky = 0; ky < K; ky++) for (int kx = 0; kx < K; kx++) {
    vec4 v = fetch(u_src, u_in, gi, x * S - P + kx * D, y * S - P + ky * D);
    int b = 4 * ((gi * K + ky) * K + kx);
    acc += mat4(texelFetch(u_w, ivec2(b, g), 0), texelFetch(u_w, ivec2(b + 1, g), 0),
                texelFetch(u_w, ivec2(b + 2, g), 0), texelFetch(u_w, ivec2(b + 3, g), 0)) * v;
  }
  `}
  if (u_hasRes == 1) acc += fetch(u_res, u_out, g, x, y);
  if (u_relu == 1) acc = max(acc, 0.0);
  o = acc;
}`;
  return COMMON.replace("#version 300 es\n", `#version 300 es\n${head}`) + tail;
}

const UPADD = COMMON + `
uniform sampler2D u_a; uniform sampler2D u_b; uniform T u_bt;
void main() {
  int g, x, y;
  if (!locate(g, x, y)) { o = vec4(0.0); return; }
  o = fetch(u_a, u_out, g, x, y) + fetch(u_b, u_bt, g, x / 2, y / 2);
}`;

/** Weight texture for a conv op.
 *  dense:     width 4*Gin*K*K + 1, height Gout. Texel (4*((gi*K+ky)*K+kx)+j, go) is column j of the
 *             4x4 block mapping input group gi to output group go at tap (ky,kx); last texel = bias.
 *  depthwise: width K*K + 1, height G. Texel (tap, g) = per-channel weights; last texel = bias. */
function packWeights(op: ConvOp, wts: Float32Array): { w: number; h: number; data: Float32Array } {
  const taps = op.k * op.k;
  const gout = Math.ceil(op.cout / 4);
  if (op.depthwise) {
    const w = taps + 1;
    const data = new Float32Array(w * gout * 4);
    for (let c = 0; c < op.cout; c++) {
      const g = c >> 2, lane = c & 3;
      for (let t = 0; t < taps; t++) data[(g * w + t) * 4 + lane] = wts[op.w + c * taps + t];
      data[(g * w + taps) * 4 + lane] = wts[op.b + c];
    }
    return { w, h: gout, data };
  }
  const gin = Math.ceil(op.cin / 4);
  const w = 4 * gin * taps + 1;
  const data = new Float32Array(w * gout * 4);
  for (let o = 0; o < op.cout; o++) {
    const go = o >> 2, lane = o & 3;
    for (let i = 0; i < op.cin; i++) {
      const gi = i >> 2, j = i & 3;
      for (let t = 0; t < taps; t++) {
        const col = 4 * (gi * taps + t) + j;
        data[(go * w + col) * 4 + lane] = wts[op.w + (o * op.cin + i) * taps + t];
      }
    }
    data[(go * w + w - 1) * 4 + lane] = wts[op.b + o];
  }
  return { w, h: gout, data };
}

export class WebGLEngine {
  readonly name = "GPU (WebGL2)";
  readonly gl: WebGL2RenderingContext;
  readonly maxTex: number;
  private programs = new Map<string, WebGLProgram>();
  private weightTex = new Map<number, WebGLTexture>(); // by op index
  private model: Model;
  private pool: GpuTensor[] = [];

  /** Throws if this device can't render to float textures (then use the CPU engine). */
  constructor(model: Model, canvas: HTMLCanvasElement | OffscreenCanvas = new OffscreenCanvas(1, 1)) {
    const gl = canvas.getContext("webgl2", { antialias: false, depth: false, stencil: false, premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: "high-performance" }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("WebGL2 is not available");
    if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("float render targets are not available");
    this.gl = gl;
    this.model = model;
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW); // one big triangle
    model.meta.ops.forEach((op, i) => {
      if (op.op === "conv") {
        this.program(convShader(op));
        const p = packWeights(op, model.weights);
        this.weightTex.set(i, this.floatTexture(p.w, p.h, p.data));
      }
    });
    this.program(UPADD);
  }

  private program(src: string): WebGLProgram {
    const cached = this.programs.get(src);
    if (cached) return cached;
    const gl = this.gl;
    const sh = (type: number, s: string) => {
      const x = gl.createShader(type)!;
      gl.shaderSource(x, s);
      gl.compileShader(x);
      if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error("shader: " + gl.getShaderInfoLog(x));
      return x;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, src));
    gl.bindAttribLocation(p, 0, "p");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("program: " + gl.getProgramInfoLog(p));
    this.programs.set(src, p);
    return p;
  }

  private floatTexture(w: number, h: number, data: Float32Array | null): WebGLTexture {
    const gl = this.gl;
    if (w > this.maxTex || h > this.maxTex) throw new Error(`texture ${w}x${h} exceeds the GPU limit ${this.maxTex}`);
    const t = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
    return t;
  }

  private layout(c: number, h: number, w: number): Layout {
    const groups = Math.ceil(c / 4);
    let cols = Math.max(1, Math.min(groups, Math.ceil(Math.sqrt((groups * h) / w))));
    while (cols < groups && Math.ceil(groups / cols) * h > this.maxTex) cols++;
    const rows = Math.ceil(groups / cols);
    if (cols * w > this.maxTex || rows * h > this.maxTex) throw new Error("image too large for this GPU");
    return { c, h, w, groups, cols, rows };
  }

  private acquire(l: Layout): GpuTensor {
    const i = this.pool.findIndex((t) => t.cols * t.w === l.cols * l.w && t.rows * t.h === l.rows * l.h);
    if (i >= 0) {
      const t = this.pool.splice(i, 1)[0];
      return { ...l, tex: t.tex, fb: t.fb };
    }
    const gl = this.gl;
    const tex = this.floatTexture(l.cols * l.w, l.rows * l.h, null);
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error("float framebuffer incomplete");
    return { ...l, tex, fb };
  }

  private release(t: GpuTensor) {
    this.pool.push(t);
  }

  /** Free all GPU memory held for intermediate tensors (weights stay). */
  trim() {
    for (const t of this.pool) {
      this.gl.deleteTexture(t.tex);
      this.gl.deleteFramebuffer(t.fb);
    }
    this.pool = [];
  }

  private setT(p: WebGLProgram, name: string, t: Layout) {
    const gl = this.gl;
    gl.uniform2i(gl.getUniformLocation(p, `${name}.size`), t.w, t.h);
    gl.uniform1i(gl.getUniformLocation(p, `${name}.cols`), t.cols);
  }

  private bindTex(p: WebGLProgram, name: string, unit: number, tex: WebGLTexture) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(gl.getUniformLocation(p, name), unit);
  }

  private draw(p: WebGLProgram, dst: GpuTensor) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
    gl.viewport(0, 0, dst.cols * dst.w, dst.rows * dst.h);
    gl.uniform2i(gl.getUniformLocation(p, "u_out.size"), dst.w, dst.h);
    gl.uniform1i(gl.getUniformLocation(p, "u_out.cols"), dst.cols);
    gl.uniform1i(gl.getUniformLocation(p, "u_groups"), dst.groups);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  run(input: Tensor): Tensor {
    const gl = this.gl;
    const { meta } = this.model;
    const sh = shapes(meta, input.h, input.w);
    const last = lastUse(meta);
    // upload input (3 channels -> one RGBA group)
    const inL = this.layout(input.c, input.h, input.w);
    const inT = this.acquire(inL);
    const px = new Float32Array(inL.cols * inL.w * inL.rows * inL.h * 4);
    const n = input.h * input.w;
    for (let c = 0; c < input.c; c++) for (let i = 0; i < n; i++) px[i * 4 + c] = input.data[c * n + i];
    gl.bindTexture(gl.TEXTURE_2D, inT.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, inL.cols * inL.w, inL.rows * inL.h, gl.RGBA, gl.FLOAT, px);

    const env = new Map<string, GpuTensor>([[meta.input, inT]]);
    meta.ops.forEach((op, i) => {
      const [c, h, w] = sh.get(op.dst)!;
      const dst = this.acquire(this.layout(c, h, w));
      if (op.op === "conv") {
        const p = this.program(convShader(op));
        gl.useProgram(p);
        const src = env.get(op.src)!;
        this.bindTex(p, "u_src", 0, src.tex);
        this.setT(p, "u_in", src);
        this.bindTex(p, "u_w", 1, this.weightTex.get(i)!);
        const res = op.res ? env.get(op.res)! : null;
        this.bindTex(p, "u_res", 2, res ? res.tex : src.tex);
        gl.uniform1i(gl.getUniformLocation(p, "u_hasRes"), res ? 1 : 0);
        gl.uniform1i(gl.getUniformLocation(p, "u_relu"), op.relu ? 1 : 0);
        gl.uniform1i(gl.getUniformLocation(p, "u_gin"), Math.ceil(op.cin / 4));
        this.draw(p, dst);
      } else {
        const p = this.program(UPADD);
        gl.useProgram(p);
        const a = env.get(op.a)!, b = env.get(op.b)!;
        this.bindTex(p, "u_a", 0, a.tex);
        this.bindTex(p, "u_b", 1, b.tex);
        this.setT(p, "u_bt", b);
        this.draw(p, dst);
      }
      env.set(op.dst, dst);
      for (const [t, j] of last) {
        if (j === i && t !== meta.output && env.has(t)) {
          this.release(env.get(t)!);
          env.delete(t);
        }
      }
    });

    // read back the output and unpack groups -> CHW
    const out = env.get(meta.output)!;
    const W = out.cols * out.w, H = out.rows * out.h;
    const raw = new Float32Array(W * H * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, out.fb);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, raw);
    this.release(out);
    const on = out.h * out.w;
    const data = new Float32Array(out.c * on);
    for (let ch = 0; ch < out.c; ch++) {
      const g = ch >> 2, lane = ch & 3;
      const ox = (g % out.cols) * out.w, oy = Math.floor(g / out.cols) * out.h;
      for (let y = 0; y < out.h; y++) {
        const row = ((oy + y) * W + ox) * 4 + lane;
        for (let x = 0; x < out.w; x++) data[ch * on + y * out.w + x] = raw[row + x * 4];
      }
    }
    return { c: out.c, h: out.h, w: out.w, data };
  }
}
