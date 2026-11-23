// Image decoding and OpenCV-equivalent resizing, shared by the page (app.js) and the workers.
// TIFF decoding needs the UTIF global (image_worker.js loads it).

// Python round(): half to even
export const pyRound = (v) => (Math.abs(v % 1) === 0.5 ? 2 * Math.round(v / 2) : Math.round(v));

export async function decode(file) {
  if (/\.tiff?$/i.test(file.name)) {
    const { UTIF } = globalThis;
    const buf = await file.arrayBuffer();
    const ifds = UTIF.decode(buf);
    const ifd = ifds.reduce((a, b) => (b.t256[0] * b.t257[0] > a.t256[0] * a.t257[0] ? b : a));
    UTIF.decodeImage(buf, ifd);
    return { w: ifd.width, h: ifd.height, px: UTIF.toRGBA8(ifd), cn: 4 };
  }
  const bmp = await createImageBitmap(file, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height), ctx = canvas.getContext("2d");
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const w = bmp.width, h = bmp.height;
  bmp.close();
  canvas.width = canvas.height = 0;
  return { w, h, px, cn: 4 };
}

// OpenCV computeResizeAreaTab: source indices and weights for each destination index
function areaTab(ssize, dsize) {
  const scale = ssize / dsize;
  const tab = [];
  for (let d = 0; d < dsize; d++) {
    const f1 = d * scale, f2 = f1 + scale, cell = Math.min(scale, ssize - f1);
    let s2 = Math.min(Math.floor(f2), ssize - 1);
    const s1 = Math.min(Math.ceil(f1), s2);
    const e = [];
    if (s1 - f1 > 1e-3) e.push([s1 - 1, (s1 - f1) / cell]);
    for (let s = s1; s < s2; s++) e.push([s, 1 / cell]);
    if (f2 - s2 > 1e-3) e.push([s2, Math.min(f2 - s2, 1, cell) / cell]);
    tab.push(e);
  }
  return tab;
}

// cv2.INTER_LINEAR pixel-centre taps with BORDER_REPLICATE
function linearTab(ssize, dsize) {
  const tab = [];
  for (let d = 0; d < dsize; d++) {
    const s = (d + 0.5) * (ssize / dsize) - 0.5, i = Math.floor(s), f = s - i;
    const c = (k) => Math.min(Math.max(k, 0), ssize - 1);
    tab.push([[c(i), 1 - f], [c(i + 1), f]]);
  }
  return tab;
}

// source columns read by each destination column of resize(sw x sh -> dw x dh): [[sx, weight], ...] ascending
export function xTaps(sw, sh, dw, dh) {
  return (dw < sw || dh < sh ? areaTab : linearTab)(sw, dw);
}

// cv2.resize (INTER_AREA down, INTER_LINEAR up) fed source rows in order, so huge images can be resampled band by
// band while decoding; float32 accumulation in the same order as a whole-image pass. Optionally only destination
// columns [x0, x1) from bands whose first column is source column sx0 (columns are independent: same result)
export class Resampler {
  constructor(sw, sh, dw, dh, { x0 = 0, x1 = dw, sx0 = 0 } = {}) {
    const tab = dw < sw || dh < sh ? areaTab : linearTab;
    const xt = tab(sw, dw).slice(x0, x1), yt = tab(sh, dh), cols = x1 - x0;
    this.sw = sw; this.sh = sh; this.dw = cols; this.dh = dh; this.y = 0; this.next = 0;
    this.xStart = new Int32Array(cols + 1);
    xt.forEach((t, dx) => { this.xStart[dx + 1] = this.xStart[dx] + t.length; });
    this.xIdx = new Int32Array(this.xStart[cols]);
    this.xW = new Float64Array(this.xStart[cols]);
    xt.flat().forEach(([sx, ax], i) => { this.xIdx[i] = sx - sx0; this.xW[i] = ax; });
    this.bySrc = Array.from({ length: sh }, () => []);
    this.last = new Int32Array(dh);
    yt.forEach((t, dy) => {
      for (const [sy, ay] of t) this.bySrc[sy].push(dy, ay);
      this.last[dy] = Math.max(0, ...t.map(([sy]) => sy));
    });
    this.row = new Float32Array(cols * 3);
    this.open = new Map();
    this.out = new Uint8Array(cols * dh * 3);
  }

  // rows of `stride` pixels with cn (3 or 4) channels, continuing from the previous push
  push(px, rows, cn, stride = this.sw) {
    const { dw, row, xStart, xIdx, xW } = this, f = Math.fround;
    for (let r = 0; r < rows; r++, this.y++) {
      const contrib = this.bySrc[this.y];
      if (contrib.length) {
        const base = r * stride * cn;
        for (let dx = 0; dx < dw; dx++) {
          let r0 = 0, r1 = 0, r2 = 0;
          for (let t = xStart[dx]; t < xStart[dx + 1]; t++) {
            const i = base + xIdx[t] * cn, ax = xW[t];
            r0 = f(r0 + ax * px[i]); r1 = f(r1 + ax * px[i + 1]); r2 = f(r2 + ax * px[i + 2]);
          }
          row[dx * 3] = r0; row[dx * 3 + 1] = r1; row[dx * 3 + 2] = r2;
        }
        for (let c = 0; c < contrib.length; c += 2) {
          const dy = contrib[c], ay = contrib[c + 1];
          let acc = this.open.get(dy);
          if (!acc) this.open.set(dy, acc = new Float32Array(dw * 3));
          for (let i = 0; i < acc.length; i++) acc[i] += ay * row[i];
        }
      }
      while (this.next < this.dh && this.last[this.next] <= this.y) {
        const acc = this.open.get(this.next), o = this.next * dw * 3;
        if (acc) for (let i = 0; i < acc.length; i++) this.out[o + i] = Math.min(255, Math.max(0, Math.round(acc[i])));
        this.open.delete(this.next++);
      }
    }
  }

  result() {
    return { w: this.dw, h: this.dh, px: this.out, cn: 3 };
  }
}

export function resize(img, dw, dh) {
  const r = new Resampler(img.w, img.h, dw, dh);
  r.push(img.px, img.h, img.cn);
  return r.result();
}

// image size measured at (interface/app.py MAX_POST_MP / POST_UM_PER_PX): large images are reduced to postUm
export function workSize(w, h, um, maxMP, postUm) {
  if (w * h <= maxMP * 1e6 || um >= postUm) return [w, h];
  const s = um / postUm;
  return [Math.max(1, pyRound(w * s)), Math.max(1, pyRound(h * s))];
}

// Bounding box [x0, y0, w, h] of the tissue (HSV saturation > sat, as measurement.tissue_area_um2) plus any
// mask / box pixels, ignoring specks far smaller than the section; null if it would not save much.
// Blocks of ~20 um; components under minFrac of the largest are dropped; margin in px
export function tissueBox(img, { mask = null, extra = null, sat = 20, block = 32, minFrac = 0.02, margin = 0 } = {}) {
  const { w, h, px, cn } = img, bw = Math.ceil(w / block), bh = Math.ceil(h / block);
  const count = new Uint32Array(bw * bh), hit = new Uint8Array(bw * bh);
  for (let y = 0; y < h; y++) {
    const by = Math.floor(y / block) * bw;
    for (let x = 0, i = y * w * cn; x < w; x++, i += cn) {
      const r = px[i], g = px[i + 1], b = px[i + 2], mx = Math.max(r, g, b);
      // cv2 HSV S (8-bit) = 255 * (max - min) / max, rounded
      if (mx && Math.round((255 * (mx - Math.min(r, g, b))) / mx) > sat) count[by + Math.floor(x / block)]++;
      if (mask && mask[y * w + x]) hit[by + Math.floor(x / block)] = 1;
    }
  }
  const on = new Uint8Array(bw * bh);
  for (let b = 0; b < on.length; b++) on[b] = hit[b] || count[b] * 4 >= block * block ? 1 : 0;
  // 8-connected components of the block grid
  const comp = new Int32Array(bw * bh).fill(-1), boxes = [];
  for (let s = 0; s < on.length; s++) {
    if (!on[s] || comp[s] >= 0) continue;
    const stack = [s], bx = [bw, bh, -1, -1, 0];
    comp[s] = boxes.length;
    while (stack.length) {
      const b = stack.pop(), x = b % bw, y = (b - x) / bw;
      bx[0] = Math.min(bx[0], x); bx[1] = Math.min(bx[1], y); bx[2] = Math.max(bx[2], x); bx[3] = Math.max(bx[3], y);
      bx[4]++;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy, n = ny * bw + nx;
        if (nx >= 0 && ny >= 0 && nx < bw && ny < bh && on[n] && comp[n] < 0) { comp[n] = boxes.length; stack.push(n); }
      }
    }
    boxes.push(bx);
  }
  if (!boxes.length) return null;
  const big = Math.max(...boxes.map((b) => b[4]));
  let x0 = w, y0 = h, x1 = 0, y1 = 0;
  for (const [bx0, by0, bx1, by1, n] of boxes) {
    if (n < minFrac * big) continue;
    x0 = Math.min(x0, bx0 * block); y0 = Math.min(y0, by0 * block);
    x1 = Math.max(x1, (bx1 + 1) * block); y1 = Math.max(y1, (by1 + 1) * block);
  }
  if (extra) {
    x0 = Math.min(x0, extra[0]); y0 = Math.min(y0, extra[1]);
    x1 = Math.max(x1, extra[0] + extra[2]); y1 = Math.max(y1, extra[1] + extra[3]);
  }
  x0 = Math.max(0, Math.floor(x0 - margin)); y0 = Math.max(0, Math.floor(y0 - margin));
  x1 = Math.min(w, Math.ceil(x1 + margin)); y1 = Math.min(h, Math.ceil(y1 + margin));
  return (x1 - x0) * (y1 - y0) < 0.9 * w * h ? [x0, y0, x1 - x0, y1 - y0] : null;
}

// region [x0, y0, cw, ch] of an image ({w, h, px, cn}) or of a single-channel array (cn = 1)
export function crop(img, [x0, y0, cw, ch]) {
  const { w, px, cn } = img, out = new px.constructor(cw * ch * cn);
  for (let y = 0; y < ch; y++) out.set(px.subarray(((y0 + y) * w + x0) * cn, ((y0 + y) * w + x0 + cw) * cn), y * cw * cn);
  return { w: cw, h: ch, px: out, cn };
}

export function toRGB(img) {
  if (img.cn === 3) return img.px;
  const out = new Uint8Array(img.w * img.h * 3);
  for (let i = 0, o = 0; o < out.length; i += 4, o += 3) {
    out[o] = img.px[i]; out[o + 1] = img.px[i + 1]; out[o + 2] = img.px[i + 2];
  }
  return out;
}

// nearest-neighbour RGBA copy at most maxSide px, for drawing on a canvas
export function thumbnail(img, maxSide) {
  const s = Math.min(1, maxSide / Math.max(img.w, img.h));
  const w = Math.max(1, Math.round(img.w * s)), h = Math.max(1, Math.round(img.h * s));
  const px = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(img.h - 1, Math.floor((y + 0.5) / s));
    for (let x = 0; x < w; x++) {
      const si = (sy * img.w + Math.min(img.w - 1, Math.floor((x + 0.5) / s))) * img.cn, o = 4 * (y * w + x);
      px[o] = img.px[si]; px[o + 1] = img.px[si + 1]; px[o + 2] = img.px[si + 2]; px[o + 3] = 255;
    }
  }
  return { w, h, px };
}

// predict._upsample_argmax: cv2.remap INTER_LINEAR, which quantises coordinates to 1/32 px
function remapTab(ssize, dsize) {
  const i0 = new Int32Array(dsize), i1 = new Int32Array(dsize), f = new Float32Array(dsize);
  for (let d = 0; d < dsize; d++) {
    const X = Math.round(((d + 0.5) * (ssize / dsize) - 0.5) * 32), i = X >> 5;
    i0[d] = Math.min(Math.max(i, 0), ssize - 1);
    i1[d] = Math.min(Math.max(i + 1, 0), ssize - 1);
    f[d] = (X & 31) / 32;
  }
  return { i0, i1, f };
}

export function upsampleArgmax(probs, nCls, w, h, ow, oh) {
  const mask = new Uint8Array(ow * oh), plane = w * h;
  if (ow === w && oh === h) {
    for (let o = 0; o < plane; o++) {
      let best = -1;
      for (let c = 0; c < nCls; c++) if (probs[c * plane + o] > best) { best = probs[c * plane + o]; mask[o] = c; }
    }
    return mask;
  }
  const X = remapTab(w, ow), Y = remapTab(h, oh);
  for (let oy = 0; oy < oh; oy++) {
    const r0 = Y.i0[oy] * w, r1 = Y.i1[oy] * w, fy = Y.f[oy];
    for (let ox = 0; ox < ow; ox++) {
      const a = r0 + X.i0[ox], b = r0 + X.i1[ox], c0 = r1 + X.i0[ox], d = r1 + X.i1[ox], fx = X.f[ox];
      let best = -1, arg = 0;
      for (let c = 0, q = 0; c < nCls; c++, q += plane) {
        const top = probs[q + a] + fx * (probs[q + b] - probs[q + a]);
        const bot = probs[q + c0] + fx * (probs[q + d] - probs[q + c0]);
        const v = top + fy * (bot - top);
        if (v > best) { best = v; arg = c; }
      }
      mask[oy * ow + ox] = arg;
    }
  }
  return mask;
}

function toImageData(img) {
  if (img.cn === 4) {
    const px = img.px instanceof Uint8ClampedArray ? img.px
      : new Uint8ClampedArray(img.px.buffer, img.px.byteOffset, img.w * img.h * 4);
    return new ImageData(px, img.w, img.h);
  }
  const out = new ImageData(img.w, img.h);
  for (let i = 0, o = 0; o < out.data.length; i += 3, o += 4) {
    out.data[o] = img.px[i]; out.data[o + 1] = img.px[i + 1]; out.data[o + 2] = img.px[i + 2]; out.data[o + 3] = 255;
  }
  return out;
}

// the Review editor's display image (interface/seg_editor.py: at most 4096 px, JPEG)
export async function previewJpeg(img, maxSide = 4096, quality = 0.85) {
  const f = Math.min(1, maxSide / Math.max(img.w, img.h));
  const w = Math.max(1, Math.round(img.w * f)), h = Math.max(1, Math.round(img.h * f));
  let bmp;
  try {
    bmp = await createImageBitmap(toImageData(img), { resizeWidth: w, resizeHeight: h, resizeQuality: "high" });
  } catch {
    // browser limits on huge bitmaps: shrink in JS first
    bmp = await createImageBitmap(toImageData(f < 1 ? resize(img, w, h) : img));
  }
  const canvas = new OffscreenCanvas(w, h);
  canvas.getContext("2d").drawImage(bmp, 0, 0);
  bmp.close();
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality });
  canvas.width = canvas.height = 0;
  return blob;
}
