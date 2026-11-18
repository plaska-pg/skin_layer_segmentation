// Image decoding and OpenCV-equivalent resizing, shared by the page (app.js) and image_worker.js.
// TIFF decoding needs the UTIF global (index.html script tag; image_worker.js loads it itself).

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

export function resize(img, dw, dh) {
  const { w, h, px, cn } = img;
  const area = dw < w || dh < h;
  const xt = (area ? areaTab : linearTab)(w, dw), yt = (area ? areaTab : linearTab)(h, dh);
  const out = new Uint8Array(dw * dh * 3), row = new Float32Array(dw * 3), acc = new Float32Array(dw * 3);
  for (let dy = 0; dy < dh; dy++) {
    acc.fill(0);
    for (const [sy, ay] of yt[dy]) {
      row.fill(0);
      const base = sy * w * cn;
      for (let dx = 0; dx < dw; dx++) {
        for (const [sx, ax] of xt[dx]) {
          const i = base + sx * cn;
          row[dx * 3] += ax * px[i]; row[dx * 3 + 1] += ax * px[i + 1]; row[dx * 3 + 2] += ax * px[i + 2];
        }
      }
      for (let i = 0; i < acc.length; i++) acc[i] += ay * row[i];
    }
    const o = dy * dw * 3;
    for (let i = 0; i < acc.length; i++) out[o + i] = Math.min(255, Math.max(0, Math.round(acc[i])));
  }
  return { w: dw, h: dh, px: out, cn: 3 };
}

export function toRGB(img) {
  if (img.cn === 3) return img.px;
  const out = new Uint8Array(img.w * img.h * 3);
  for (let i = 0, o = 0; o < out.length; i += 4, o += 3) {
    out[o] = img.px[i]; out[o + 1] = img.px[i + 1]; out[o + 2] = img.px[i + 2];
  }
  return out;
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
