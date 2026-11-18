// Aperio-style .svs (tiled JPEG TIFF) without loading the whole file: level 0 (the first image) is read one tile row
// at a time, decoded by the browser's JPEG decoder and resampled on the fly to the sizes the pipeline needs.
import { xTaps } from "./imaging.js?v=dev";

export const isSvs = (name) => /\.svs$/i.test(name);

const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 6: 1, 7: 1, 8: 2, 9: 4, 16: 8, 17: 8 };
const MPP_RE = /MPP\s*=\s*([\d.]+)/;  // predict.py _MPP_RE

async function readAt(file, off, len) {
  return new Uint8Array(await file.slice(off, off + len).arrayBuffer());
}

// tags of the first IFD: {tag: number[] | Uint8Array (BYTE/UNDEFINED) | string (ASCII)}
async function firstIfd(file) {
  const head = new DataView((await readAt(file, 0, 16)).buffer);
  const order = head.getUint16(0);
  if (order !== 0x4949 && order !== 0x4d4d) throw new Error(`${file.name} is not a TIFF-based .svs file`);
  const le = order === 0x4949, magic = head.getUint16(2, le), big = magic === 43;
  if (magic !== 42 && !big) throw new Error(`${file.name} is not a TIFF-based .svs file`);
  const u64 = (dv, o) => Number(dv.getBigUint64(o, le));
  const ifdOff = big ? u64(head, 8) : head.getUint32(4, le);
  const nDv = new DataView((await readAt(file, ifdOff, 8)).buffer);
  const n = big ? u64(nDv, 0) : nDv.getUint16(0, le), entSize = big ? 20 : 12, inline = big ? 8 : 4;
  const ents = new DataView((await readAt(file, ifdOff + (big ? 8 : 2), n * entSize)).buffer);
  const tags = {};
  for (let e = 0; e < n; e++) {
    const o = e * entSize, tag = ents.getUint16(o, le), type = ents.getUint16(o + 2, le);
    const count = big ? u64(ents, o + 4) : ents.getUint32(o + 4, le), size = (TYPE_SIZE[type] ?? 0) * count;
    if (!size) continue;
    const vo = o + (big ? 12 : 8);
    const raw = size <= inline ? new Uint8Array(ents.buffer, vo, size)
      : await readAt(file, big ? u64(ents, vo) : ents.getUint32(vo, le), size);
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    if (type === 2) tags[tag] = new TextDecoder("latin1").decode(raw).replace(/\0+$/, "");
    else if (type === 1 || type === 6 || type === 7) tags[tag] = raw.slice();
    else {
      const get = { 3: (i) => dv.getUint16(i * 2, le), 8: (i) => dv.getInt16(i * 2, le), 4: (i) => dv.getUint32(i * 4, le),
        9: (i) => dv.getInt32(i * 4, le), 16: (i) => u64(dv, i * 8), 17: (i) => u64(dv, i * 8) }[type];
      tags[tag] = Array.from({ length: count }, (_, i) => get(i));
    }
  }
  return tags;
}

// level 0 layout + native um/px from the ImageDescription (null if absent)
export async function openSvs(file) {
  const t = await firstIfd(file);
  const w = t[256]?.[0], h = t[257]?.[0], compression = t[259]?.[0] ?? 1, spp = t[277]?.[0] ?? 1;
  if (!w || !h) throw new Error(`${file.name}: no image size in the .svs header`);
  if (compression === 33003 || compression === 33005)
    throw new Error(`${file.name}: JPEG 2000 .svs files can't be read in the browser; use the desktop app`);
  if (compression !== 7 && compression !== 1)
    throw new Error(`${file.name}: .svs compression ${compression} is not supported in the browser`);
  if (spp !== 3 || (t[258] ?? [8]).some((b) => b !== 8) || (t[284]?.[0] ?? 1) !== 1)
    throw new Error(`${file.name}: only 8-bit RGB .svs files are supported`);
  const tiled = Boolean(t[322]);
  const tw = tiled ? t[322][0] : w, th = tiled ? t[323][0] : (t[278]?.[0] ?? h);
  const offsets = tiled ? t[324] : t[273], counts = tiled ? t[325] : t[279];
  if (!offsets || !counts) throw new Error(`${file.name}: no tile offsets in the .svs header`);
  const m = MPP_RE.exec(t[270] ?? "");
  return { file, w, h, tw, th, offsets, counts, compression, photometric: t[262]?.[0] ?? 2,
    tables: t[347] ?? null, mpp: m ? Number(m[1]) || null : null };
}

const SOI = new Uint8Array([0xff, 0xd8]);
// Adobe APP14 with transform 0: the tile's components are RGB, not YCbCr (libtiff decodes them that way too)
const ADOBE_RGB = new Uint8Array([0xff, 0xee, 0, 14, 0x41, 0x64, 0x6f, 0x62, 0x65, 0, 100, 0, 0, 0, 0, 0]);

function tileJpeg(svs, data) {
  const parts = [SOI];
  if (svs.tables) parts.push(svs.tables.subarray(2, svs.tables.length - 2));
  if (svs.photometric === 2) parts.push(ADOBE_RGB);
  parts.push(data.subarray(2));
  return new Blob(parts, { type: "image/jpeg" });
}

// tiles [c0, c1) of tile row r (whole row as one read when the tiles are stored together)
async function readTiles(svs, r, c0, c1) {
  const { offsets, counts, file } = svs, first = r * Math.ceil(svs.w / svs.tw) + c0, n = c1 - c0;
  let lo = Infinity, hi = 0, sum = 0;
  for (let i = first; i < first + n; i++) {
    lo = Math.min(lo, offsets[i]); hi = Math.max(hi, offsets[i] + counts[i]); sum += counts[i];
  }
  if (hi - lo <= 2 * sum) {
    const buf = await readAt(file, lo, hi - lo);
    return Array.from({ length: n }, (_, c) => buf.subarray(offsets[first + c] - lo, offsets[first + c] - lo + counts[first + c]));
  }
  return Promise.all(Array.from({ length: n }, (_, c) => readAt(file, offsets[first + c], counts[first + c])));
}

// RGBA of tile columns [c0, c1) of tile row r into band (stride = band width in px); returns the row count
export async function decodeTiles(svs, r, c0, c1, band, stride, ctx) {
  const { w, h, tw, th } = svs, rows = Math.min(th, h - r * th), tiles = await readTiles(svs, r, c0, c1);
  const decoded = svs.compression === 7
    ? await Promise.all(tiles.map((d) => createImageBitmap(tileJpeg(svs, d),
      { colorSpaceConversion: "none", premultiplyAlpha: "none" })))
    : tiles;
  for (let c = c0; c < c1; c++) {
    const cw = Math.min(tw, w - c * tw), bx = (c - c0) * tw;
    if (svs.compression === 7) {
      const bmp = decoded[c - c0];
      ctx.drawImage(bmp, 0, 0);
      bmp.close();
      const px = ctx.getImageData(0, 0, cw, rows).data;
      for (let y = 0; y < rows; y++) band.set(px.subarray(y * cw * 4, (y + 1) * cw * 4), (y * stride + bx) * 4);
    } else {
      const raw = decoded[c - c0];
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cw; x++) {
          const i = (y * tw + x) * 3, o = (y * stride + bx + x) * 4;
          band[o] = raw[i]; band[o + 1] = raw[i + 1]; band[o + 2] = raw[i + 2]; band[o + 3] = 255;
        }
      }
    }
  }
  return rows;
}

// the lossless copy is tiled like the .svs; TIFF tiles must be multiples of 16 px
export const canCopyTiff = (svs) => svs.tw % 16 === 0 && svs.th % 16 === 0 && svs.tw < svs.w + 16;

const WORKER_URL = new URL("./svs_worker.js?v=dev", import.meta.url);

// sizes: {key: [w, h]} -> {key: RGB image}, INTER_AREA / INTER_LINEAR from level 0 exactly like imaging.resize.
// Tile columns are split over workers (destination columns are independent); tiff: optional TiffWriter;
// onRow(done, total) as tile rows complete
export async function readSvsScaled(svs, sizes, { tiff = null, onRow = null } = {}) {
  const { w, h, tw, th } = svs, across = Math.ceil(w / tw), down = Math.ceil(h / th);
  const n = Math.max(1, Math.min(across, (globalThis.navigator?.hardwareConcurrency ?? 4) - 1, 8));
  const jobs = Array.from({ length: n }, (_, j) => {
    const c0 = Math.floor((j * across) / n), c1 = Math.floor(((j + 1) * across) / n);
    return { c0, c1, s0: c0 * tw, s1: Math.min(c1 * tw, w), need: Math.min(c1 * tw, w) - 1, sizes: [] };
  });
  for (const [key, [dw, dh]] of Object.entries(sizes)) {
    if (dw === w && dh === h) {
      for (const j of jobs) j.sizes.push({ key, dw, dh, copy: true, x0: j.s0, x1: j.s1 });
      continue;
    }
    const xt = xTaps(w, h, dw, dh);
    let dx = 0;
    for (const j of jobs) {
      const x0 = dx;
      while (dx < dw && xt[dx][0][0] < j.s1) j.need = Math.max(j.need, xt[dx++].at(-1)[0]);
      j.sizes.push({ key, dw, dh, x0, x1: dx });
    }
  }
  const { offsets, counts, compression, photometric, tables } = svs;
  const meta = { file: svs.file, w, h, tw, th, offsets, counts, compression, photometric, tables };
  const done = new Array(n).fill(0), workers = [];
  let reported = 0;
  try {
    const parts = await Promise.all(jobs.map((j, i) => new Promise((resolve, reject) => {
      const wk = new Worker(WORKER_URL, { type: "module" });
      workers.push(wk);
      wk.onerror = (e) => reject(new Error(e.message || ".svs reader crashed (likely out of memory)"));
      wk.onmessage = ({ data }) => {
        if (data.type === "tile") tiff.writeTile(data.index, data.data).catch(reject);
        else if (data.type === "row") {
          done[i] = data.r + 1;
          const m = Math.min(...done);
          if (m > reported) onRow?.((reported = m), down);
        } else if (data.type === "done") resolve(data.outs);
        else if (data.type === "error") reject(new Error(data.message));
      };
      wk.postMessage({ svs: meta, c0: j.c0, c1: j.c1, tc1: Math.min(across, Math.floor(j.need / tw) + 1),
        sizes: j.sizes, tiff: Boolean(tiff) });
    })));
    await tiff?.finish();
    return Object.fromEntries(Object.entries(sizes).map(([key, [dw, dh]]) => {
      const px = new Uint8Array(dw * dh * 3);
      for (const outs of parts) {
        const { x0, x1, px: part } = outs[key], cols = (x1 - x0) * 3;
        for (let y = 0; y < dh && cols; y++) px.set(part.subarray(y * cols, (y + 1) * cols), (y * dw + x0) * 3);
        outs[key] = null;
      }
      return [key, { w: dw, h: dh, px, cn: 3 }];
    }));
  } finally {
    for (const wk of workers) wk.terminate();
  }
}

// lossless tiled RGB TIFF (deflate) streamed to a FileSystemWritableFileStream in any tile order, like
// predict.py's TIFF_<folder>/<stem>.tif copy of an .svs
export class TiffWriter {
  constructor(writable, w, h, tw, th) {
    this.writable = writable; this.w = w; this.h = h; this.tw = tw; this.th = th;
    const n = Math.ceil(w / tw) * Math.ceil(h / th);
    this.offsets = new Array(n).fill(0); this.counts = new Array(n).fill(0);
    this.pos = 8;
    this.chain = writable.write(new Uint8Array(8));
  }

  writeTile(index, data) {
    this.chain = this.chain.then(async () => {
      if (this.pos + data.byteLength > 0xfff00000) throw new Error("TIFF copy would exceed 4 GB");
      await this.writable.write(data);
      this.offsets[index] = this.pos; this.counts[index] = data.byteLength;
      this.pos += data.byteLength;
    });
    return this.chain;
  }

  async finish() {
    await this.chain;
    const ifdOff = this.pos + (this.pos & 1);
    // tag, type (3 SHORT, 4 LONG), values
    const ents = [[256, 4, [this.w]], [257, 4, [this.h]], [258, 3, [8, 8, 8]], [259, 3, [8]], [262, 3, [2]],
      [277, 3, [3]], [284, 3, [1]], [322, 4, [this.tw]], [323, 4, [this.th]], [324, 4, this.offsets],
      [325, 4, this.counts]];
    const ifdSize = 2 + ents.length * 12 + 4, buf = new DataView(new ArrayBuffer(ifdSize)), extraParts = [];
    let extra = ifdOff + ifdSize;
    buf.setUint16(0, ents.length, true);
    ents.forEach(([tag, type, vals], e) => {
      const o = 2 + e * 12, size = type === 3 ? 2 : 4;
      const put = (dv, at, v) => (size === 2 ? dv.setUint16(at, v, true) : dv.setUint32(at, v, true));
      buf.setUint16(o, tag, true); buf.setUint16(o + 2, type, true); buf.setUint32(o + 4, vals.length, true);
      if (vals.length * size <= 4) vals.forEach((v, i) => put(buf, o + 8 + i * size, v));
      else {
        const arr = new DataView(new ArrayBuffer(vals.length * size));
        vals.forEach((v, i) => put(arr, i * size, v));
        buf.setUint32(o + 8, extra, true);
        extraParts.push(new Uint8Array(arr.buffer));
        extra += arr.byteLength;
      }
    });
    if (this.pos & 1) await this.writable.write(new Uint8Array(1));
    await this.writable.write(new Uint8Array(buf.buffer));
    for (const p of extraParts) await this.writable.write(p);
    const head = new DataView(new ArrayBuffer(8));
    head.setUint16(0, 0x4949); head.setUint16(2, 42, true); head.setUint32(4, ifdOff, true);
    await this.writable.write({ type: "write", position: 0, data: new Uint8Array(head.buffer) });
    await this.writable.close();
    this.done = true;
  }
}
