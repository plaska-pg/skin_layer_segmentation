// Image work kept off the page's main thread: Review previews and re-measure inputs, and for Run the decoding,
// resizing, preview, mask and measuring input of each image.
import { crop, decode, previewJpeg, pyRound, resize, thumbnail, tissueBox, toRGB, upsampleArgmax, workSize }
  from "./imaging.js?v=dev";
import { isSvs, openSvs, readSvsScaled } from "./svs.js?v=dev";

const UTIF_URL = "https://cdn.jsdelivr.net/npm/utif@3.1.0/UTIF.js";
const UTIF_SRI = "sha384-RyBmXHdfZ/Uon+ud+/AqSyWpUWnKYt2tkRG/P4gWoRUGDU+qIAV3tGBPNlYTBZEF";

async function loadUtif() {
  const res = await fetch(UTIF_URL);
  if (!res.ok) throw new Error(`${UTIF_URL}: HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-384", buf));
  if ("sha384-" + btoa(String.fromCharCode(...digest)) !== UTIF_SRI) throw new Error(`integrity check failed for ${UTIF_URL}`);
  // classic (non-module) script that sets self.UTIF
  new Function(new TextDecoder().decode(buf))();
  boundLzw(self.UTIF);
}

// UTIF's LZW decoder reads codes until an end code and never checks the strip's length, so a strip whose end
// code it misses (the last strip of S_EX 8_24 l10c4.svs.tif) loops forever. Same decoder, stopped at the end
// of the strip's bytes and of its rows.
function boundLzw(UTIF) {
  const decompress = UTIF.decode._decompress;
  let endBit = 0, outEnd = 0;
  UTIF.decode._decompress = function (img, ifds, data, off, len, cmpr, tgt, toff, fo) {
    if (cmpr === 5) {
      const tiled = img.t322 != null, w = tiled ? img.t322[0] : img.width;
      const rows = tiled ? img.t323[0] : Math.min(img.t278?.[0] ?? img.height, img.height);
      const bipp = img.t258 ? Math.min(32, img.t258[0]) * img.t258.length : (img.t277?.[0] ?? 1);
      endBit = (off + len) * 8;
      outEnd = Math.min(tgt.length, toff + rows * Math.ceil(w * bipp / 8));
    }
    return decompress.call(this, img, ifds, data, off, len, cmpr, tgt, toff, fo);
  };
  let tab = null, tln = null, chr = null;
  UTIF.decode._decodeLZW = function (data, off, tgt, toff) {
    if (!tab) {
      tab = new Uint32Array(0xffff); tln = new Uint16Array(0xffff); chr = new Uint8Array(2e6);
      for (let i = 0; i < 256; i++) { chr[i << 2] = i; tab[i] = i << 2; tln[i] = 1; }
    }
    const copy = UTIF.decode._copyData;
    let totl = 258, chrl = 258 << 2, bits = 9, boff = off << 3, Code = 0, OldCode = 0;
    const read = () => {
      const v = (data[boff >>> 3] << 16) | (data[(boff + 8) >>> 3] << 8) | data[(boff + 16) >>> 3];
      const c = (v >> (24 - (boff & 7) - bits)) & ((1 << bits) - 1);
      boff += bits;
      return c;
    };
    while (boff + bits <= endBit && toff < outEnd) {
      Code = read();
      if (Code === 257) break;
      if (Code === 256) {
        bits = 9; totl = 258; chrl = 258 << 2;
        if (boff + bits > endBit) break;
        Code = read();
        if (Code === 257) break;
        tgt[toff++] = Code;
      } else if (Code < totl) {
        const cd = tab[Code], cl = tln[Code];
        copy(chr, cd, tgt, toff, cl); toff += cl;
        if (OldCode >= totl) { tab[totl] = chrl; chr[tab[totl]] = cd[0]; tln[totl] = 1; chrl = (chrl + 1 + 3) & ~0x03; totl++; }
        else {
          tab[totl] = chrl;
          const nit = tab[OldCode];
          let nil = tln[OldCode];
          copy(chr, nit, chr, chrl, nil);
          chr[chrl + nil] = chr[cd]; nil++;
          tln[totl] = nil; totl++;
          chrl = (chrl + nil + 3) & ~0x03;
        }
        if (totl + 1 === (1 << bits)) bits++;
      } else if (Code === totl) {
        if (OldCode >= totl) { tab[totl] = chrl; tln[totl] = 0; totl++; }
        else {
          tab[totl] = chrl;
          const nit = tab[OldCode];
          let nil = tln[OldCode];
          copy(chr, nit, chr, chrl, nil);
          chr[chrl + nil] = chr[chrl]; nil++;
          tln[totl] = nil; totl++;
          copy(chr, chrl, tgt, toff, nil); toff += nil;
          chrl = (chrl + nil + 3) & ~0x03;
        }
        if (totl + 1 === (1 << bits)) bits++;
      } else break;
      OldCode = Code;
    }
  };
}

const ms = (t0) => `${((performance.now() - t0) / 1000).toFixed(2)} s`;

// Run: the image at the measuring resolution, kept here between "prepare" / "hold" and "finish"
let held = null;

async function preview(img, maxSide, logs) {
  const t0 = performance.now();
  try {
    const blob = await previewJpeg(img, maxSide);
    logs.push(`review preview: ${ms(t0)}`);
    return blob;
  } catch (e) {
    logs.push(`review preview failed (${e.message ?? e}); Review will build it`);
    return null;
  }
}

// Run, non-.svs: decode, then everything the page needs before segmenting
async function prepare({ id, file, um, scale, maxMP, postUm, rotSize, maxSide, dispSide }, report) {
  const logs = [];
  let t0 = performance.now();
  let img = await decode(file);
  const iw = img.w, ih = img.h;
  logs.push(`${file.name}: ${iw} x ${ih} px, decode ${ms(t0)}`);
  const disp = thumbnail(img, dispSide);
  report("Finding the orientation", 0.1, disp);
  const sq = resize(img, rotSize, rotSize);
  const blob = await preview(img, maxSide, logs);
  const tw = Math.max(1, pyRound(iw * scale)), th = Math.max(1, pyRound(ih * scale));
  const [ww, wh] = workSize(iw, ih, um, maxMP, postUm);
  if (ww !== iw || wh !== ih) {
    t0 = performance.now();
    held = resize(img, ww, wh);
    logs.push(`${(iw * ih / 1e6).toFixed(0)} MP > ${maxMP} MP: measuring at ${ww} x ${wh} px, ${ms(t0)}`);
  } else held = img;
  let net = img;
  if (scale !== 1 && (tw !== iw || th !== ih)) {
    report("Resizing", 0.12);
    t0 = performance.now();
    net = resize(img, tw, th);
    logs.push(`resize to ${tw} x ${th} px: ${ms(t0)}`);
  } else if (held === img) {
    // both sent to the page and kept here
    const rgb = toRGB(img);
    net = { w: iw, h: ih, px: rgb === img.px ? rgb.slice() : rgb, cn: 3 };
  }
  img = null;
  return [{ id, iw, ih, ww, wh, net, sq: sq.px, blob, logs },
    [net.px.buffer, sq.px.buffer]];
}

// Run, .svs: the page read it (it also writes the TIFF copy); keep the measuring image here
async function hold({ id, work, maxSide, dispSide }, report) {
  held = work;
  report("Finding the orientation", 0.1, thumbnail(work, dispSide));
  const logs = [];
  const blob = await preview(work, maxSide, logs);
  return [{ id, blob, logs }, []];
}

// Run: probabilities -> mask at the measuring resolution; a huge image is cropped to its tissue
function finish({ id, probs, wsum, nCls, nw, nh, maxMP, workUm, cropMarginUm, detSide }) {
  if (!held) throw new Error("the image was dropped by the image worker");
  let work = held;
  held = null;
  const logs = [];
  let t0 = performance.now();
  const plane = nw * nh;
  for (let c = 0; c < nCls; c++) for (let o = 0; o < plane; o++) probs[c * plane + o] /= wsum[o];
  const mask = upsampleArgmax(probs, nCls, nw, nh, work.w, work.h);
  logs.push(`mask at ${work.w} x ${work.h} px: ${ms(t0)}`);
  let region = null, wm = mask, det = null;
  if (work.w * work.h > maxMP * 1e6) {
    t0 = performance.now();
    const box = tissueBox(work, { mask, block: Math.max(8, Math.round(20 / workUm)),
      margin: Math.round(cropMarginUm / workUm) });
    if (box) {
      region = [...box, work.w, work.h];
      // the detectors and quality model downscale the whole image to their max side anyway
      const f = Math.min(1, detSide / Math.max(work.w, work.h));
      const d = resize(work, Math.max(1, pyRound(work.w * f)), Math.max(1, pyRound(work.h * f)));
      det = { rgb: d.px, w: d.w, h: d.h };
      wm = crop({ w: work.w, h: work.h, px: mask, cn: 1 }, box).px;
      work = crop(work, box);
      logs.push(`measuring the tissue only: ${box[2]} x ${box[3]} px at (${box[0]}, ${box[1]}), ${ms(t0)}`);
    }
  }
  const rgb = toRGB(work);
  return [{ id, rgb, mask: wm, w: work.w, h: work.h, region, det, logs },
    [rgb.buffer, wm.buffer, ...(det ? [det.rgb.buffer] : [])]];
}

async function handle(data, report) {
  if (data.op === "prepare") return prepare(data, report);
  if (data.op === "hold") return hold(data, report);
  if (data.op === "finish") return finish(data);
  if (data.op === "drop") { held = null; return [{ id: data.id }, []]; }
  return source(data);
}

async function source({ id, op, file, maxSide, um, maxMP, postUm, rotSize, labelBox, cropMarginUm }) {
  // .svs: read band by band straight to the needed sizes; other formats are decoded whole
  const svs = isSvs(file.name) ? await openSvs(file) : null;
  let img = svs ? null : await decode(file);
  const iw = svs?.w ?? img.w, ih = svs?.h ?? img.h;
  if (op === "preview") {
    if (svs) {
      const f = Math.min(1, maxSide / Math.max(iw, ih));
      img = (await readSvsScaled(svs, { p: [Math.max(1, Math.round(iw * f)), Math.max(1, Math.round(ih * f))] })).p;
    }
    const blob = await previewJpeg(img, maxSide);
    return [{ id, iw, ih, blob }, []];
  }
  // re-measure input: 224 px square for the rotation model (if needed) + image at the measuring resolution
  const [ww, wh] = workSize(iw, ih, um, maxMP, postUm);
  let sq = null, work;
  if (svs) {
    ({ sq = null, work } = await readSvsScaled(svs, { work: [ww, wh], ...(rotSize ? { sq: [rotSize, rotSize] } : {}) }));
  } else {
    sq = rotSize ? resize(img, rotSize, rotSize) : null;
    work = ww === iw && wh === ih ? img : resize(img, ww, wh);
  }
  img = null;
  const workUm = work.w === iw ? um : um * iw / work.w, sqPx = sq?.px ?? null;
  // as processFile: a huge image is measured on its tissue + the labels only
  let region = null;
  if (work.w * work.h > maxMP * 1e6) {
    const lb = labelBox && [labelBox[0] * work.w, labelBox[1] * work.h, (labelBox[2] - labelBox[0]) * work.w,
      (labelBox[3] - labelBox[1]) * work.h];
    const box = tissueBox(work, { extra: lb, block: Math.max(8, Math.round(20 / workUm)),
      margin: Math.round((cropMarginUm ?? 0) / workUm) });
    if (box) {
      region = [...box, work.w, work.h];
      work = crop(work, box);
    }
  }
  const rgb = toRGB(work);
  return [{ id, iw, ih, w: work.w, h: work.h, rgb, workUm, sq: sqPx, region },
    [rgb.buffer, ...(sqPx ? [sqPx.buffer] : [])]];
}

// one image at a time: a 100 MP TIFF needs ~1 GB while decoding
let utif = null, queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      await (utif ??= loadUtif());
      const report = (label, frac, disp) => postMessage({ id: data.id, type: "progress", label, frac, disp },
        disp ? [disp.px.buffer] : []);
      const [msg, transfer] = await handle(data, report);
      postMessage(msg, transfer);
    } catch (e) {
      utif = globalThis.UTIF ? utif : null;
      postMessage({ id: data.id, error: String(e.message ?? e) });
    }
  });
};
