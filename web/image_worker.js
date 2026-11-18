// Decodes source images off the page's main thread: Review previews and re-measure inputs.
import { decode, previewJpeg, resize, toRGB, workSize } from "./imaging.js?v=dev";
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
}

async function handle({ id, op, file, maxSide, um, maxMP, postUm, rotSize }) {
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
  const workUm = work.w === iw ? um : um * iw / work.w, rgb = toRGB(work), sqPx = sq?.px ?? null;
  return [{ id, iw, ih, w: work.w, h: work.h, rgb, workUm, sq: sqPx }, [rgb.buffer, ...(sqPx ? [sqPx.buffer] : [])]];
}

// one image at a time: a 100 MP TIFF needs ~1 GB while decoding
let utif = null, queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      await (utif ??= loadUtif());
      const [msg, transfer] = await handle(data);
      postMessage(msg, transfer);
    } catch (e) {
      utif = globalThis.UTIF ? utif : null;
      postMessage({ id: data.id, error: String(e.message ?? e) });
    }
  });
};
