// Decodes source images off the page's main thread: Review previews and re-measure inputs.
import { decode, previewJpeg, pyRound, resize, toRGB } from "./imaging.js?v=dev";

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
  let img = await decode(file);
  const iw = img.w, ih = img.h;
  if (op === "preview") {
    const blob = await previewJpeg(img, maxSide);
    return [{ id, iw, ih, blob }, []];
  }
  // re-measure input: 224 px square for the rotation model (if needed) + image at the measuring resolution
  const sq = rotSize ? resize(img, rotSize, rotSize).px : null;
  let work = img, workUm = um;
  if (iw * ih > maxMP * 1e6 && um < postUm) {
    const s = um / postUm;
    work = resize(img, Math.max(1, pyRound(iw * s)), Math.max(1, pyRound(ih * s)));
    workUm = um * iw / work.w;
  }
  img = null;
  const rgb = toRGB(work);
  return [{ id, iw, ih, w: work.w, h: work.h, rgb, workUm, sq }, [rgb.buffer, ...(sq ? [sq.buffer] : [])]];
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
