// One column strip of an .svs for svs.readSvsScaled: decodes tile columns [c0, tc1) row by row and resamples the
// destination columns it was given; deflated TIFF tiles of its own columns [c0, c1) go back to the page.
import { Resampler } from "./imaging.js?v=dev";
import { decodeTiles } from "./svs.js?v=dev";

class Copy {
  constructor(w, h, x0, x1) {
    this.cols = x1 - x0; this.y = 0;
    this.px = new Uint8Array(this.cols * h * 3);
  }

  push(px, rows, cn, stride) {
    for (let r = 0; r < rows; r++, this.y++) {
      for (let x = 0, i = r * stride * cn, o = this.y * this.cols * 3; x < this.cols; x++, i += cn, o += 3) {
        this.px[o] = px[i]; this.px[o + 1] = px[i + 1]; this.px[o + 2] = px[i + 2];
      }
    }
  }

  result() {
    return { px: this.px };
  }
}

async function deflateTile(band, stride, bx, rows, tw, th) {
  const rgb = new Uint8Array(tw * th * 3), cols = Math.min(tw, stride - bx);
  for (let y = 0; y < rows; y++) {
    for (let x = 0, i = (y * stride + bx) * 4, o = y * tw * 3; x < cols; x++, i += 4, o += 3) {
      rgb[o] = band[i]; rgb[o + 1] = band[i + 1]; rgb[o + 2] = band[i + 2];
    }
  }
  return new Uint8Array(await new Response(new Blob([rgb]).stream()
    .pipeThrough(new CompressionStream("deflate"))).arrayBuffer());
}

self.onmessage = async ({ data: { svs, c0, c1, tc1, sizes, tiff } }) => {
  try {
    const { w, h, tw, th } = svs, across = Math.ceil(w / tw), down = Math.ceil(h / th);
    const sx0 = c0 * tw, stride = Math.min(tc1 * tw, w) - sx0;
    const sinks = sizes.map((s) => (s.copy ? new Copy(w, h, s.x0, s.x1)
      : new Resampler(w, h, s.dw, s.dh, { x0: s.x0, x1: s.x1, sx0 })));
    const band = new Uint8ClampedArray(stride * th * 4);
    const canvas = new OffscreenCanvas(tw, th), ctx = canvas.getContext("2d", { willReadFrequently: true });
    for (let r = 0; r < down; r++) {
      const rows = await decodeTiles(svs, r, c0, tc1, band, stride, ctx);
      for (const s of sinks) s.push(band, rows, 4, stride);
      if (tiff) {
        for (let c = c0; c < c1; c++) {
          const data = await deflateTile(band, stride, (c - c0) * tw, rows, tw, th);
          postMessage({ type: "tile", index: r * across + c, data }, [data.buffer]);
        }
      }
      postMessage({ type: "row", r });
    }
    const outs = {}, transfer = [];
    sizes.forEach((s, i) => {
      const { px } = sinks[i].result();
      outs[s.key] = { x0: s.x0, x1: s.x1, px };
      transfer.push(px.buffer);
    });
    postMessage({ type: "done", outs }, transfer);
  } catch (e) {
    postMessage({ type: "error", message: String(e.message ?? e) });
  }
};
