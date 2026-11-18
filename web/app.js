// Browser version of interface/app.py: Run / Review / Results with a working folder (File System Access API).
// Segmentation: port of predict.predict_mask (resize to the model's um/px, overlapping tiles cross-faded with a
// linear ramp, bilinear upsampling of probabilities, argmax). Measuring and the Review editor's merge / cut /
// re-measure run postprocess.py & co. in Pyodide workers (worker.js); the editor is interface/seg_editor.js.
import mountEditor from "../interface/seg_editor.js?v=dev";
import { decode, previewJpeg, pyRound, resize, toRGB } from "./imaging.js?v=dev";

// ?v=<commit> on every app file (set by the Pages build) keeps cached files from different deploys apart
const VERSION = new URL(import.meta.url).searchParams.get("v") ?? "";
const { ort } = window;
const $ = (id) => document.getElementById(id);
const COLORS = [null, [230, 80, 60], [60, 170, 90], [70, 110, 230]];
// larger images are measured at POST_UM_PER_PX so Pyodide stays under its 4 GB memory limit
const MAX_POST_MP = 45, POST_UM_PER_PX = 0.4;
const WORKER_RECYCLE_MB = 1024;
const EDITOR_MAX_SIDE = 4096, EDITOR_HEIGHT = 720;  // interface/seg_editor.py
// display-size copy of each source image, written next to the results so Review opens without decoding the source
const PREVIEW_DIR = "review_cache";
const CLASS_COLORS = { SC: "#00c8ff", Epidermis: "#ffc800", gland: "#00ff00" };
const SEMSEG_FILES = {
  "training/__init__.py": "../training/__init__.py", "training/semseg/__init__.py": "../training/semseg/__init__.py",
  "training/semseg/add_segmentations.py": "../training/semseg/add_segmentations.py",
};
// Pyodide workers: measuring (process_image, re-measure) and the light Review editor engine (merge / cut / save)
const MEASURE_PY = {
  module: "review", modules: ["pipeline", "review"],
  packages: ["numpy", "scipy", "scikit-image", "opencv-python", "pyyaml"],
  files: {
    "pipeline.py": "pipeline.py", "obb.py": "obb.py", "review.py": "review.py", "postprocess.py": "../postprocess.py",
    "measurement.py": "../measurement.py", "config.py": "../config.py", "base_config.yaml": "../base_config.yaml",
    "training/__init__.py": "../training/__init__.py", "training/qc_sections.py": "../training/qc_sections.py",
  },
};
const EDIT_PY = {
  module: "review_edit", modules: ["review_edit"], packages: ["numpy", "opencv-python", "pyyaml"],
  files: { "review_edit.py": "review_edit.py", ...SEMSEG_FILES },
};

function sameOrigin(path) {
  const url = new URL(path, location.href);
  if (url.origin !== location.origin) throw new Error("model must be served from this site");
  return url;
}

const modelBase = sameOrigin(new URLSearchParams(location.search).get("model") ?? "../runs/web/seg");
let manifest, measureWorker, editPy, currentBackend, toggleDefaultsApplied = false;
const models = {}, modelUrls = {};
const sessions = {};
const state = { root: null, preds: [], pred: null, scope: "folder", busy: false, stop: false, modelsReady: false,
  page: "run" };
const logLines = [];

function setHeader(text, cls = "") {
  $("header-status").textContent = text;
  $("header-status").className = "caption " + cls;
}
function log(line) {
  logLines.push(line);
  if (logLines.length > 400) logLines.splice(0, logLines.length - 400);
  $("log").textContent = logLines.slice(-25).join("\n");
}
let runLabel = "";
// progress of the current step (model download, tile, measuring stage) goes to the page header
function status(text) {
  setHeader(state.busy ? `\u25B6 ${runLabel} \u00B7 ${text}` : text);
}
function progress(frac, label) {
  runLabel = label;
  $("progress-bar").style.width = `${Math.round(Math.min(1, frac) * 100)}%`;
  $("progress-text").textContent = label;
}
const tick = () => new Promise((r) => setTimeout(r));
const ms = (t0) => `${((performance.now() - t0) / 1000).toFixed(2)} s`;

async function fetchBytes(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    status(`${label} ${(got / 2 ** 20).toFixed(0)}${total ? ` / ${(total / 2 ** 20).toFixed(0)}` : ""} MB`,
      total ? got / total : undefined);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

async function getSession(backend, key = "seg") {
  const id = `${key}-${backend}`;
  if (!sessions[id]) {
    const t0 = performance.now();
    status(`Starting ${backend}...`);
    const bytes = models[key] ?? await fetchBytes(modelUrls[key], `Loading ${key} model`);
    sessions[id] = await ort.InferenceSession.create(bytes, { executionProviders: [backend] });
    // ONNX Runtime keeps its own copy; another backend re-fetches (normally from the browser cache)
    delete models[key];
    log(`${key} session ${backend}: ${ms(t0)}`);
  }
  return sessions[id];
}

function ramp(n, r) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.min(Math.max(Math.min(i + 0.5, n - i - 0.5) / Math.max(r, 1), 1e-3), 1);
  return a;
}

function starts(n, tile, stride) {
  const s = new Set([Math.max(0, n - tile)]);
  for (let v = 0; v < Math.max(1, n - tile); v += stride) s.add(v);
  return [...s].sort((a, b) => a - b);
}

// cv2.BORDER_REFLECT (edge pixel repeated)
const reflect = (i, n) => { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - 1 - i; return i; };

// predict.TTA_OPS: (quarter turns, horizontal flip) per pass, predictions mapped back and averaged
const TTA_OPS = {
  none: [[0, false]],
  flip: [[0, false], [0, true], [2, false], [2, true]],
  d4: [0, 1, 2, 3].flatMap((k) => [[k, false], [k, true]]),
};

// torch.rot90(x, k, (H, W)) of planar (C, H, W) data; returns [data, H', W']
function rot90(src, C, H, W, k) {
  k = ((k % 4) + 4) % 4;
  if (k === 0) return [src, H, W];
  const [H2, W2] = k === 2 ? [H, W] : [W, H];
  const out = new Float32Array(src.length);
  for (let c = 0; c < C; c++) {
    const s = c * H * W;
    for (let i = 0; i < H2; i++) {
      for (let j = 0; j < W2; j++) {
        const [y, x] = k === 1 ? [j, W - 1 - i] : k === 2 ? [H - 1 - i, W - 1 - j] : [H - 1 - j, i];
        out[s + i * W2 + j] = src[s + y * W + x];
      }
    }
  }
  return [out, H2, W2];
}

function flipW(src, C, H, W) {
  const out = new Float32Array(src.length);
  for (let r = 0; r < C * H; r++) for (let x = 0; x < W; x++) out[r * W + x] = src[r * W + W - 1 - x];
  return out;
}

async function ttaProbs(session, data, PH, PW, nCls, tta) {
  const ops = TTA_OPS[tta] ?? TTA_OPS.none;
  let acc = null;
  for (const [k, flip] of ops) {
    let [x, h, w] = rot90(data, 3, PH, PW, k);
    if (flip) x = flipW(x, 3, h, w);
    const out = await session.run({ [manifest.input.name]: new ort.Tensor("float32", x, [1, 3, h, w]) });
    let p = out[manifest.output.name].data;
    if (flip) p = flipW(p, nCls, h, w);
    [p] = rot90(p, nCls, h, w, -k);
    if (ops.length === 1) return p;
    if (!acc) acc = new Float32Array(p.length);
    for (let i = 0; i < p.length; i++) acc[i] += p[i];
  }
  for (let i = 0; i < acc.length; i++) acc[i] /= ops.length;
  return acc;
}

async function tiledProbs(session, img, nCls, tta = "none") {
  const { w, h, px, cn } = img, tile = manifest.tile;
  const m = Math.floor(tile * manifest.tile_overlap / 2), stride = tile - 2 * m;
  const ys = starts(h, tile, stride), xs = starts(w, tile, stride);
  const probs = new Float32Array(nCls * w * h), wsum = new Float32Array(w * h);
  const n = ys.length * xs.length, plane = w * h;
  let k = 0, runMs = 0;
  for (const y0 of ys) {
    for (const x0 of xs) {
      const ph = Math.min(tile, h - y0), pw = Math.min(tile, w - x0);
      const PH = ph + ((32 - (ph % 32)) % 32), PW = pw + ((32 - (pw % 32)) % 32);
      const data = new Float32Array(3 * PH * PW);
      for (let y = 0; y < PH; y++) {
        const sy = y0 + reflect(y, ph);
        for (let x = 0; x < PW; x++) {
          const i = (sy * w + x0 + reflect(x, pw)) * cn, o = y * PW + x;
          data[o] = px[i]; data[PH * PW + o] = px[i + 1]; data[2 * PH * PW + o] = px[i + 2];
        }
      }
      const t0 = performance.now();
      const p = await ttaProbs(session, data, PH, PW, nCls, tta);
      runMs += performance.now() - t0;
      const wy = ramp(ph, 2 * m), wx = ramp(pw, 2 * m);
      for (let y = 0; y < ph; y++) {
        for (let x = 0; x < pw; x++) {
          const wt = wy[y] * wx[x], o = (y0 + y) * w + x0 + x;
          for (let c = 0; c < nCls; c++) probs[c * plane + o] += p[c * PH * PW + y * PW + x] * wt;
          wsum[o] += wt;
        }
      }
      status(`Tile ${++k} / ${n}`, k / n);
      await tick();
    }
  }
  for (let c = 0; c < nCls; c++) for (let o = 0; o < plane; o++) probs[c * plane + o] /= wsum[o];
  log(`tiles: ${n} x ${tile}px, TTA ${tta}, inference ${(runMs / 1000).toFixed(2)} s (${(runMs / n).toFixed(0)} ms/tile)`);
  return probs;
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

function upsampleArgmax(probs, nCls, w, h, ow, oh) {
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

function drawPreview(img, mask) {
  const s = Math.min(1, 1200 / img.w), pw = Math.max(1, Math.round(img.w * s)), ph = Math.max(1, Math.round(img.h * s));
  const canvas = $("run-preview"), ctx = canvas.getContext("2d");
  canvas.hidden = false;
  $("run-latest").hidden = true;
  canvas.width = pw; canvas.height = ph;
  const im = ctx.createImageData(pw, ph);
  for (let y = 0; y < ph; y++) {
    const sy = Math.min(img.h - 1, Math.floor((y + 0.5) / s));
    for (let x = 0; x < pw; x++) {
      const sx = Math.min(img.w - 1, Math.floor((x + 0.5) / s)), si = sy * img.w + sx, o = 4 * (y * pw + x);
      const col = COLORS[mask[si]];
      for (let c = 0; c < 3; c++) {
        const v = img.px[si * img.cn + c];
        im.data[o + c] = col ? 0.55 * v + 0.45 * col[c] : v;
      }
      im.data[o + 3] = 255;
    }
  }
  ctx.putImageData(im, 0, 0);
}

async function prepareBackend() {
  let backend = $("backend").value, session;
  try {
    session = await getSession(backend);
  } catch (e) {
    if (backend !== "webgpu") throw e;
    log(`WebGPU unavailable (${e.message}); using CPU`);
    backend = "wasm";
    session = await getSession(backend);
  }
  currentBackend = backend;
  return session;
}

function setBusy(busy) {
  $("run").disabled = busy || !state.root || !state.modelsReady;
  $("run-progress").hidden = !busy;
}

// segment + measure one image; shows the mask preview and measurements, returns the worker's result
async function processFile(file, mpp, session, folderName = "", opts = readOptions()) {
  const backend = currentBackend;
  const tAll = performance.now();
  let t0 = performance.now();
  status(`${file.name}: reading...`);
  await tick();
  let img = await decode(file);
  log(`${file.name}: ${img.w} x ${img.h} px, decode ${ms(t0)}`);

  t0 = performance.now();
  const k = await rotationK(backend, img);
  log(`rotation: ${k * 90} deg clockwise to upright, ${ms(t0)}`);

  t0 = performance.now();
  let preview = null;
  try {
    preview = { blob: await previewJpeg(img, EDITOR_MAX_SIDE), width: img.w, height: img.h, k };
    log(`review preview: ${ms(t0)}`);
  } catch (e) {
    log(`review preview failed (${describeError(e)}); Review will build it`);
  }

  const scale = mpp / manifest.reference_um_per_px;
  const tw = Math.max(1, pyRound(img.w * scale)), th = Math.max(1, pyRound(img.h * scale));
  let net = img;
  if (scale !== 1 && (tw !== img.w || th !== img.h)) {
    t0 = performance.now();
    status(`${file.name}: resizing...`);
    await tick();
    net = resize(img, tw, th);
    log(`resize to ${tw} x ${th} px (${manifest.reference_um_per_px} um/px): ${ms(t0)}`);
  }
  let work = img, workMpp = mpp;
  if (img.w * img.h > MAX_POST_MP * 1e6 && mpp < POST_UM_PER_PX) {
    t0 = performance.now();
    status(`${file.name}: reducing resolution for measuring...`);
    await tick();
    const s = mpp / POST_UM_PER_PX;
    work = resize(img, Math.max(1, pyRound(img.w * s)), Math.max(1, pyRound(img.h * s)));
    workMpp = mpp * img.w / work.w;
    log(`${(img.w * img.h / 1e6).toFixed(0)} MP > ${MAX_POST_MP} MP: measuring at ${work.w} x ${work.h} px ` +
      `(${workMpp.toFixed(4)} um/px), ${ms(t0)}`);
  }
  img = null;
  const nCls = manifest.classes.length;
  let probs = await tiledProbs(session, net, nCls, opts.tta);

  t0 = performance.now();
  status(`${file.name}: building mask...`);
  await tick();
  const mask = upsampleArgmax(probs, nCls, net.w, net.h, work.w, work.h);
  probs = net = null;
  log(`mask at ${work.w} x ${work.h} px: ${ms(t0)}`);

  const stem = file.name.replace(/\.[^.]+$/, "");
  drawPreview(work, mask);
  log(`segmentation (${backend}): ${ms(tAll)}`);

  t0 = performance.now();
  status(`${file.name}: measuring...`);
  const rgb = toRGB(work);
  const res = await measureWorker.run(
    { name: file.name, rgb, mask, w: work.w, h: work.h, umPerPx: workMpp, k, lastModified: file.lastModified,
      quality: JSON.stringify(manifest.quality), folderName, options: opts.detect },
    [rgb.buffer, mask.buffer],
    (line) => { log(line); status(`Measuring: ${line.replace(/^[^:]*: /, "")}`); });
  log(`measuring: ${ms(t0)} (measurement engine memory ${res.heapMB} MB)`);
  log(`total: ${ms(tAll)}`);
  return { res, stem, workMpp, preview };
}

// ---------------------------------------------------------------- folder mode (File System Access API)
const IMAGE_RE = /\.(tiff?|jpe?g|png|bmp|webp)$/i;

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [header = [], ...body] = rows.filter((r) => r.some((v) => v !== ""));
  return { fields: header, rows: body.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""]))) };
}

async function readText(dir, name) {
  try {
    return await (await (await dir.getFileHandle(name)).getFile()).text();
  } catch (e) {
    if (e.name === "NotFoundError") return null;
    throw e;
  }
}

async function writeFile(dir, path, data) {
  const parts = path.split("/");
  for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, { create: true });
  const w = await (await dir.getFileHandle(parts.at(-1), { create: true })).createWritable();
  await w.write(data);
  await w.close();
}

// worker output name -> path inside predicted_<folder>, as predict.py lays it out
function outputPath(stem, f) {
  return {
    [`${stem}_measurement.jpg`]: f, [`${stem}_steps.jpg`]: f,
    [`${stem}.txt`]: `postprocessed_segmentations/${stem}.txt`, [`raw/${stem}.txt`]: `raw_segmentations/${stem}.txt`,
  }[f];
}

async function readBytes(dir, name) {
  try {
    return new Uint8Array(await (await (await dir.getFileHandle(name)).getFile()).arrayBuffer());
  } catch (e) {
    if (e.name === "NotFoundError") return null;
    throw e;
  }
}

async function subdir(dir, name) {
  try {
    return await dir.getDirectoryHandle(name);
  } catch (e) {
    if (e.name === "NotFoundError" || e.name === "TypeMismatchError") return null;
    throw e;
  }
}

function setImg(img, blob) {
  if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
  img.dataset.url = blob ? URL.createObjectURL(blob) : "";
  if (blob) img.src = img.dataset.url; else img.removeAttribute("src");
  img.hidden = !blob;
}

// ---------------------------------------------------------------- Run page (interface/app.py run_page / run_jobs)
async function runJobs() {
  const mpp = Number($("mpp").value);
  if (!(mpp > 0)) { setHeader("Enter the \u00b5m per pixel of the images first.", "err"); return; }
  const root = state.root, opts = readOptions(), resume = $("resume").checked;
  const limit = Math.max(0, Math.floor(Number($("limit").value) || 0));
  let jobs;
  if (state.scope === "one") {
    const name = $("run-image").value;
    if (!name) return;
    jobs = [{ dir: root, path: "", files: [await root.getFileHandle(name)],
      outName: `predicted_${name.replace(/\.[^.]+$/, "")}` }];
  } else {
    jobs = (await imageFolders(root, "", state.scope === "subfolders"))
      .map((f) => ({ ...f, files: limit ? f.files.slice(0, limit) : f.files, outName: `predicted_${f.dir.name}` }));
  }
  const total = jobs.reduce((n, j) => n + j.files.length, 0);
  if (!total) { setHeader(`No images in ${root.name}.`, "err"); return; }
  state.busy = true;
  state.stop = false;
  logLines.length = 0;
  $("log").textContent = "";
  $("stop").disabled = false;
  $("last-run").hidden = true;
  $("run-preview").hidden = true;
  setImg($("run-latest"), null);
  setBusy(true);
  $("run-caption").textContent = `${jobs.length} folder(s), ${total} image(s). Results go to ` +
    `predicted_<name> inside each folder; switching pages does not stop the run.`;
  progress(0, "Starting...");
  const counts = { done: 0, skipped: 0, failed: 0, n: 0, total }, outs = [], warnings = [];
  try {
    const session = await prepareBackend();
    for (const [i, job] of jobs.entries()) {
      if (state.stop) break;
      outs.push(await processJob(job, i, jobs.length, mpp, session, opts, resume, counts, warnings));
    }
  } catch (e) {
    console.error(e);
    warnings.push(`Error: ${e.message}`);
    log(String(e.stack ?? e));
  }
  state.busy = false;
  setBusy(false);
  progress(1, `${counts.done} processed, ${counts.skipped} already done, ${counts.failed} failed`);
  setHeader(`${state.stop ? "\u25A0 Stopped" : "\u2713 Done"}: ${counts.done} processed, ${counts.skipped} already done, ` +
    `${counts.failed} failed`, counts.failed ? "err" : "");
  await refreshPreds();
  await showLastRun(outs, warnings);
}

async function processJob({ dir, path, files, outName }, jobIdx, nJobs, mpp, session, opts, resume, counts, warnings) {
  const out = await dir.getDirectoryHandle(outName, { create: true });
  // resume like predict.py --resume: images with a row in _results.csv are skipped
  const prior = parseCsv((await readText(out, "_results.csv")) ?? "");
  const rows = new Map(prior.rows.filter((r) => r["file name"]).map((r) => [r["file name"], r]));
  let fields = prior.fields;
  const priorCfg = await readText(out, "run_config.yaml");
  let cfg = null;
  try { cfg = priorCfg === null ? null : JSON.parse(priorCfg); } catch { cfg = undefined; }
  if (cfg === undefined) log(`${outName}/run_config.yaml was written by predict.py; leaving it unchanged`);
  const inputFiles = new Map((cfg?.resolved?.input_files ?? []).map((f) => [f.file, f]));
  log(`===== [${jobIdx + 1}/${nJobs}] ${path || dir.name}`);

  for (const [i, h] of files.entries()) {
    const stem = h.name.replace(/\.[^.]+$/, "");
    progress(counts.n / counts.total, `Folder ${jobIdx + 1}/${nJobs}, image ${i + 1}/${files.length}: ${h.name}`);
    counts.n++;
    if (resume && rows.has(stem)) {
      log(`[${i + 1}/${files.length}] ${h.name}: already processed, reusing its CSV row`);
      counts.skipped++;
      continue;
    }
    if (state.stop) break;
    const t0 = performance.now();
    try {
      const file = await h.getFile();
      const { res, workMpp, preview } = await processFile(file, mpp, session, dir.name, opts);
      for (const [f, data] of Object.entries(res.files)) {
        const p = outputPath(stem, f);
        if (p) await writeFile(out, p, data);
      }
      if (preview) await writePreview(out, stem, file, preview);
      const meas = res.files[`${stem}_measurement.jpg`];
      if (meas) {
        setImg($("run-latest"), new Blob([meas], { type: "image/jpeg" }));
        $("run-preview").hidden = true;
      }
      if (res.row) rows.set(stem, { ...res.row, processed_um_per_px: +workMpp.toFixed(4) });
      fields = [...new Set([...res.columns, ...fields, "processed_um_per_px"])];
      const csv = [csvLine(fields), ...[...rows.values()].map((r) => csvLine(fields.map((c) => r[c])))].join("\n");
      await writeFile(out, "_results.csv", csv + "\n");
      if (cfg !== undefined) {
        inputFiles.set(h.name, { file: h.name, um_per_px: mpp, processed_um_per_px: +workMpp.toFixed(4),
          inference_scale: mpp / manifest.reference_um_per_px });
        const config = runConfig(dir.name, outName, mpp, [...inputFiles.values()], opts);
        await writeFile(out, "run_config.yaml", JSON.stringify(config, null, 1));
      }
      log(`${h.name}: ${res.row ? "done" : "SC or Epidermis empty, no CSV row"} (${ms(t0)})`);
      counts.done++;
    } catch (e) {
      console.error(e);
      log(String(e.stack ?? e));
      warnings.push(`${path ? `${path}/` : ""}${h.name}: ${describeError(e)}`);
      counts.failed++;
    }
  }
  return { out, path: path ? `${path}/${outName}` : outName };
}

async function showLastRun(outs, warnings) {
  state.lastRun = outs;
  $("last-run").hidden = false;
  $("last-warnings").replaceChildren(...warnings.map((w) => {
    const div = document.createElement("div");
    div.className = "warn";
    div.textContent = w;
    return div;
  }));
  const latest = $("run-latest").dataset.url;
  $("last-image").hidden = !latest;
  if (latest) $("last-image").src = latest;
  let fields = [];
  const rows = [];
  for (const o of outs) {
    const p = parseCsv((await readText(o.out, "_results.csv")) ?? "");
    fields = [...new Set([...fields, ...p.fields])];
    rows.push(...p.rows);
  }
  renderTable($("last-table"), fields, rows);
  $("last-table").hidden = !rows.length;
  $("review-last").disabled = !outs.length;
}

// folders that directly hold images; predicted_* (and legacy *_predicted) output folders are skipped, as in app.py
async function imageFolders(dir, path, recursive, found = []) {
  const files = [], subdirs = [];
  for await (const [name, h] of dir.entries()) {
    if (h.kind === "file" && IMAGE_RE.test(name)) files.push(h);
    else if (h.kind === "directory" && recursive && !/^predicted_|_predicted/i.test(name)) subdirs.push(h);
  }
  const byName = (a, b) => a.name.localeCompare(b.name);
  if (files.length) found.push({ dir, path, files: files.sort(byName) });
  for (const d of subdirs.sort(byName)) await imageFolders(d, path ? `${path}/${d.name}` : d.name, recursive, found);
  return found;
}

// JSON is valid YAML, so interface/seg_editor.run_info reads this like predict.py's run_config.yaml
function runConfig(source, output, mpp, inputFiles, opts) {
  return {
    run: { created_at: new Date().toISOString(), source: null, source_name: source, output, generator: "web",
      image_count: inputFiles.length, device: `browser (${currentBackend})`,
      model_class_names: manifest.classes.slice(1) },
    parameters: { um_per_px: mpp, reference_um_per_px: manifest.reference_um_per_px, tile: manifest.tile,
      tta: opts.tta, weights: manifest.source_weights, measure_above_mp_at_um_per_px: [MAX_POST_MP, POST_UM_PER_PX] },
    resolved: { input_files: inputFiles, roi_enabled: opts.detect.roi, follicle_enabled: opts.detect.follicle,
      quality_enabled: opts.detect.quality },
  };
}

// ---------------------------------------------------------------- run options
// interface/app.py SCANNERS
const SCANNERS = { "Motic 40X (0.2577 \u00b5m/px)": 0.2577, "BX61 10X (0.645 \u00b5m/px)": 0.645 };

const segValue = (id) => $(id).querySelector("button.on")?.dataset.v;
function segControl(id, onChange) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled || b.classList.contains("on")) return;
    for (const x of $(id).children) x.classList.toggle("on", x === b);
    onChange?.(b.dataset.v);
  });
}

function readOptions() {
  return { tta: segValue("tta"),
    detect: { roi: $("opt-roi").checked, follicle: $("opt-follicle").checked, quality: $("opt-quality").checked } };
}

function initScanner(mpp) {
  const sel = $("scanner");
  sel.replaceChildren(...[...Object.keys(SCANNERS), "Custom"].map((name) => {
    const o = document.createElement("option");
    o.value = o.textContent = name;
    return o;
  }));
  const apply = () => {
    const um = SCANNERS[sel.value];
    $("mpp-field").hidden = um !== undefined;
    if (um !== undefined) $("mpp").value = um;
  };
  sel.value = Object.keys(SCANNERS).find((n) => SCANNERS[n] === mpp) ?? "Custom";
  $("mpp").value = mpp;
  apply();
  sel.addEventListener("change", apply);
}

async function rotationK(backend, img) {
  const s = manifest.rotation.size;
  return rotationFromSquare(backend, resize(img, s, s).px);
}

// sq: RGB size x size, INTER_AREA from the full image
async function rotationFromSquare(backend, sq) {
  const r = manifest.rotation, s = r.size, plane = s * s;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    data[i] = sq[3 * i]; data[plane + i] = sq[3 * i + 1]; data[2 * plane + i] = sq[3 * i + 2];
  }
  const session = await getSession(backend, "rot");
  const out = await session.run({ [r.input.name]: new ort.Tensor("float32", data, [1, 3, s, s]) });
  const logits = Array.from(out[r.output.name].data);
  return (4 - logits.indexOf(Math.max(...logits))) % 4;
}

// A Pyodide worker (worker.js) with request ids; recycled when its never-shrinking heap gets large
class PyWorker {
  constructor(name, profile) {
    this.name = name;
    this.profile = profile;
    this.start();
  }

  start() {
    this.worker?.terminate();
    this.pending = new Map();
    this.next = 0;
    this.onLog = null;
    const w = this.worker = new Worker(`worker.js?v=${VERSION}`, { type: "module" });
    this.ready = new Promise((resolve, reject) => { this.readyCb = { resolve, reject }; });
    this.ready.catch(() => {});
    w.onmessage = ({ data }) => this.message(data);
    w.onerror = (e) => this.fail(new Error(e.message || `${this.name} crashed (likely out of memory)`));
    const { files, packages, modules } = this.profile;
    w.postMessage({ type: "init", version: VERSION, packages, modules,
      files: Object.fromEntries(Object.entries(files).map(([n, p]) => [n, sameOrigin(p).href])) });
  }

  message(data) {
    if (data.type === "ready") return this.readyCb.resolve(data.defaults);
    if (data.type === "log") return this.onLog?.(data.line);
    if (data.type === "infer") return runForWorker(this.worker, data);
    const p = this.pending.get(data.id);
    if (!p) {
      if (data.type === "error") this.readyCb.reject(new Error(`${this.name}: ${data.message}`));
      return;
    }
    this.pending.delete(data.id);
    if (data.type === "error") p.reject(new Error(`${data.message}\n${data.stack}`));
    else p.resolve(data);
    if (data.heapMB > WORKER_RECYCLE_MB && !this.pending.size) this.start();
  }

  fail(err) {
    const pending = [...this.pending.values()];
    this.readyCb.reject(err);
    this.start();
    for (const p of pending) p.reject(err);
  }

  async send(msg, transfer = []) {
    await this.ready;
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...msg, id }, transfer);
    });
  }

  run(msg, transfer, onLog) {
    this.onLog = onLog;
    return withMeasure(() => this.send({ type: "run", ...msg }, transfer));
  }

  call(fn, args, transfer = []) {
    return this.send({ type: "call", mod: this.profile.module, fn, args }, transfer);
  }
}

// runs and re-measures share the measurement worker one at a time
let measureChain = Promise.resolve();
function withMeasure(fn) {
  const p = measureChain.then(fn, fn);
  measureChain = p.catch(() => {});
  return p;
}

async function runForWorker(worker, { id, name, data, dims }) {
  try {
    if (!manifest[name]) throw new Error(`unknown model ${name}`);
    const session = await getSession(currentBackend, name);
    const out = await session.run({ [manifest[name].input]: new ort.Tensor("float32", new Float32Array(data), dims) });
    const t = out[session.outputNames[0]];
    const buf = t.data.slice().buffer;
    worker.postMessage({ type: "inferResult", id, data: buf, dims: t.dims }, [buf]);
  } catch (e) {
    worker.postMessage({ type: "inferResult", id, error: String(e.message ?? e) });
  }
}

function csvLine(values) {
  return values.map((v) => {
    let s = v == null ? "" : String(v);
    // keep spreadsheet apps from treating text cells as formulas
    if (/^[=+\-@\t\r]/.test(s) && Number.isNaN(Number(s))) s = `'${s}`;
    return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }).join(",");
}

function renderTable(el, cols, rows, opts = {}) {
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const c of cols) {
    const th = document.createElement("th");
    th.textContent = c;
    head.append(th);
  }
  const body = table.createTBody();
  rows.forEach((r, i) => {
    const tr = body.insertRow();
    if (opts.onPick) {
      tr.className = "pick" + (opts.selected === i ? " sel" : "");
      tr.addEventListener("click", () => opts.onPick(i));
    }
    for (const c of cols) {
      const td = tr.insertCell(), v = r[c] ?? "";
      td.textContent = v;
      if (v !== "" && !Number.isNaN(Number(v))) td.className = "num";
    }
  });
  el.replaceChildren(table);
}

function option(value, text = value) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = text;
  return o;
}

const byName = (a, b) => a.name.localeCompare(b.name);
const isoNow = () => {
  const d = new Date(), p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
};

function describeError(e) {
  if (e?.name === "NoModificationAllowedError" || e?.name === "InvalidStateError")
    return "an output file is open in another program (e.g. Excel or an image viewer); close it and save again";
  if (e?.name === "NotAllowedError") return "no permission to write to this folder; choose it again in the sidebar";
  const msg = String(e?.message ?? e);
  // a Python traceback: its last line names the exception
  const tb = msg.match(/Traceback \(most recent call last\):[\s\S]*?\n(\w[\w.]*(?:Error|Exception)\b[^\n]*)/);
  return tb ? `${msg.split("Traceback")[0]}${tb[1]}` : msg.split("\n")[0];
}

// ---------------------------------------------------------------- sidebar: working folder + predicted folder
function idb(mode, fn) {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open("skin-seg", 1);
    open.onupgradeneeded = () => open.result.createObjectStore("kv");
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const tx = open.result.transaction("kv", mode), req = fn(tx.objectStore("kv"));
      tx.oncomplete = () => resolve(req.result);
      tx.onerror = () => reject(tx.error);
    };
  });
}

async function chooseFolder() {
  try {
    await setRoot(await window.showDirectoryPicker({ id: "skin-seg-folder", mode: "readwrite" }));
  } catch (e) {
    if (e.name !== "AbortError") setHeader(`Error: ${e.message}`, "err");
  }
}

async function reopenFolder() {
  if (state.saved && (await state.saved.requestPermission({ mode: "readwrite" })) === "granted") await setRoot(state.saved);
}

// the working folder is remembered across visits (interface/app.py LAST_USED); reopening needs a click
async function setRoot(handle) {
  state.root = handle;
  state.pred = null;
  $("folder-name").value = handle.name;
  $("reopen-folder").hidden = true;
  idb("readwrite", (s) => s.put(handle, "root")).catch(() => {});
  await refreshPreds();
  await refreshRunImages();
  renderPage();
}

// root and every folder below it holding predict.py output (interface/app.py prediction_folders)
async function predictionFolders(root) {
  const found = [];
  async function walk(dir, path, parent) {
    let isPred = false;
    const subs = [];
    for await (const [name, h] of dir.entries()) {
      if (h.kind === "file") isPred ||= name === "run_config.yaml" || name === "_results.csv";
      else if (name.endsWith("_segmentations")) isPred = true;
      else subs.push(h);
    }
    if (isPred) found.push({ path: path || root.name, dir, parent });
    for (const s of subs.sort(byName)) await walk(s, path ? `${path}/${s.name}` : s.name, dir);
  }
  await walk(root, "", null);
  return found.sort((a, b) => a.path.localeCompare(b.path));
}

async function refreshPreds() {
  if (!state.root) return;
  const keep = state.pred?.path;
  state.preds = await predictionFolders(state.root);
  state.pred = state.preds.find((p) => p.path === keep) ?? state.preds[0] ?? null;
  const sel = $("pred-select");
  sel.replaceChildren(...(state.preds.length ? state.preds.map((p, i) => option(String(i), p.path))
    : [option("", "None yet")]));
  sel.value = state.pred ? String(state.preds.indexOf(state.pred)) : "";
}

async function refreshRunImages() {
  const names = [];
  for await (const [name, h] of state.root.entries()) if (h.kind === "file" && IMAGE_RE.test(name)) names.push(name);
  const sel = $("run-image"), keep = sel.value;
  sel.replaceChildren(...names.sort().map((n) => option(n)));
  if (names.includes(keep)) sel.value = keep;
}

// ---------------------------------------------------------------- navigation
function showPage(page) {
  state.page = ["run", "review", "results"].includes(page) ? page : "run";
  for (const a of document.querySelectorAll("header a")) a.classList.toggle("on", a.dataset.page === state.page);
  for (const s of document.querySelectorAll("main section[data-page]")) s.hidden = s.dataset.page !== state.page;
  renderPage();
}

function renderPage() {
  if (state.page === "run") renderRun();
  else if (state.page === "review") loadReview().catch(showError);
  else loadResults().catch(showError);
}

function showError(e) {
  console.error(e);
  setHeader(`Error: ${describeError(e)}`, "err");
}

function renderRun() {
  $("run-nofolder").hidden = Boolean(state.root);
  $("run-body").hidden = !state.root;
  $("image-field").hidden = state.scope !== "one";
  setBusy(state.busy);
}

function openInReview(path, stem) {
  const p = state.preds.find((x) => x.path === path);
  if (p) {
    state.pred = p;
    $("pred-select").value = String(state.preds.indexOf(p));
  }
  rv.target = stem;
  location.hash = "#review";
}

// ---------------------------------------------------------------- Review page (interface/app.py review_page)
const rv = { pred: null, stems: [], rows: new Map(), stem: null, view: "edit", doc: null, info: null, target: null,
  loads: 0 };

function ensureEditPy() {
  editPy ??= new PyWorker("editor engine", EDIT_PY);
  return editPy;
}

// ---------------------------------------------------------------- Review previews (<predicted>/review_cache)
async function writePreview(predDir, stem, file, { blob, width, height, k = null }) {
  const meta = { source: file.name, size: file.size, last_modified: file.lastModified, width, height, k };
  await writeFile(predDir, `${PREVIEW_DIR}/${stem}.jpg`, blob);
  await writeFile(predDir, `${PREVIEW_DIR}/${stem}.json`, JSON.stringify(meta));
  return meta;
}

// the saved preview, if it was made from this exact source file
async function readPreview(predDir, stem, file) {
  const dir = await subdir(predDir, PREVIEW_DIR);
  if (!dir) return null;
  let meta;
  try {
    meta = JSON.parse((await readText(dir, `${stem}.json`)) ?? "null");
  } catch {
    return null;
  }
  if (!meta || meta.size !== file.size || meta.last_modified !== file.lastModified) return null;
  try {
    return { blob: await (await dir.getFileHandle(`${stem}.jpg`)).getFile(), meta };
  } catch (e) {
    if (e.name === "NotFoundError") return null;
    throw e;
  }
}

const previewJobs = new Map();

// saved preview, else built from the source in image_worker.js and saved for next time
function ensurePreview(pred, stem, src) {
  const key = `${pred.path}|${stem}`;
  if (!previewJobs.has(key)) {
    previewJobs.set(key, (async () => {
      const file = await src.getFile();
      const hit = await readPreview(pred.dir, stem, file);
      if (hit) return { ...hit, built: false };
      const r = await imageJob({ op: "preview", file, maxSide: EDITOR_MAX_SIDE });
      let meta = { width: r.iw, height: r.ih, k: null };
      try {
        meta = await writePreview(pred.dir, stem, file, { blob: r.blob, width: r.iw, height: r.ih });
      } catch (e) {
        console.warn(`could not save the preview of ${stem}`, e);
      }
      return { blob: r.blob, meta, built: true };
    })().finally(() => previewJobs.delete(key)));
  }
  return previewJobs.get(key);
}

async function savePreviewK(d) {
  const dir = await subdir(d.pred.dir, PREVIEW_DIR);
  const meta = dir && JSON.parse((await readText(dir, `${d.stem}.json`)) ?? "null");
  if (meta && meta.k == null) await writeFile(dir, `${d.stem}.json`, JSON.stringify({ ...meta, k: d.k }));
}

let imageWorker = null, imageNext = 0;
const imagePending = new Map();

function imageJob(msg) {
  if (!imageWorker) {
    imageWorker = new Worker(`image_worker.js?v=${VERSION}`, { type: "module" });
    imageWorker.onmessage = ({ data }) => {
      const p = imagePending.get(data.id);
      imagePending.delete(data.id);
      if (data.error) p?.reject(new Error(data.error));
      else p?.resolve(data);
    };
    imageWorker.onerror = (e) => {
      const err = new Error(e.message || "image decoder crashed (likely out of memory)");
      for (const p of imagePending.values()) p.reject(err);
      imagePending.clear();
      imageWorker.terminate();
      imageWorker = null;
    };
  }
  const id = imageNext++;
  return new Promise((resolve, reject) => {
    imagePending.set(id, { resolve, reject });
    imageWorker.postMessage({ ...msg, id });
  });
}

// the next image's preview is built while this one is being edited
function prefetchNext() {
  const pred = rv.pred, stem = rv.stems[rv.stems.indexOf(rv.stem) + 1];
  if (!stem || state.busy || saveJobs.some((j) => !j.end)) return;
  (async () => {
    const src = await findSource(pred, stem, rv.info);
    if (src) await ensurePreview(pred, stem, src);
  })().catch((e) => console.warn(`preparing ${stem}:`, e));
}

async function loadReview() {
  const empty = (msg) => {
    $("review-empty").textContent = msg;
    $("review-empty").hidden = false;
    $("review-body").hidden = true;
  };
  if (!state.root) return empty("Choose a working folder in the sidebar.");
  const pred = state.pred;
  if (!pred) return empty("No predicted folders in this working folder yet. Process images on the Run page first.");
  ensureEditPy();
  if (rv.pred !== pred) Object.assign(rv, { pred, info: null });
  const csv = parseCsv((await readText(pred.dir, "_results.csv")) ?? "");
  rv.rows = new Map(csv.rows.filter((r) => r["file name"]).map((r) => [r["file name"], r]));
  const stems = new Set(rv.rows.keys());
  for (const kind of ["postprocessed", "raw"]) {
    const d = await subdir(pred.dir, `${kind}_segmentations`);
    if (d) for await (const [n, h] of d.entries()) if (h.kind === "file" && n.endsWith(".txt")) stems.add(n.slice(0, -4));
  }
  rv.stems = [...stems].sort();
  if (!rv.stems.length) return empty("No images in this predicted folder.");
  $("review-empty").hidden = true;
  $("review-body").hidden = false;
  if (rv.target && rv.stems.includes(rv.target)) rv.stem = rv.target;
  rv.target = null;
  if (!rv.stems.includes(rv.stem)) rv.stem = rv.stems[0];
  $("rv-image").replaceChildren(...rv.stems.map((s) => option(s, s + (rv.rows.get(s)?.edited ? "  (edited)" : ""))));
  $("rv-image").value = rv.stem;
  await showReviewImage();
}

async function showReviewImage() {
  const i = rv.stems.indexOf(rv.stem), n = rv.stems.length;
  $("rv-pos").textContent = `${i + 1}/${n}`;
  $("rv-prev").disabled = i <= 0;
  $("rv-next").disabled = i >= n - 1;
  const row = rv.rows.get(rv.stem);
  if (row) renderTable($("rv-values"), ["field", "value"], Object.entries(row).map(([field, value]) => ({ field, value })));
  else $("rv-values").textContent = "No CSV row for this image (it may have been skipped during postprocessing).";
  $("rv-error").hidden = true;
  const edit = rv.view === "edit";
  $("rv-editor").hidden = !edit;
  $("rv-image-view").hidden = edit;
  if (edit) return openDoc();
  $("rv-steps-note").hidden = rv.view !== "steps";
  const bytes = await readBytes(rv.pred.dir, `${rv.stem}_${rv.view}.jpg`);
  setImg($("rv-img"), bytes && new Blob([bytes], { type: "image/jpeg" }));
  $("rv-img-missing").hidden = Boolean(bytes);
  $("rv-img-missing").textContent = `No ${rv.view} image for ${rv.stem}`;
}

function stepReview(d) {
  const i = rv.stems.indexOf(rv.stem) + d;
  rv.stem = rv.stems[Math.max(0, Math.min(i, rv.stems.length - 1))];
  $("rv-image").value = rv.stem;
  showReviewImage().catch(showError);
}

async function labelFile(dir, stem) {
  for (const kind of ["postprocessed", "raw"]) {
    const d = await subdir(dir, `${kind}_segmentations`);
    const text = d && await readText(d, `${stem}.txt`);
    if (text !== null && text !== undefined) return { kind, text };
  }
  return null;
}

// interface/seg_editor.find_image, in the folder holding the predicted folder (.svs can't be decoded here)
async function findSource(pred, stem, info) {
  const d = pred.parent;
  if (!d) return null;
  const names = [info?.files?.[stem]?.file, ...[".tif", ".tiff", ".png", ".jpg", ".jpeg"].map((e) => stem + e)];
  const tiffDir = await subdir(d, `TIFF_${d.name}`);
  const cands = [...(tiffDir ? [[tiffDir, `${stem}.tif`]] : []), ...names.filter(Boolean).map((n) => [d, n])];
  for (const [dir, name] of cands) {
    if (/\.svs$/i.test(name)) continue;
    try {
      return await dir.getFileHandle(name);
    } catch (e) {
      if (e.name !== "NotFoundError" && e.name !== "TypeMismatchError") throw e;
    }
  }
  return null;
}

async function openDoc() {
  const pred = rv.pred, stem = rv.stem;
  if (rv.doc?.pred === pred && rv.doc?.stem === stem) return updateEditor();
  const token = rv.token = {};
  const tAll = performance.now(), laps = [];
  let t0 = tAll;
  const lap = (name) => { laps.push(`${name} ${ms(t0)}`); t0 = performance.now(); };
  try {
    setHeader(`Loading ${stem}...`);
    const py = ensureEditPy();
    const infoP = rv.info ? Promise.resolve(rv.info) : readText(pred.dir, "run_config.yaml")
      .then((text) => py.call("run_info", [text ?? ""]))
      .then(({ result }) => { if (rv.pred === pred) rv.info = result; return result; });
    infoP.catch(() => {});
    const lbl = await labelFile(pred.dir, stem);
    if (!lbl) throw new Error(`No raw or postprocessed segmentation found for ${stem} in ${pred.path}`);
    // run_config (read by the editor engine) is only needed when the source isn't <stem>.<ext>
    const src = (await findSource(pred, stem, rv.info)) ?? (await findSource(pred, stem, await infoP));
    if (!src) throw new Error(`Source image of ${stem} not found next to ${pred.path} (.svs files need their TIFF copy)`);
    lap("label + source");
    const pv = await ensurePreview(pred, stem, src);
    lap(pv.built ? "preview built" : "preview from review_cache");
    const info = await infoP;
    lap("editor engine");
    const iw = pv.meta.width, ih = pv.meta.height;
    const { polys, other } = (await py.call("read_polys", [lbl.text, JSON.stringify(info.names), iw, ih])).result;
    lap("polygons");
    if (token !== rv.token) return;
    if (rv.doc?.image) URL.revokeObjectURL(rv.doc.image);
    rv.doc = { id: `${pred.path}|${stem}|${++rv.loads}`, pred, stem, src, info, iw, ih, k: pv.meta.k ?? null, polys,
      other, image: URL.createObjectURL(pv.blob), version: 0, dirty: false, noop: false,
      message: `loaded ${lbl.kind} segmentation` };
    updateEditor();
    log(`Review ${stem}: ${laps.join(", ")}; total ${ms(tAll)}`);
    prefetchNext();
  } catch (e) {
    console.error(e);
    $("rv-error").textContent = describeError(e);
    $("rv-error").hidden = false;
  }
  if (!state.busy) setHeader(idleHeader());
}

function updateEditor() {
  const d = rv.doc;
  if (!d || rv.view !== "edit") return;
  const i = rv.stems.indexOf(d.stem);
  mountEditor({
    data: { iw: d.iw, ih: d.ih, polys: d.polys, version: d.version, dirty: d.dirty, noop: d.noop, message: d.message,
      doc: d.id, height: EDITOR_HEIGHT, image: d.image,
      nav: { label: `${i + 1}/${rv.stems.length}`, prev: i > 0, next: i < rv.stems.length - 1 },
      saving: saveJobs.some((j) => !j.end && j.pred === d.pred && j.stem === d.stem),
      classes: Object.keys(d.polys).map((name) => ({ name, color: CLASS_COLORS[name] ?? "#ffffff" })) },
    parentElement: $("rv-editor"),
    setTriggerValue: (name, value) => onEditorTrigger(name, value).catch(showError),
  });
}

// interface/seg_editor.py's on_op / on_save / on_nav / on_need_image callbacks
async function onEditorTrigger(name, value) {
  const d = rv.doc;
  if (!d) return;
  if (name === "nav") return stepReview(value);
  if (name === "need_image") return updateEditor();
  if (name === "save") return saveDoc(d, value.polys);
  if (name !== "op") return;
  const polys = value.polys;
  try {
    const r = (await editPy.call("op", [value.kind, value.cls, JSON.stringify(value.lasso),
      JSON.stringify(polys[value.cls] ?? []), d.iw, d.ih])).result;
    if (r.done) polys[value.cls] = r.polys;
    Object.assign(d, { polys, version: d.version + 1, dirty: Boolean(value.dirty) || r.done, noop: !r.done,
      message: r.message });
  } catch (e) {
    Object.assign(d, { version: d.version + 1, noop: true, message: `failed: ${describeError(e)}` });
  }
  if (rv.doc === d) updateEditor();
}

// seg_editor.write_labels: overwrite postprocessed_segmentations/<stem>.txt, original kept once in original/
async function saveDoc(d, polys) {
  try {
    const { text } = (await editPy.call("label_text", [JSON.stringify(polys), JSON.stringify(d.other),
      JSON.stringify(d.info.names)])).result;
    const segDir = await d.pred.dir.getDirectoryHandle("postprocessed_segmentations", { create: true });
    const current = await readText(segDir, `${d.stem}.txt`);
    if (current !== null) {
      const orig = await segDir.getDirectoryHandle("original", { create: true });
      if ((await readText(orig, `${d.stem}.txt`)) === null) await writeFile(orig, `${d.stem}.txt`, current);
    }
    await writeFile(segDir, `${d.stem}.txt`, text);
    Object.assign(d, { polys, dirty: false, noop: false, version: d.version + 1,
      message: "saved; re-measuring in the background" });
    queueRemeasure(d, text);
  } catch (e) {
    console.error(e);
    Object.assign(d, { polys, dirty: true, noop: false, version: d.version + 1, message: `save failed: ${describeError(e)}` });
  }
  if (rv.doc === d) updateEditor();
}

// ---------------------------------------------------------------- background re-measure (seg_editor.submit_save)
const saveJobs = [];
let saveTimer = 0;

function queueRemeasure(d, label) {
  const job = { stem: d.stem, pred: d.pred, queued: Date.now(), started: null, end: null, message: "", error: false };
  saveJobs.push(job);
  if (saveJobs.length > 20) saveJobs.shift();
  withMeasure(() => remeasureJob(d, label, job));
  saveTimer ||= setInterval(() => { if (!state.busy) setHeader(idleHeader(), saveJobs.at(-1)?.error ? "err" : ""); }, 1000);
}

async function remeasureJob(d, label, job) {
  job.started = Date.now();
  try {
    const file = await d.src.getFile();
    const um = d.info.files[d.stem]?.um_per_px || d.info.um_default || 1.0;
    // decoded off the page; the rotation comes from review_cache when the run saved it
    const needK = d.k == null;
    const w = await imageJob({ op: "work", file, um, maxMP: MAX_POST_MP, postUm: POST_UM_PER_PX,
      rotSize: needK ? manifest.rotation.size : 0 });
    if (needK) {
      if (!currentBackend) await prepareBackend();
      d.k = await rotationFromSquare(currentBackend, w.sq);
      await savePreviewK(d).catch((e) => console.warn("could not save the rotation", e));
    }
    const workUm = w.workUm, steps = await readBytes(d.pred.dir, `${d.stem}_steps.jpg`);
    const r = await measureWorker.call("remeasure", [file.name, w.rgb, w.w, w.h, workUm, d.k, label,
      JSON.stringify(d.info.names), steps, d.info.lateral_clip_frac ?? null, file.lastModified],
      [w.rgb.buffer, ...(steps ? [steps.buffer] : [])]);
    if (!r.result.row) {
      job.message = "labels saved; SC or Epidermis is empty, so measurements were not updated";
    } else {
      await writeFile(d.pred.dir, `${d.stem}_measurement.jpg`, r.files["measurement.jpg"]);
      await writeFile(d.pred.dir, `${d.stem}_steps.jpg`, r.files["steps.jpg"]);
      await updateRow(d.pred.dir, d.stem,
        { ...r.result.row, edited: isoNow(), processed_um_per_px: +workUm.toFixed(4) },
        { "folder name": d.pred.parent?.name ?? "", total_tissue_um2: r.result.tissue }, r.result.columns);
      job.message = "measurements, steps image and CSV row updated";
    }
  } catch (e) {
    console.error(e);
    Object.assign(job, { message: `re-measure failed: ${describeError(e)}`, error: true });
  }
  job.end = Date.now();
  if (!saveJobs.some((j) => !j.end)) {
    clearInterval(saveTimer);
    saveTimer = 0;
    if (!state.busy) setHeader(idleHeader(), job.error ? "err" : "");
    // refresh the edited mark, values and images of the image being reviewed
    if (state.page === "review" && rv.pred === d.pred) await loadReview().catch(showError);
  }
}

// seg_editor.update_row
async function updateRow(dir, stem, updates, newRow, columns) {
  const p = parseCsv((await readText(dir, "_results.csv")) ?? "");
  const fields = [...new Set([...p.fields, ...columns, ...Object.keys(updates)])];
  let row = p.rows.find((r) => r["file name"] === stem);
  if (!row) p.rows.push(row = { "file name": stem, ...newRow });
  Object.assign(row, updates);
  await writeFile(dir, "_results.csv", [csvLine(fields), ...p.rows.map((r) => csvLine(fields.map((c) => r[c])))].join("\n") + "\n");
}

// interface/app.py header_status / _save_status
function idleHeader() {
  const active = saveJobs.filter((j) => !j.end);
  if (active.length) {
    const j = active[0];
    const when = j.started ? `${Math.round((Date.now() - j.started) / 1000)} s` : "queued";
    const more = active.length > 1 ? ` \u00B7 ${active.length - 1} more queued` : "";
    return `\u21BB Saved ${j.stem}, re-measuring in the background (${when})${more}`;
  }
  const j = saveJobs.at(-1);
  if (!j) return state.modelsReady ? "\u2713 Ready" : "Loading models...";
  const at = new Date(j.end).toLocaleTimeString();
  return j.error ? `\u2715 ${j.stem}: ${j.message} (${at})`
    : `\u2713 Saved ${j.stem}: ${j.message} (${at}, ${Math.round((j.end - (j.started ?? j.queued)) / 1000)} s)`;
}

// ---------------------------------------------------------------- Results page (interface/app.py results_page)
const KEY_METRICS = ["SC mean height", "Epidermal mean height", "EDJ tortuosity", "quality"];
const RESULT_COLUMNS = ["folder name", "file name", ...KEY_METRICS, "edited"];
const res = { rows: [], fields: [], folders: new Set(), extra: new Set(), pick: null, metric: null };

async function findResults(dir, path, out) {
  const names = new Set(), subs = [];
  for await (const [name, h] of dir.entries()) {
    if (h.kind === "file") names.add(name);
    else if (!name.endsWith("_segmentations")) subs.push(h);
  }
  const where = path || dir.name;
  // legacy results.csv only when no _results.csv sits next to it
  const csv = names.has("_results.csv") ? "_results.csv" : names.has("results.csv") ? "results.csv" : null;
  if (csv) out.push({ dirPath: where, path: `${where}/${csv}`, handle: await dir.getFileHandle(csv) });
  for (const s of subs.sort(byName)) await findResults(s, path ? `${path}/${s.name}` : s.name, out);
  return out;
}

async function loadResults() {
  const empty = (msg) => {
    $("results-empty").textContent = msg;
    $("results-empty").hidden = false;
    $("results-body").hidden = true;
  };
  if (!state.root) return empty("Choose a working folder in the sidebar.");
  setHeader("Searching for _results.csv...");
  const files = await findResults(state.root, "", []);
  res.rows = [];
  res.fields = [];
  for (const f of files) {
    try {
      const p = parseCsv(await (await f.handle.getFile()).text());
      res.fields = [...new Set([...res.fields, ...p.fields])];
      for (const r of p.rows) res.rows.push({ ...r, "results path": f.path, _dir: f.dirPath });
    } catch (e) {
      console.warn(`could not read ${f.path}`, e);
    }
  }
  res.fields.push("results path");
  res.files = files.length;
  res.pick = null;
  if (!state.busy) setHeader(idleHeader());
  if (!res.rows.length) return empty("No _results.csv files in this folder yet. Process images on the Run page first.");
  $("results-empty").hidden = true;
  $("results-body").hidden = false;
  const folders = [...new Set(res.rows.map((r) => r["folder name"] ?? ""))].sort();
  for (const f of res.folders) if (!folders.includes(f)) res.folders.delete(f);
  multiSelect("res-folders", folders, res.folders, "All folders", renderResults);
  multiSelect("res-columns", res.fields.filter((c) => !RESULT_COLUMNS.includes(c) && c !== "results path"),
    res.extra, "Add columns", renderResults);
  renderResults();
}

function multiSelect(id, items, selected, placeholder, onChange) {
  const el = $(id), box = el.querySelector("div"), summary = el.querySelector("summary");
  const label = () => { summary.textContent = selected.size ? [...selected].join(", ") : placeholder; };
  box.replaceChildren(...items.map((item) => {
    const l = document.createElement("label"), cb = document.createElement("input");
    l.className = "check";
    cb.type = "checkbox";
    cb.checked = selected.has(item);
    cb.addEventListener("change", () => {
      if (cb.checked) selected.add(item); else selected.delete(item);
      label();
      onChange();
    });
    l.append(cb, item || "(blank)");
    return l;
  }));
  label();
}

const resView = () => (res.folders.size ? res.rows.filter((r) => res.folders.has(r["folder name"] ?? "")) : res.rows);
const isNum = (v) => v !== "" && v != null && !Number.isNaN(Number(v));

function renderResults() {
  const view = resView();
  const cols = [...RESULT_COLUMNS.filter((c) => res.fields.includes(c)), ...res.extra];
  $("res-caption").textContent = `${view.length} images from ${new Set(view.map((r) => r["results path"])).size} ` +
    "_results.csv file(s). Select a row to open it in Review.";
  if (res.pick !== null && !view.includes(res.pick)) res.pick = null;
  renderTable($("res-table"), cols, view, { selected: view.indexOf(res.pick), onPick: (i) => {
    res.pick = view[i];
    renderResults();
  } });
  $("res-open").hidden = !res.pick;
  if (res.pick) $("res-open").textContent = `\u2611 Open ${res.pick["file name"]} in Review`;

  const numeric = res.fields.filter((c) => c !== "results path" && view.some((r) => isNum(r[c])) &&
    view.every((r) => r[c] === "" || r[c] == null || isNum(r[c])));
  const folders = [...new Set(view.map((r) => r["folder name"] ?? ""))].sort();
  $("res-by-folder").hidden = !numeric.length || !res.fields.includes("folder name");
  if ($("res-by-folder").hidden) return;
  if (!numeric.includes(res.metric)) res.metric = KEY_METRICS.find((k) => numeric.includes(k)) ?? numeric[0];
  $("res-metric").replaceChildren(...numeric.map((c) => option(c)));
  $("res-metric").value = res.metric;
  const stats = folders.map((folder) => {
    const out = { "folder name": folder };
    for (const c of numeric) {
      const v = view.filter((r) => (r["folder name"] ?? "") === folder && isNum(r[c])).map((r) => Number(r[c]));
      const mean = v.reduce((a, b) => a + b, 0) / (v.length || 1);
      const std = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1)) : NaN;
      Object.assign(out, { [`${c} mean`]: v.length ? +mean.toFixed(4) : "", [`${c} std`]: Number.isNaN(std) ? "" : +std.toFixed(4),
        [`${c} count`]: v.length });
    }
    return out;
  });
  const max = Math.max(...stats.map((s) => Number(s[`${res.metric} mean`]) || 0), 1e-9);
  $("res-bars").replaceChildren(...stats.flatMap((s) => {
    const name = document.createElement("span"), track = document.createElement("div"), bar = document.createElement("div");
    const val = document.createElement("span");
    name.textContent = s["folder name"] || "(blank)";
    bar.className = "bar";
    bar.style.width = `${Math.max(0, (Number(s[`${res.metric} mean`]) || 0) / max) * 100}%`;
    track.append(bar);
    val.textContent = s[`${res.metric} mean`];
    val.className = "caption";
    return [name, track, val];
  }));
  renderTable($("res-stats"), ["folder name", ...numeric.flatMap((c) => [`${c} mean`, `${c} std`, `${c} count`])], stats);
}

function downloadResults() {
  const view = resView(), cols = res.fields;
  const blob = new Blob([[csvLine(cols), ...view.map((r) => csvLine(cols.map((c) => r[c])))].join("\n") + "\n"],
    { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "combined_results.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------------------------------------------------------------- startup
function wire() {
  window.addEventListener("hashchange", () => showPage(location.hash.slice(1)));
  $("choose-folder").addEventListener("click", chooseFolder);
  $("reopen-folder").addEventListener("click", () => reopenFolder().catch(showError));
  $("pred-select").addEventListener("change", (e) => {
    state.pred = state.preds[Number(e.target.value)] ?? null;
    if (state.page === "review") loadReview().catch(showError);
  });
  $("pred-refresh").addEventListener("click", () => refreshPreds().then(renderPage).catch(showError));
  segControl("scope", (v) => { state.scope = v; renderRun(); });
  segControl("tta");
  for (const k of ["roi", "follicle", "quality"]) {
    $(`opt-${k}`).addEventListener("change", () => { toggleDefaultsApplied = true; });
  }
  $("run").addEventListener("click", () => runJobs().catch((e) => {
    state.busy = false;
    setBusy(false);
    showError(e);
  }));
  $("stop").addEventListener("click", () => {
    state.stop = true;
    $("stop").disabled = true;
    setHeader("Stopping after the current image...");
  });
  $("review-last").addEventListener("click", () => state.lastRun?.[0] && openInReview(state.lastRun[0].path, null));
  $("rv-image").addEventListener("change", (e) => {
    rv.stem = e.target.value;
    showReviewImage().catch(showError);
  });
  segControl("rv-view", (v) => {
    rv.view = v;
    showReviewImage().catch(showError);
  });
  $("rv-prev").addEventListener("click", () => stepReview(-1));
  $("rv-next").addEventListener("click", () => stepReview(1));
  $("results-refresh").addEventListener("click", () => loadResults().catch(showError));
  $("res-metric").addEventListener("change", (e) => { res.metric = e.target.value; renderResults(); });
  $("res-open").addEventListener("click", () => res.pick && openInReview(res.pick._dir, res.pick["file name"]));
  $("res-download").addEventListener("click", downloadResults);
}

async function init() {
  wire();
  showPage(location.hash.slice(1));
  if (!window.showDirectoryPicker) {
    $("choose-folder").disabled = true;
    $("folder-support").hidden = false;
  }
  try {
    if (!navigator.gpu) {
      $("backend").value = "wasm";
      $("backend").querySelector('option[value="webgpu"]').disabled = true;
    }
    const jsonUrl = new URL(modelBase.href + ".json");
    const r = await fetch(jsonUrl);
    if (!r.ok) throw new Error(`model settings: HTTP ${r.status}`);
    manifest = await r.json();
    measureWorker = new PyWorker("measurement engine", MEASURE_PY);
    // base_config's ROI/follicle/quality defaults, unless the user already changed a toggle
    measureWorker.ready.then((defaults) => {
      if (toggleDefaultsApplied || !defaults) return;
      toggleDefaultsApplied = true;
      for (const k of ["roi", "follicle", "quality"]) $(`opt-${k}`).checked = defaults[k];
    }).catch(() => {});
    initScanner(manifest.default_um_per_px);
    const saved = window.showDirectoryPicker ? await idb("readonly", (s) => s.get("root")).catch(() => null) : null;
    if (saved) {
      if ((await saved.queryPermission({ mode: "readwrite" })) === "granted") await setRoot(saved);
      else {
        state.saved = saved;
        $("reopen-folder").textContent = `Reopen ${saved.name}`;
        $("reopen-folder").hidden = false;
      }
    }
    modelUrls.seg = sameOrigin(new URL(manifest.model, jsonUrl).href);
    models.seg = await fetchBytes(modelUrls.seg, "Downloading segmentation model");
    for (const key of ["rotation", "roi", "follicle", "quality"]) {
      const k = key === "rotation" ? "rot" : key;
      modelUrls[k] = sameOrigin(new URL(manifest[key].model, jsonUrl).href);
      models[k] = await fetchBytes(modelUrls[k], `Downloading ${key} model`);
    }
    state.modelsReady = true;
    ensureEditPy();
    if (!state.busy) setHeader(idleHeader());
    setBusy(state.busy);
  } catch (e) {
    showError(e);
  }
}

init();
