// Runs postprocess.process_image (via web/pipeline.py) and the Review helpers in Pyodide, off the page's main thread
// (module worker). Each worker loads only the packages and modules it is given at init.
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/";
const PYODIDE_SRI = "sha384-VR+eYp8LZKGhhQ7Nql3TONJIMizM0VRxz87k6V7IyuQHElicjy7ULmkOuB2LFR3b";
let py, pipeline;
const mods = {};
let nextInferId = 0;
const pendingInfer = new Map();

// detectors run on the page's ONNX Runtime (GPU); Python awaits the result
function infer(name, bytes, dims) {
  const id = nextInferId++;
  const buf = bytes.slice().buffer;
  return new Promise((resolve, reject) => {
    pendingInfer.set(id, { resolve, reject });
    postMessage({ type: "infer", id, name, data: buf, dims: Array.from(dims) }, [buf]);
  });
}

async function sha384(buf) {
  const bytes = new Uint8Array(await crypto.subtle.digest("SHA-384", buf));
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return "sha384-" + btoa(bin);
}

// dynamic import() has no integrity option, so check the pinned hash ourselves
async function importVerified(url, sri) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  if ((await sha384(buf)) !== sri) throw new Error(`integrity check failed for ${url}`);
  const blobUrl = URL.createObjectURL(new Blob([buf], { type: "text/javascript" }));
  try {
    return await import(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

async function init({ files, packages, modules }) {
  const { loadPyodide } = await importVerified(PYODIDE_URL + "pyodide.mjs", PYODIDE_SRI);
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  py.setStdout({ batched: (line) => postMessage({ type: "log", line }) });
  py.setStderr({ batched: (line) => postMessage({ type: "log", line }) });
  await py.loadPackage(packages);
  py.FS.mkdirTree("/app");
  for (const [name, url] of Object.entries(files)) {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
    py.FS.mkdirTree(`/app/${name}`.replace(/\/[^/]+$/, ""));
    py.FS.writeFile(`/app/${name}`, new Uint8Array(await res.arrayBuffer()));
  }
  py.runPython("import sys; sys.path.insert(0, '/app')");
  for (const m of modules) mods[m] = py.pyimport(m);
  pipeline = mods.pipeline;
  return pipeline ? JSON.parse(pipeline.defaults()) : null;
}

const heapMB = () => Math.round(py._module.HEAPU8.buffer.byteLength / 2 ** 20);

async function run(msg) {
  const { id, name, rgb, mask, w, h, umPerPx, k, lastModified, quality, folderName, options } = msg;
  const result = JSON.parse(await pipeline.run(name, rgb, mask, w, h, umPerPx, k, lastModified, infer, quality,
    folderName ?? "", JSON.stringify(options ?? {})));
  const files = {};
  for (const f of result.files) files[f] = py.FS.readFile(`/out/${f}`);
  postMessage({ type: "done", id, ...result, files, heapMB: heapMB() }, Object.values(files).map((a) => a.buffer));
}

// <mod>.<fn>(...args); a JSON result listing "outputs" brings those /out_review files back with it
async function call({ id, mod, fn, args }) {
  let out = mods[mod][fn](...args);
  if (out && typeof out.then === "function") out = await out;
  const result = typeof out === "string" ? JSON.parse(out) : out;
  const files = {};
  for (const f of result?.outputs ?? []) files[f] = py.FS.readFile(`/out_review/${f}`);
  postMessage({ type: "done", id, result, files, heapMB: heapMB() }, Object.values(files).map((a) => a.buffer));
}

let ready;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      ready = init(data);
      postMessage({ type: "ready", defaults: await ready });
    } else if (data.type === "run") {
      await ready;
      await run(data);
    } else if (data.type === "call") {
      await ready;
      await call(data);
    } else if (data.type === "inferResult") {
      const p = pendingInfer.get(data.id);
      pendingInfer.delete(data.id);
      if (data.error) p.reject(new Error(data.error));
      else p.resolve({ data: new Float32Array(data.data), dims: data.dims });
    }
  } catch (e) {
    postMessage({ type: "error", id: data.id, message: String(e.message ?? e), stack: String(e.stack ?? "") });
  }
};
