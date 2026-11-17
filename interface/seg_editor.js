// In-browser polygon editor for the Edit and review tab (CCv2 component, see seg_editor.py).
// Mirrors training/semseg/add_segmentations.py; merge / cut run in Python (op trigger).

const CLOSE_PX = 10;
const TRACE_PX = 5;
const VERTEX_PX = 8;
const EDGE_PX = 6;
const MAX_HANDLES = 4000;
const NUDGE_R_MIN = 5;
const NUDGE_R_MAX = 400;
const UNDO_MAX = 50;

const INSTANCES = new WeakMap();

const clone = (polys) => {
  const out = {};
  for (const [c, lst] of Object.entries(polys)) out[c] = lst.map((p) => p.map((v) => [v[0], v[1]]));
  return out;
};

function segDist(q, a, b) {
  const abx = b[0] - a[0], aby = b[1] - a[1];
  const t = Math.min(1, Math.max(0, ((q[0] - a[0]) * abx + (q[1] - a[1]) * aby) / Math.max(abx * abx + aby * aby, 1e-9)));
  return Math.hypot(a[0] + t * abx - q[0], a[1] + t * aby - q[1]);
}

function pointInPoly(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function area(pts) {
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) a += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1]);
  return Math.abs(a / 2);
}

function createEditor(root) {
  const wrap = root.querySelector(".seg-ed");
  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const bar = root.querySelector(".bar");
  const statusEl = root.querySelector(".status");
  const hintEl = root.querySelector(".hint");

  const S = {
    doc: null, version: -1, iw: 1, ih: 1, img: null, classes: [], polys: {}, active: null,
    mode: "draw", showOthers: true, s: 1, fitS: 1, ox: 0, oy: 0,
    points: [], strokes: [], tracing: false, drag: null, dragMoved: false, nudgeLast: null, nudged: false,
    pan: null, mouse: [0, 0], undo: [], dirty: false, pending: false, message: "", brushR: 40,
    trigger: null, raf: 0,
  };

  // ------------------------------------------------------------ view
  const toImg = (x, y) => [S.ox + x / S.s, S.oy + y / S.s];
  const toDisp = (ix, iy) => [(ix - S.ox) * S.s, (iy - S.oy) * S.s];
  const clampImg = ([ix, iy]) => [Math.min(Math.max(ix, 0), S.iw), Math.min(Math.max(iy, 0), S.ih)];
  const normToDisp = (p) => toDisp(p[0] * S.iw, p[1] * S.ih);

  function fit() {
    const W = canvas.width, H = canvas.height;
    S.s = S.fitS = Math.min(W / S.iw, H / S.ih);
    S.ox = -((W - S.iw * S.s) / 2) / S.s;
    S.oy = -((H - S.ih * S.s) / 2) / S.s;
    redraw();
  }

  function zoom(f, x, y) {
    const [ix, iy] = toImg(x, y);
    S.s = Math.min(Math.max(S.s * f, S.fitS / 2), 8);
    S.ox = ix - x / S.s;
    S.oy = iy - y / S.s;
    redraw();
  }

  function resize() {
    const w = Math.max(200, Math.floor(wrap.clientWidth));
    if (w === canvas.width) return;
    const keepFit = Math.abs(S.s - S.fitS) < 1e-9;
    canvas.width = w;
    if (keepFit) fit(); else redraw();
  }

  // ------------------------------------------------------------ polygons
  const color = (c) => (S.classes.find((k) => k.name === c) || {}).color || "#ffffff";
  const cur = () => S.polys[S.active] || (S.polys[S.active] = []);
  const editable = () => {
    const out = [];
    for (const c of S.showOthers ? Object.keys(S.polys) : [S.active])
      (S.polys[c] || []).forEach((p, k) => out.push([c, k, p]));
    return out;
  };

  function pushUndo() {
    S.undo.push(clone(S.polys));
    if (S.undo.length > UNDO_MAX) S.undo.shift();
  }

  function changed() {
    S.dirty = true;
    S.message = "";
  }

  function vertexAt(x, y) {
    let best = null, bd = VERTEX_PX;
    for (const [c, k, p] of editable()) {
      p.forEach((v, vi) => {
        const [dx, dy] = normToDisp(v);
        const d = Math.hypot(dx - x, dy - y);
        if (d <= bd) { best = [c, k, vi]; bd = d; }
      });
    }
    return best;
  }

  function edgeAt(x, y) {
    let best = null, bd = EDGE_PX;
    for (const [c, k, p] of editable()) {
      const a = p.map(normToDisp);
      for (let i = 0; i < a.length; i++) {
        const d = segDist([x, y], a[i], a[(i + 1) % a.length]);
        if (d <= bd) { best = [c, k, i + 1]; bd = d; }
      }
    }
    return best;
  }

  function deleteVertex(hit) {
    if (!hit) return;
    const [c, k, vi] = hit;
    const p = S.polys[c][k];
    if (p.length <= 3) {
      S.message = "a polygon needs at least 3 vertices; use x to delete the whole polygon";
      return;
    }
    pushUndo();
    p.splice(vi, 1);
    changed();
  }

  function deleteUnderCursor() {
    const [ix, iy] = toImg(...S.mouse);
    let best = -1, ba = Infinity;
    cur().forEach((p, k) => {
      const px = p.map((v) => [v[0] * S.iw, v[1] * S.ih]);
      if (pointInPoly(ix, iy, px)) {
        const a = area(px);
        if (a < ba) { best = k; ba = a; }
      }
    });
    if (best < 0) return;
    pushUndo();
    cur().splice(best, 1);
    changed();
  }

  function nudge(ix, iy) {
    const [lx, ly] = S.nudgeLast;
    S.nudgeLast = [ix, iy];
    const dx = ix - lx, dy = iy - ly;
    if (!dx && !dy) return;
    const r = S.brushR / S.s, c = [lx, ly], step = Math.max(r / 4, 1);
    for (const [cls, k, poly] of editable()) {
      const a = poly.map((v) => [v[0] * S.iw, v[1] * S.ih]);
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const [x, y] of a) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
      if (x0 - r > c[0] || y0 - r > c[1] || x1 + r < c[0] || y1 + r < c[1]) continue;
      // subdivide edges passing through the brush so the outline bends smoothly instead of kinking
      const px = [];
      for (let i = 0; i < a.length; i++) {
        const A = a[i], B = a[(i + 1) % a.length];
        const n = segDist(c, A, B) < r ? Math.max(1, Math.ceil(Math.hypot(B[0] - A[0], B[1] - A[1]) / step)) : 1;
        for (let j = 0; j < n; j++) px.push([A[0] + ((B[0] - A[0]) * j) / n, A[1] + ((B[1] - A[1]) * j) / n]);
      }
      let moved = false;
      for (const p of px) {
        const w = Math.max(0, 1 - ((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) / (r * r)) ** 2;
        if (w > 0) { p[0] += w * dx; p[1] += w * dy; moved = true; }
      }
      if (!moved) continue;
      S.polys[cls][k] = px.map(([x, y]) => [Math.min(Math.max(x, 0), S.iw) / S.iw, Math.min(Math.max(y, 0), S.ih) / S.ih]);
      S.nudged = true;
    }
    if (S.nudged) changed();
  }

  function nearFirst(x, y) {
    if (S.points.length < 3) return false;
    const [fx, fy] = toDisp(...S.points[0]);
    return Math.hypot(x - fx, y - fy) <= CLOSE_PX;
  }

  function closePoly() {
    const pts = S.points;
    S.points = []; S.strokes = []; S.tracing = false;
    if (pts.length < 3 || area(pts) * S.s * S.s < 25) {
      S.message = "polygon too small, discarded";
      return;
    }
    pushUndo();
    S.pending = true;
    S.trigger("op", {
      kind: S.mode === "cut" ? "cut" : "merge", cls: S.active, dirty: S.dirty,
      lasso: pts.map(([x, y]) => [x / S.iw, y / S.ih]), polys: S.polys,
    });
  }

  function undo() {
    if (S.points.length) {
      S.points.splice(S.strokes.pop());
      return;
    }
    S.drag = null;
    if (!S.undo.length) return;
    S.polys = S.undo.pop();
    changed();
  }

  function setMode(m) {
    S.mode = S.mode === m ? "draw" : m;
    S.points = []; S.strokes = []; S.tracing = false; S.drag = null; S.pan = null; S.nudgeLast = null;
  }

  // ------------------------------------------------------------ drawing
  function redraw() {
    if (S.raf) return;
    S.raf = requestAnimationFrame(() => { S.raf = 0; render(); });
  }

  function tracePath(p) {
    ctx.beginPath();
    p.forEach((v, i) => {
      const [x, y] = normToDisp(v);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.closePath();
  }

  function render() {
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = "#282828";
    ctx.fillRect(0, 0, W, H);
    if (S.img && S.img.complete) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(S.img, 0, 0, S.img.naturalWidth, S.img.naturalHeight, -S.ox * S.s, -S.oy * S.s, S.iw * S.s, S.ih * S.s);
    }
    if (S.showOthers) {
      for (const [c, lst] of Object.entries(S.polys)) {
        if (c === S.active) continue;
        ctx.strokeStyle = color(c);
        ctx.lineWidth = 1;
        for (const p of lst) { tracePath(p); ctx.stroke(); }
      }
    }
    ctx.strokeStyle = ctx.fillStyle = color(S.active);
    ctx.lineWidth = 2;
    for (const p of cur()) {
      tracePath(p);
      ctx.globalAlpha = 0.25; ctx.fill();
      ctx.globalAlpha = 1; ctx.stroke();
    }
    if (S.mode === "edit") {
      const handles = [];
      for (const [c, , p] of editable()) {
        for (const v of p) {
          const [x, y] = normToDisp(v);
          if (x >= 0 && x < W && y >= 0 && y < H) handles.push([c, x, y]);
          if (handles.length > MAX_HANDLES) break;
        }
      }
      if (handles.length <= MAX_HANDLES) {
        for (const [c, x, y] of handles) {
          ctx.fillStyle = color(c);
          ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill();
        }
      }
      const hit = S.drag || vertexAt(...S.mouse);
      if (hit) {
        const [x, y] = normToDisp(S.polys[hit[0]][hit[1]][hit[2]]);
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.stroke();
      }
    }
    if (S.points.length) {
      const pts = S.points.map(([x, y]) => toDisp(x, y)).concat([S.mouse]);
      ctx.strokeStyle = S.mode === "cut" ? "#ff3030" : "#ffffff";
      ctx.lineWidth = S.mode === "cut" ? 2 : 1;
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
      ctx.strokeStyle = "#ff3030"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], CLOSE_PX, 0, 2 * Math.PI); ctx.stroke();
    }
    if (S.mode === "nudge") {
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(S.mouse[0], S.mouse[1], S.brushR, 0, 2 * Math.PI); ctx.stroke();
    }
    updateBar();
  }

  const HINTS = {
    draw: "Draw: click vertices or drag to trace; click first vertex / Enter closes (merges with overlapping polygons).",
    edit: "Edit: drag vertex, click edge = add vertex, right-click / Delete = remove vertex.",
    cut: "Cut: outline the part of the active class to remove (must reach the polygon edge).",
    nudge: "Nudge: drag to push outlines; [ / ] or Ctrl+wheel = brush size.",
    pan: "Pan: left-drag to pan.",
  };

  function updateBar() {
    for (const b of bar.querySelectorAll("button[data-cls]")) b.classList.toggle("on", b.dataset.cls === S.active);
    for (const b of bar.querySelectorAll("button[data-mode]")) b.classList.toggle("on", b.dataset.mode === S.mode);
    const hide = bar.querySelector("button[data-act=hide]");
    if (hide) hide.classList.toggle("on", !S.showOthers);
    const nav = S.nav || {};
    const prev = bar.querySelector("button[data-act=prev]"), next = bar.querySelector("button[data-act=next]");
    if (prev) prev.disabled = !nav.prev;
    if (next) next.disabled = !nav.next;
    const pos = bar.querySelector(".pos");
    if (pos) pos.textContent = nav.label || "";
    const sv = bar.querySelector("button[data-act=save]");
    const svState = S.saving ? "saving" : "idle";
    // redraw runs on every mouse move, so only touch the DOM when the state flips
    if (sv && sv.dataset.state !== svState) {
      sv.dataset.state = svState;
      sv.innerHTML = S.saving ? '<span class="spin"></span>Saving\u2026' : "Save";
      sv.title = S.saving ? "Re-measuring this image in the background; saving again queues another"
                          : "Save and re-measure (s)";
    }
    const counts = S.classes.map((c) => `${c.name}: ${(S.polys[c.name] || []).length}`).join("  ");
    statusEl.textContent = [counts, S.dirty ? "unsaved changes" : "saved", S.pending ? "working..." : "", S.message]
      .filter(Boolean).join("  |  ");
    statusEl.classList.toggle("dirty", S.dirty);
    hintEl.textContent = HINTS[S.mode] + "  Right/middle-drag = pan, wheel = zoom, 1/2/3 class, d/e/c/m/g modes, " +
      "x delete polygon, h hide others, f fit, u / Ctrl+Z undo, s save, p / n previous / next image.";
  }

  function buildBar() {
    bar.replaceChildren();
    let group = null;
    const newGroup = (cls = "") => {
      group = document.createElement("div");
      group.className = "grp " + cls;
      bar.appendChild(group);
    };
    const add = (label, attrs, title) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      if (title) b.title = title;
      for (const [k, v] of Object.entries(attrs)) b.dataset[k] = v;
      group.appendChild(b);
      return b;
    };
    newGroup();
    S.classes.forEach((c, i) => {
      const b = add(`${i + 1}: ${c.name}`, { cls: c.name }, `Active class (${i + 1})`);
      b.style.borderColor = c.color;
    });
    newGroup();
    add("Draw", { mode: "draw" }, "Draw / merge polygons (d)");
    add("Edit", { mode: "edit" }, "Move, add, delete vertices (e)");
    add("Cut", { mode: "cut" }, "Cut away part of the active class (c)");
    add("Nudge", { mode: "nudge" }, "Push outlines with a brush (m)");
    add("Pan", { mode: "pan" }, "Left-drag pans (g)");
    newGroup();
    add("Hide others", { act: "hide" }, "Show only the active class (h)");
    add("Undo", { act: "undo" }, "u / Ctrl+Z");
    add("Fit", { act: "fit" }, "Fit image to view (f)");
    newGroup("right");
    add("Save", { act: "save" }, "Save and re-measure (s)").classList.add("primary");
    add("\u25C0", { act: "prev" }, "Previous image (p)");
    const pos = document.createElement("span");
    pos.className = "pos";
    group.appendChild(pos);
    add("\u25B6", { act: "next" }, "Next image (n)");
  }

  function save() {
    if (S.pending) return;
    S.pending = true;
    S.trigger("save", { polys: S.polys });
    redraw();
  }

  function navigate(d) {
    const nav = S.nav || {};
    if (!(d < 0 ? nav.prev : nav.next)) return;
    if ((S.dirty || S.points.length) &&
        !window.confirm(`Discard unsaved edits and go to the ${d < 0 ? "previous" : "next"} image?`)) return;
    S.trigger("nav", d);
  }

  bar.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || (S.pending && b.dataset.act !== "fit" && b.dataset.act !== "hide")) return;
    if (b.dataset.cls) S.active = b.dataset.cls;
    else if (b.dataset.mode) setMode(b.dataset.mode);
    else if (b.dataset.act === "hide") S.showOthers = !S.showOthers;
    else if (b.dataset.act === "undo") undo();
    else if (b.dataset.act === "fit") fit();
    else if (b.dataset.act === "save") save();
    else if (b.dataset.act === "prev") navigate(-1);
    else if (b.dataset.act === "next") navigate(1);
    wrap.focus();
    redraw();
  });

  // ------------------------------------------------------------ events
  const pos = (e) => {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
  };

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("pointerdown", (e) => {
    wrap.focus();
    if (S.pending) return;
    const [x, y] = pos(e);
    S.mouse = [x, y];
    const [ix, iy] = clampImg(toImg(x, y));
    canvas.setPointerCapture(e.pointerId);
    if (e.button === 2 && S.mode === "edit") {
      const hit = vertexAt(x, y);
      if (hit) { deleteVertex(hit); redraw(); return; }
    }
    if (e.button === 1 || e.button === 2 || (e.button === 0 && S.mode === "pan")) {
      S.pan = [e.button, x, y];
      S.tracing = false;
      return;
    }
    if (e.button !== 0) return;
    if (S.mode === "edit") {
      const hit = vertexAt(x, y);
      const edge = hit ? null : edgeAt(x, y);
      if (!hit && !edge) return;
      const [c, k, vi] = hit || edge;
      pushUndo();
      if (edge) S.polys[c][k].splice(vi, 0, [ix / S.iw, iy / S.ih]);
      S.drag = [c, k, vi];
      S.dragMoved = !!edge;
    } else if (S.mode === "nudge") {
      pushUndo();
      S.nudgeLast = toImg(x, y);
      S.nudged = false;
    } else if (nearFirst(x, y)) {
      closePoly();
    } else {
      S.strokes.push(S.points.length);
      S.points.push([ix, iy]);
      S.tracing = true;
    }
    redraw();
  });

  canvas.addEventListener("pointermove", (e) => {
    const [x, y] = pos(e);
    S.mouse = [x, y];
    if (S.pan) {
      S.ox -= (x - S.pan[1]) / S.s;
      S.oy -= (y - S.pan[2]) / S.s;
      S.pan = [S.pan[0], x, y];
    } else if (S.drag) {
      const [ix, iy] = clampImg(toImg(x, y));
      const [c, k, vi] = S.drag;
      S.polys[c][k][vi] = [ix / S.iw, iy / S.ih];
      S.dragMoved = true;
      changed();
    } else if (S.nudgeLast) {
      nudge(...toImg(x, y));
    } else if (S.tracing && e.buttons & 1) {
      const [lx, ly] = toDisp(...S.points[S.points.length - 1]);
      if (Math.hypot(x - lx, y - ly) >= TRACE_PX) S.points.push(clampImg(toImg(x, y)));
    }
    redraw();
  });

  const endPointer = (e) => {
    const [x, y] = pos(e);
    if (S.pan && S.pan[0] === e.button) S.pan = null;
    if (S.drag) {
      S.drag = null;
      // a click on a vertex without moving it leaves nothing to undo
      if (!S.dragMoved) S.undo.pop();
    }
    if (S.nudgeLast) {
      S.nudgeLast = null;
      if (!S.nudged) S.undo.pop();
    }
    if (S.tracing && S.points.length - S.strokes[S.strokes.length - 1] > 1 && nearFirst(x, y)) closePoly();
    S.tracing = false;
    redraw();
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("pointerleave", () => redraw());

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [x, y] = pos(e);
    const up = e.deltaY < 0;
    if (S.mode === "nudge" && (e.ctrlKey || e.shiftKey)) {
      S.brushR = Math.min(Math.max(S.brushR * (up ? 1.2 : 1 / 1.2), NUDGE_R_MIN), NUDGE_R_MAX);
      redraw();
    } else {
      zoom(up ? 1.25 : 0.8, x, y);
    }
  }, { passive: false });

  wrap.addEventListener("keydown", (e) => {
    const k = e.key;
    // edits made while Python works on a merge / cut would be overwritten by its result
    if (S.pending && !"fh+=-".includes(k)) { e.preventDefault(); return; }
    let handled = true;
    if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === "z") undo();
    else if (e.ctrlKey || e.metaKey || e.altKey) handled = false;
    else if (/^[1-9]$/.test(k) && S.classes[+k - 1]) S.active = S.classes[+k - 1].name;
    else if (k === "Tab") {
      const i = S.classes.findIndex((c) => c.name === S.active);
      S.active = S.classes[(i + 1) % S.classes.length].name;
    } else if (k === "e") setMode("edit");
    else if (k === "d") setMode("draw");
    else if (k === "c") setMode("cut");
    else if (k === "m") setMode("nudge");
    else if (k === "g") setMode("pan");
    else if (k === "h") S.showOthers = !S.showOthers;
    else if (k === "f") fit();
    else if (k === "+" || k === "=") zoom(1.25, canvas.width / 2, canvas.height / 2);
    else if (k === "-") zoom(0.8, canvas.width / 2, canvas.height / 2);
    else if (k === "u") undo();
    else if (k === "x") deleteUnderCursor();
    else if (k === "[") S.brushR = Math.max(S.brushR / 1.2, NUDGE_R_MIN);
    else if (k === "]") S.brushR = Math.min(S.brushR * 1.2, NUDGE_R_MAX);
    else if (k === "s") save();
    else if (k === "p") navigate(-1);
    else if (k === "n") navigate(1);
    else if (k === "Enter" && S.points.length) closePoly();
    else if (k === "Delete" || k === "Backspace") {
      if (S.mode === "edit") deleteVertex(vertexAt(...S.mouse));
    } else if (k === "Escape") {
      if (S.points.length) { S.points = []; S.strokes = []; S.tracing = false; }
      else S.mode = "draw";
    } else handled = false;
    if (handled) { e.preventDefault(); e.stopPropagation(); redraw(); }
  });

  new ResizeObserver(resize).observe(wrap);

  // ------------------------------------------------------------ Python -> JS
  function update(data) {
    if (!data) return;
    canvas.height = data.height || 720;
    S.nav = data.nav || null;
    S.saving = !!data.saving;
    if (data.doc !== S.doc) {
      S.doc = data.doc;
      S.iw = data.iw; S.ih = data.ih;
      S.classes = data.classes || [];
      S.active = S.classes.some((c) => c.name === S.active) ? S.active : (S.classes[0] || {}).name;
      S.undo = []; S.points = []; S.strokes = []; S.mode = "draw"; S.version = -1;
      buildBar();
      canvas.width = Math.max(200, Math.floor(wrap.clientWidth));
      S.img = null;
      fit();
    }
    if (data.image && (!S.img || S.img.src !== data.image)) {
      const img = new Image();
      img.onload = () => { if (S.img === img) fit(); };
      img.src = data.image;
      S.img = img;
    } else if (!data.image && !S.img) {
      S.trigger("need_image", true);
    }
    if (data.version !== S.version) {
      S.version = data.version;
      S.polys = clone(data.polys || {});
      for (const c of S.classes) S.polys[c.name] = S.polys[c.name] || [];
      S.dirty = !!data.dirty;
      S.message = data.message || "";
      if (data.noop && S.pending) S.undo.pop();
      S.pending = false;
    }
    redraw();
  }

  return { update, S };
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let ed = INSTANCES.get(parentElement);
  if (!ed) {
    ed = createEditor(parentElement);
    INSTANCES.set(parentElement, ed);
  }
  ed.S.trigger = setTriggerValue;
  ed.update(data);
}
