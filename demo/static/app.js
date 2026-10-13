/* Live demo frontend. Plain JS, no libraries, works offline.
   Talks to demo/server.py: /api/info, /api/predict, /api/metrics. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

/* ---------- safe HTML rendering ----------
   Every dynamic value goes through esc(). render() parses markup in an inert document,
   drops <script> and on* attributes, then swaps the nodes in. */
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function render(el, markup) {
  const doc = new DOMParser().parseFromString(`<body>${markup}</body>`, "text/html");
  doc.querySelectorAll("script, iframe, object, embed").forEach(n => n.remove());
  doc.querySelectorAll("*").forEach(n => [...n.attributes].forEach(a => {
    if (/^on/i.test(a.name) || /^\s*javascript:/i.test(a.value)) n.removeAttribute(a.name);
  }));
  el.replaceChildren(...[...doc.body.childNodes].map(n => document.importNode(n, true)));
}

const state = {
  info: null, metrics: null,
  result: null,            // last /api/predict response
  source: null,            // {kind: "sample"|"upload"|"webcam", sample?}
  model: "driver",         // model shown in "Inside the network"
  step: -1, timer: null,
  bars: new Array(10).fill(0),
};

const LAST_STEP = 8;
const DECK_SIZES = [5.8, 5.3, 4.8, 4.3, 3.8, 3.8];   // rem; the decks shrink like the real tensors do
const MODEL_COLOR = { random: "#c97a10", driver: "#3d8fe0" };
const MODEL_NAME = { random: "Random-split model", driver: "Driver-split model" };

/* ---------- colour ramps ---------- */
// Activation ramp: one hue (signal mint), dark -> bright.
const SIGNAL_LUT = makeLut([[0, "#0f1011"], [0.3, "#0d3634"], [0.6, "#1c9282"], [0.85, "#7fe3cf"], [1, "#f2fffb"]]);

function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
function makeLut(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  const s = stops.map(([p, c]) => [p, hexToRgb(c)]);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0; while (k < s.length - 2 && t > s[k + 1][0]) k++;
    const [p0, c0] = s[k], [p1, c1] = s[k + 1];
    const f = Math.min(1, Math.max(0, (t - p0) / (p1 - p0)));
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * f;
  }
  return lut;
}
function b64ToU8(b64) {
  const bin = atob(b64); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/* ---------- drawing ---------- */
function drawMap(canvas, u8, size) {
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d"); const img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = u8[i] * 3;
    img.data[i * 4] = SIGNAL_LUT[v]; img.data[i * 4 + 1] = SIGNAL_LUT[v + 1]; img.data[i * 4 + 2] = SIGNAL_LUT[v + 2]; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// Photo, optionally with a Grad-CAM heatmap on the same mint ramp as the activations (white-hot = looked hardest).
function drawPhoto(canvas, rgbB64, camB64, _color, opacity) {
  const size = state.result.size;
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d"); const img = ctx.createImageData(size, size);
  const rgb = b64ToU8(rgbB64);
  const cam = camB64 && opacity > 0 ? b64ToU8(camB64) : null;
  for (let i = 0; i < size * size; i++) {
    let r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    if (cam) {
      // darken where the model did not look; paint the heat ramp where it did
      const v = cam[i] / 255, k = cam[i] * 3;
      const a = Math.min(1, Math.pow(v, 0.9) * 1.1) * opacity;
      const dim = 1 - 0.8 * opacity * (1 - v);
      r = r * dim * (1 - a) + SIGNAL_LUT[k] * a;
      g = g * dim * (1 - a) + SIGNAL_LUT[k + 1] * a;
      b = b * dim * (1 - a) + SIGNAL_LUT[k + 2] * a;
    }
    img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// The 960 averaged numbers drawn one pixel each, as a 30 x 32 block (pixel-honest).
function drawPool(canvas, pooled) {
  const w = 30, h = Math.ceil(pooled.length / w);
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d"); const img = ctx.createImageData(w, h);
  const max = Math.max(...pooled, 1e-6);
  pooled.forEach((v, i) => {
    const k = Math.round(Math.pow(Math.max(0, v) / max, 0.6) * 255) * 3;
    img.data[i * 4] = SIGNAL_LUT[k]; img.data[i * 4 + 1] = SIGNAL_LUT[k + 1]; img.data[i * 4 + 2] = SIGNAL_LUT[k + 2]; img.data[i * 4 + 3] = 255;
  });
  ctx.putImageData(img, 0, 0);
}

/* ---------- static structure ---------- */
function buildStages() {
  const ol = $("#stages");
  render(ol, DECK_SIZES.map((d, s) => `
    <li class="node stage" data-step="${s + 1}" data-stage="${s}" tabindex="0" role="button" style="--deck:${d}rem">
      <header><span class="no">${String(s + 2).padStart(2, "0")}</span><h2>Layer</h2></header>
      <div class="vis"><div class="deck">${
        Array.from({ length: 8 }, (_, i) => `<canvas style="--i:${i}"></canvas>`).reverse().join("")
      }</div></div>
      <footer><p class="dims"></p><p class="cap"></p></footer>
    </li>`).join(""));
  $$(".stage", ol).forEach(li => {
    const s = +li.dataset.stage;
    li.addEventListener("click", () => openDrawer(s));
    li.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); openDrawer(s); } });
  });
}

function buildOutputs() {
  const { classes, class_names } = state.info;
  render($("#outputs"), classes.map(c => `
    <li data-c="${esc(c)}"><span class="code">${esc(c)}</span>
      <span class="bar"><i></i><span class="name">${esc(nice(class_names[c]))}</span><span class="true-tag">true</span></span>
      <span class="pct">–</span></li>`).join(""));
}

function buildRack() {
  // one photo per class first, so the visible row covers c0–c9
  const seen = new Set(), first = [], rest = [];
  state.info.samples.forEach(s => (seen.has(s.true_class) ? rest : (seen.add(s.true_class), first)).push(s));
  state.info.samples = [...first, ...rest];
  $("#rack-count").textContent = `${state.info.samples.length} photos`;
  const ul = $("#thumbs");
  render(ul, state.info.samples.map((s, i) => `
    <li><button data-i="${i}" title="${esc(s.true_class)}: ${esc(s.true_name)} · driver ${esc(s.subject)}">
      <img src="/samples/${encodeURIComponent(s.file)}" alt="${esc(s.true_name)}, driver ${esc(s.subject)}">
      <span class="tag">${esc(s.true_class)}</span></button></li>`).join(""));
  $("#rack-left").addEventListener("click", () => ul.scrollBy({ left: -ul.clientWidth * 0.8 }));
  $("#rack-right").addEventListener("click", () => ul.scrollBy({ left: ul.clientWidth * 0.8 }));
  ul.addEventListener("click", e => {
    const b = e.target.closest("button[data-i]"); if (!b) return;
    runSample(state.info.samples[+b.dataset.i]);
  });
}

/* ---------- prediction ---------- */
async function predict(form) {
  form.append("models", "random,driver");
  const res = await fetch("/api/predict", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

async function runSample(sample) {
  $$("#thumbs button").forEach(b => b.setAttribute("aria-current", String(state.info.samples[+b.dataset.i] === sample)));
  const f = new FormData(); f.append("sample", sample.file);
  await run(f, { kind: "sample", sample });
}

async function runFile(file, kind = "upload") {
  $$("#thumbs button").forEach(b => b.setAttribute("aria-current", "false"));
  const f = new FormData(); f.append("file", file, file.name || "photo.jpg");
  await run(f, { kind });
}

async function run(form, source) {
  stopPlay();
  sayText("Running both models on this laptop's CPU…");
  try {
    state.result = await predict(form);
    state.source = source;
  } catch (err) {
    sayText(`Couldn't classify that image (${err.message}). Try a JPG or PNG photo, or pick one from the rack.`);
    return;
  }
  renderInside();
  renderCompare();
  play();
}

/* ---------- INSIDE THE NETWORK ---------- */
function current() { return state.result.models[state.model]; }

function renderInside() {
  const m = current();
  document.documentElement.style.setProperty("--model", MODEL_COLOR[state.model]);
  $("#cpu-ms").textContent = Math.round(m.ms);
  renderPhoto();
  m.stages.forEach((st, s) => {
    const li = $$("#stages .stage")[s];
    $("h2", li).textContent = st.title;
    li.style.setProperty("--dz", `${(0.25 + 0.05 * (Math.log2(st.channels) - 4)).toFixed(2)}rem`);   // deeper stack = more channels (capped so stacks never collide)
    const a = document.createElement("span"), b = document.createElement("span");
    a.textContent = `${st.size} × ${st.size}`; b.textContent = `${st.channels} maps`;
    $(".dims", li).replaceChildren(a, b);
    $(".cap", li).textContent = st.caption;
    const canvases = $$("canvas", li).reverse();   // front-most canvas = most active map
    st.maps.forEach((b64, i) => drawMap(canvases[i], b64ToU8(b64), st.map_size));
  });
  drawPool($("#pool"), m.pooled);
  const truth = state.source?.sample?.true_class;
  $$("#outputs li").forEach(li => {
    li.classList.remove("win");
    li.classList.toggle("truth", li.dataset.c === truth);
  });
  $("#verdict").replaceChildren();
  if (!$("#drawer").hidden) openDrawer(+$("#drawer").dataset.stage);
}

function renderPhoto() {
  if (!state.result) return;
  const on = $("#cam-on").checked;
  $("#cam-opacity").disabled = !on;
  const m = current();
  drawPhoto($("#photo"), state.result.input, on ? m.gradcam : null, MODEL_COLOR[state.model], on ? $("#cam-opacity").value / 100 : 0);
}

function nodeAt(step) {
  if (step === 0) return $(".node-input");
  if (step <= 6) return $$("#stages .stage")[step - 1];
  return step === 7 ? $(".node-pool") : $(".node-out");
}

function setStep(n) {
  state.step = n;
  for (let k = 0; k <= LAST_STEP; k++) {
    const el = nodeAt(k);
    el.classList.toggle("lit", k <= n);
    el.classList.toggle("current", k === n);
  }
  movePulse(n);
  narrate(n);
  if (n >= LAST_STEP) settleOutputs(); else resetOutputs();
}

function movePulse(n) {
  const fig = $("#figure").getBoundingClientRect();
  if (!fig.width) return;
  const centre = el => { const r = ($(".vis", el) || el).getBoundingClientRect(); return r.left + r.width / 2 - fig.left; };
  const x0 = centre(nodeAt(0));
  const xEnd = $(".node-out").getBoundingClientRect().left - fig.left - 12;
  const x = n >= LAST_STEP ? xEnd : centre(nodeAt(n));
  const wire = $(".wire");
  wire.style.left = `${x0}px`; wire.style.right = `${fig.width - xEnd}px`;
  $("#wire-lit").style.width = `${Math.max(0, x - x0)}px`;
  const p = $("#pulse");
  p.style.left = `${x - p.offsetWidth / 2}px`;
  p.classList.add("on");
}

function narrate(n) {
  const m = current();
  let text;
  if (n === 0) text = "The photo is resized to 224 × 224 pixels, so it becomes 224 × 224 × 3 numbers (red, green, blue).";
  else if (n <= 6) {
    const st = m.stages[n - 1];
    text = `${st.title} of 17. ${st.caption}. Shown: the 8 most active of its ${st.channels} feature maps, each ${st.size} × ${st.size}. Click a stack to see them up close.`;
  } else if (n === 7) text = "Average pooling squeezes each of the 960 final maps into one number: a 960-number summary of what was found.";
  else text = "A final layer turns those 960 numbers into a score for each of the 10 classes. The highest score is the prediction.";
  render($("#say"), `<span class="step">${String(n + 1).padStart(2, "0")} / 09</span>${esc(text)}`);
}

function sayText(text) { $("#say").textContent = text; }

function resetOutputs() {
  cancelAnimationFrame(springRaf); clearTimeout(lockTimer); locked = false;
  state.bars = new Array(10).fill(0);
  $$("#outputs li").forEach(li => { $("i", li).style.width = "0%"; $(".pct", li).textContent = "–"; li.classList.remove("win"); });
  $("#verdict").replaceChildren();
}

// Output bars settle with a damped spring (a little overshoot, then rest) instead of jumping.
let springRaf = null, lockTimer = null, locked = false;
function settleOutputs() {
  const targets = current().probs;
  const lis = $$("#outputs li");
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {   // no spring: show final values at once
    lis.forEach((li, i) => { $("i", li).style.width = `${targets[i] * 100}%`; });
    state.bars = targets.slice(); lockVerdict(); return;
  }
  const xs = state.bars.slice(), vs = new Array(10).fill(0);
  cancelAnimationFrame(springRaf); clearTimeout(lockTimer); locked = false;
  lockTimer = setTimeout(lockVerdict, 700);   // the verdict locks even if the spring is still settling
  let last = performance.now();
  const tick = now => {
    const dt = Math.min(0.033, (now - last) / 1000); last = now;
    let moving = false;
    for (let i = 0; i < 10; i++) {
      const a = 170 * (targets[i] - xs[i]) - 15 * vs[i];
      vs[i] += a * dt; xs[i] += vs[i] * dt;
      if (Math.abs(vs[i]) > 1e-3 || Math.abs(targets[i] - xs[i]) > 1e-3) moving = true;
      $("i", lis[i]).style.width = `${Math.max(0, Math.min(1.04, xs[i])) * 100}%`;
      if (!locked) $(".pct", lis[i]).textContent = `${(Math.max(0, xs[i]) * 100).toFixed(0)}%`;
    }
    state.bars = xs;
    if (moving) springRaf = requestAnimationFrame(tick);
    else { lis.forEach((li, i) => { $(".pct", li).textContent = `${(targets[i] * 100).toFixed(0)}%`; }); lockVerdict(); }
  };
  springRaf = requestAnimationFrame(tick);
}

function truthLine(sample, pred, who = "") {
  if (!sample) return "";
  const ok = sample.true_class === pred;
  return `<p class="truth ${ok ? "ok" : "bad"}"><svg><use href="#i-${ok ? "check" : "cross"}"/></svg>${who ? esc(who) + " · " : ""}${ok ? "Correct" : "Wrong"}: true label ${esc(sample.true_class)} ${esc(nice(sample.true_name))}</p>`;
}
const nice = s => String(s).replace(" - ", " – ");

function lockVerdict() {
  const m = current(); locked = true;
  $$("#outputs li").forEach((li, i) => {
    li.classList.toggle("win", li.dataset.c === m.pred);
    $(".pct", li).textContent = `${(m.probs[i] * 100).toFixed(0)}%`;   // final values, even mid-spring
  });
  const sample = state.source?.sample;
  const note = sample ? truthLine(sample, m.pred, MODEL_NAME[state.model])
    : `<p class="truth">${MODEL_NAME[state.model]} · ${state.source?.kind === "webcam" ? "webcam photo: experimental, no true label" : "uploaded photo: no true label"}</p>`;
  render($("#verdict"), `<p class="answer">${esc(cap(nice(m.pred_name)))}<span class="pct">${(m.confidence * 100).toFixed(0)}%</span></p>${note}`);
}
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

function play() {
  stopPlay();
  let n = 0; setStep(0);
  state.timer = setInterval(() => {
    n++; setStep(n);
    if (n >= LAST_STEP) stopPlay();
  }, 820);
}
function stopPlay() { clearInterval(state.timer); state.timer = null; }

function step(delta) {
  if (!state.result) return;
  stopPlay();
  setStep(Math.max(0, Math.min(LAST_STEP, (state.step < 0 ? 0 : state.step) + delta)));
}

function openDrawer(s) {
  if (!state.result) return;
  const st = current().stages[s];
  const d = $("#drawer"); d.hidden = false; d.dataset.stage = s;
  $("#drawer-title").textContent = `${st.title}: the 8 most active feature maps`;
  $("#drawer-sub").textContent = `${st.caption}. Each map is ${st.size} × ${st.size}, drawn pixel for pixel; brighter means a stronger response.`;
  const box = $("#drawer-maps");
  box.replaceChildren(...st.maps.map((b64, i) => {
    const fig = document.createElement("figure");
    const c = document.createElement("canvas");
    drawMap(c, b64ToU8(b64), st.map_size);
    const cap = document.createElement("figcaption"); cap.textContent = `channel ${st.map_ids[i]} of ${st.channels}`;
    fig.append(c, cap); return fig;
  }));
  d.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------- TWO MODELS ---------- */
function splitAcc(split) {
  const t = state.metrics?.test?.[split]?.MobileNetV3;
  return t ? `${(t.accuracy * 100).toFixed(1)}%` : "n/a";
}

function renderCompare() {
  if (!state.result) return;
  const s = state.source?.sample;
  render($("#compare-lede"), s
    ? `Same photo, two models. Neither saw <em>this photo</em> in training, but the random-split model did train on other photos of <em>this driver (${esc(s.subject)})</em>.`
    : `Your photo, two models. It has no true label, so compare what each model predicts and where it looks.`);
  const dTrain = state.metrics?.splits?.driver?.train?.n_drivers;
  const text = {
    random: "Trained on a random 70% of all photos. Every driver, including the test drivers, appears in its training data.",
    driver: `Trained on ${dTrain ? dTrain + " drivers" : "a separate group of drivers"}. It has never seen the test drivers.`,
  };
  const names = state.info.class_names, classes = state.info.classes;
  for (const split of ["random", "driver"]) {
    const m = state.result.models[split]; if (!m) continue;
    const top = m.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
    render($(`#reader-${split}`), `
      <canvas width="224" height="224" aria-label="Photo with ${MODEL_NAME[split]} heatmap"></canvas>
      <div>
        <h3><i class="swatch"></i>${MODEL_NAME[split]}</h3>
        <p class="trained">${esc(text[split])}</p>
        <p class="answer">${esc(cap(nice(m.pred_name)))}<span class="pct">${(m.confidence * 100).toFixed(0)}%</span></p>
        ${truthLine(s, m.pred)}
        <ol class="top3">${top.map(([p, i]) => `<li><span class="lab">${esc(classes[i])} ${esc(nice(names[classes[i]]))}</span><span class="pct">${(p * 100).toFixed(0)}%</span><span class="track"><i style="width:${(p * 100).toFixed(1)}%"></i></span></li>`).join("")}</ol>
        <div class="facts">
          <span>Test accuracy on its own split: <b>${esc(splitAcc(split))}</b></span>
          <span>CPU time for this photo: <b>${Math.round(m.ms)} ms</b></span>
        </div>
      </div>`);
  }
  drawCompareCams();
}
function drawCompareCams() {
  const o = $("#duo-opacity").value / 100;
  for (const split of ["random", "driver"]) {
    const c = $(`#reader-${split} canvas`); const m = state.result?.models[split];
    if (c && m) drawPhoto(c, state.result.input, m.gradcam, MODEL_COLOR[split], o);
  }
}

/* ---------- RESULTS (all numbers read from outputs/metrics.json) ---------- */
const pct = v => `${(v * 100).toFixed(1)}%`;

function groupedBars({ groups, series, height = 250, label = "Accuracy" }) {
  const W = 640, H = height, m = { t: 22, r: 8, b: 34, l: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const gw = iw / groups.length, pad = gw * 0.18, bw = (gw - 2 * pad - (series.length - 1) * 2) / series.length;
  const y = v => m.t + ih * (1 - v);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}"><g class="grid">`;
  for (const t of [0, .25, .5, .75, 1]) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end" font-size="13">${t * 100}%</text>`;
  s += `</g>`;
  groups.forEach((g, gi) => {
    series.forEach((se, si) => {
      const v = se.values[gi]; if (v == null) return;
      const x = m.l + gi * gw + pad + si * (bw + 2), top = y(v), h = m.t + ih - top, r = Math.min(4, h);
      s += `<path d="M${x},${top + h} L${x},${top + r} Q${x},${top} ${x + r},${top} L${x + bw - r},${top} Q${x + bw},${top} ${x + bw},${top + r} L${x + bw},${top + h} Z" fill="${se.color}"/>`;
      if (bw >= 18) s += `<text class="val" x="${x + bw / 2}" y="${top - 5}" text-anchor="middle">${(v * 100).toFixed(0)}</text>`;
      s += `<rect class="hit" x="${x - 1}" y="${m.t}" width="${bw + 2}" height="${ih}" data-tip="${esc(g)} · ${esc(se.label)}: ${pct(v)}"/>`;
    });
    s += `<text x="${m.l + gi * gw + gw / 2}" y="${H - m.b + 16}" text-anchor="middle" font-size="14">${esc(g)}</text>`;
  });
  return s + `</svg>`;
}

// Confusion matrix as a grid of cells; shade = share of the true class (row), in the model's colour.
function confusion(cm, color, classes, names) {
  const [r, g, b] = hexToRgb(color);
  let h = `<div class="cm" role="table"><span></span>${classes.map(c => `<span class="h">${esc(c)}</span>`).join("")}`;
  cm.forEach((row, i) => {
    const tot = row.reduce((x, y) => x + y, 0) || 1;
    h += `<span class="r" title="${esc(names[classes[i]])}">${esc(classes[i])} ${esc(nice(names[classes[i]]))}</span>`;
    row.forEach((v, j) => {
      const f = v / tot, ink = f > 0.55 ? "#0f1011" : "var(--ink-2)";
      h += `<span class="c" style="background:rgba(${r},${g},${b},${(0.08 + 0.92 * f).toFixed(3)});color:${ink}" data-tip="true ${esc(classes[i])} → predicted ${esc(classes[j])}: ${v} photos (${(f * 100).toFixed(0)}%)">${v || ""}</span>`;
    });
  });
  return h + `</div>`;
}

function legend(series) {
  return `<div class="legend">${series.map(s => `<span><i class="swatch" style="background:${s.color}"></i>${esc(s.label)}</span>`).join("")}</div>`;
}

function renderResults() {
  const M = state.metrics, body = $("#results-body");
  if (!M) { body.textContent = "No metrics.json found in outputs/. Download the Kaggle results first (see README)."; return; }
  const models = M.models, deep = "MobileNetV3";
  const d = M.drop[deep];
  const R = { label: "Random split (test drivers seen in training)", color: MODEL_COLOR.random };
  const D = { label: "Driver split (test drivers never seen)", color: MODEL_COLOR.driver };
  const cls = M.meta.classical_train_n;
  const tds = M.driver_split_drivers.test;
  const classes = M.classes;

  const accChart = groupedBars({ groups: models, label: "Test accuracy by model and split",
    series: [{ ...R, values: models.map(m => M.test.random[m].accuracy) }, { ...D, values: models.map(m => M.test.driver[m].accuracy) }] });
  const drvChart = groupedBars({ groups: tds, height: 200, label: "MobileNetV3 accuracy per unseen driver",
    series: [{ label: "MobileNetV3 (driver split)", color: MODEL_COLOR.driver, values: tds.map(t => M.per_test_driver_accuracy[deep][t]?.accuracy ?? null) }] });
  const clsChart = groupedBars({ groups: classes, height: 210, label: "MobileNetV3 accuracy per class",
    series: [{ ...R, values: classes.map(c => M.test.random[deep].per_class_accuracy[c]) }, { ...D, values: classes.map(c => M.test.driver[deep].per_class_accuracy[c]) }] });
  const rows = models.map(m => `<tr class="${m === deep ? "deep" : ""}"><td>${esc(m)}</td>
      <td>${pct(M.test.random[m].accuracy)}</td><td>${pct(M.test.driver[m].accuracy)}</td>
      <td>${M.drop[m].drop_points.toFixed(1)}</td>
      <td>${M.test.random[m].macro_f1.toFixed(3)}</td><td>${M.test.driver[m].macro_f1.toFixed(3)}</td></tr>`).join("");
  const worst = classes.map(c => [c, M.test.driver[deep].per_class_accuracy[c]]).sort((a, b) => a[1] - b[1]).slice(0, 2);

  const verb = d.driver_acc < d.random_acc ? "fell from" : "went from";
  render(body, `
    <h2 class="headline">On drivers it had never seen, MobileNetV3 ${verb} ${pct(d.random_acc)} to ${pct(d.driver_acc)} accuracy.</h2>
    <p class="sub">Same data, same model, two ways of splitting it. A random split puts photos of the same ${M.meta.n_drivers} drivers in both training and test, so part of what gets measured is how well the model recognises people. The classical models (CS229-style, 64×64 pixels, flattened) trained on ${cls.random.toLocaleString()} / ${cls.driver.toLocaleString()} images (random / driver split) for speed.</p>
    <div class="res-grid">
      <div class="card-chart">
        <h3>Test accuracy: random split vs driver split</h3>
        <p class="note">${M.splits.random.test.n_images.toLocaleString()} random-split test photos · ${M.splits.driver.test.n_images.toLocaleString()} driver-split test photos from ${tds.length} unseen drivers</p>
        ${legend([R, D])}<div class="chart">${accChart}</div>
      </div>
      <div class="card-chart">
        <h3>All numbers</h3>
        <p class="note">Accuracy, drop in percentage points, and macro-F1 on each test set</p>
        <table class="metrics"><thead><tr><th>Model</th><th>Random acc</th><th>Driver acc</th><th>Drop</th><th>F1 random</th><th>F1 driver</th></tr></thead><tbody>${rows}</tbody></table>
      </div>
      <div class="card-chart">
        <h3>MobileNetV3 per class</h3>
        <p class="note">Hardest classes on new drivers: ${worst.map(([c, v]) => `${esc(c)} ${esc(nice(M.class_names[c]))} (${pct(v)})`).join(", ")}</p>
        ${legend([R, D])}<div class="chart">${clsChart}</div>
      </div>
      <div class="card-chart">
        <h3>MobileNetV3 on each unseen test driver</h3>
        <p class="note">Accuracy varies a lot from person to person</p>
        <div class="chart">${drvChart}</div>
      </div>
    </div>
    <div class="cm-wrap">
      <div class="card-chart"><h3>Confusion matrix · MobileNetV3 · driver split</h3><p class="note">Rows: true class · columns: predicted · each row sums to 100%</p>${confusion(M.test.driver[deep].confusion_matrix, MODEL_COLOR.driver, classes, M.class_names)}</div>
      <div class="card-chart"><h3>Confusion matrix · MobileNetV3 · random split</h3><p class="note">Rows: true class · columns: predicted · each row sums to 100%</p>${confusion(M.test.random[deep].confusion_matrix, MODEL_COLOR.random, classes, M.class_names)}</div>
    </div>`);
}

/* ---------- tooltip for chart marks ---------- */
document.addEventListener("mousemove", e => {
  const t = $("#tooltip"); const hit = e.target.closest?.("[data-tip]");
  if (!hit) { t.hidden = true; return; }
  t.textContent = hit.dataset.tip; t.hidden = false;
  t.style.left = `${e.clientX + 14}px`; t.style.top = `${e.clientY + 14}px`;
});

/* ---------- tabs ---------- */
function showTab(id) {
  $$(".tabs [role=tab]").forEach(t => t.setAttribute("aria-selected", String(t.id === id)));
  $$(".view").forEach(v => { v.hidden = v.id !== $("#" + id).getAttribute("aria-controls"); });
  $("#rack").hidden = id === "tab-results";
  if (id === "tab-inside" && state.result) movePulse(Math.max(0, state.step));
}
$$(".tabs [role=tab]").forEach(t => t.addEventListener("click", () => showTab(t.id)));

/* ---------- controls ---------- */
$$(".switch [role=radio]").forEach(b => b.addEventListener("click", () => {
  if (state.model === b.dataset.model) return;
  state.model = b.dataset.model;
  $$(".switch [role=radio]").forEach(x => x.setAttribute("aria-checked", String(x === b)));
  updateModelNote();
  if (state.result) { renderInside(); play(); }
}));
function updateModelNote() {
  const n = state.metrics?.splits?.driver?.train?.n_drivers;
  $("#model-note").textContent = state.model === "driver"
    ? `Trained on ${n ? n + " drivers" : "separate drivers"}. The test drivers in the rack below were never in its training data.`
    : "Trained on a random 70% of all photos, so it has seen other photos of every driver, including the ones in the rack.";
}
$("#btn-prev").addEventListener("click", () => step(-1));
$("#btn-next").addEventListener("click", () => step(1));
$("#btn-replay").addEventListener("click", () => state.result && play());
$("#drawer-close").addEventListener("click", () => { $("#drawer").hidden = true; });
$("#cam-on").addEventListener("change", renderPhoto);
$("#cam-opacity").addEventListener("input", renderPhoto);
$("#duo-opacity").addEventListener("input", drawCompareCams);
document.addEventListener("keydown", e => {
  if (e.target.closest("input, textarea, button, [role=tab], .stage")) return;
  if ($("#view-inside").hidden) return;
  if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
  else if (e.key === " ") { e.preventDefault(); if (state.result) play(); }
});
window.addEventListener("resize", () => state.result && movePulse(Math.max(0, state.step)));

/* upload + drag and drop anywhere */
$("#file").addEventListener("change", e => { const f = e.target.files[0]; if (f) runFile(f); e.target.value = ""; });
let dragDepth = 0;
window.addEventListener("dragenter", e => { if (e.dataTransfer?.types.includes("Files")) { dragDepth++; $("#dropzone").hidden = false; } });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("#dropzone").hidden = true; } });
window.addEventListener("dragover", e => e.preventDefault());
window.addEventListener("drop", e => {
  e.preventDefault(); dragDepth = 0; $("#dropzone").hidden = true;
  const f = [...(e.dataTransfer?.files || [])].find(f => f.type.startsWith("image/"));
  if (f) { if (!$("#view-results").hidden) showTab("tab-inside"); runFile(f); }
  else sayText("That wasn't an image. Drop a JPG or PNG photo.");
});

/* webcam (experimental) */
let stream = null;
$("#btn-cam").addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    $("#video").srcObject = stream; $("#webcam").hidden = false;
  } catch (err) { sayText(`Couldn't open the webcam (${err.message}). Check the browser's camera permission.`); }
});
function closeCam() { stream?.getTracks().forEach(t => t.stop()); stream = null; $("#webcam").hidden = true; }
$("#btn-cam-close").addEventListener("click", closeCam);
$("#btn-snap").addEventListener("click", () => {
  const v = $("#video"); const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0);
  c.toBlob(b => { if (b) runFile(new File([b], "webcam.jpg", { type: "image/jpeg" }), "webcam"); }, "image/jpeg", 0.92);
});

/* ---------- boot ---------- */
// ?still turns off CSS transitions (used for screenshots, where transition timing is unreliable)
if (new URLSearchParams(location.search).has("still")) document.documentElement.classList.add("still");

(async function boot() {
  try {
    state.info = await (await fetch("/api/info")).json();
  } catch { sayText("Can't reach the demo server. Start it with: python demo/run.py"); return; }
  state.metrics = await fetch("/api/metrics").then(r => r.ok ? r.json() : null).catch(() => null);
  $("#smoke-banner").hidden = !state.info.smoke;
  buildStages(); buildOutputs(); buildRack(); updateModelNote(); renderResults();
  const tab = { "#compare": "tab-compare", "#results": "tab-results" }[location.hash];
  if (tab) showTab(tab);
  const first = state.info.samples.find(s => s.true_class === "c1") || state.info.samples[0];
  if (first) runSample(first); else sayText("No sample photos found. Upload a photo to start.");
})();
