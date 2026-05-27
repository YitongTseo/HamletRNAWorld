// entrypoint — modules will be extracted from this file one at a time
import * as THREE from 'three';
import { canvas, renderer, scene, camera, composer, bloom, resize, WORLD_W, WORLD_H } from './three-scene.js';
import { bodyMaterial, organMaterial, setMidline, getMidline, getWormMesh } from './worm-render.js';
import { textcanvas, drawTextCanvas } from './text-canvas.js';
import {
  buildPositions,
  drawNetworkPanel,
  isNetVisible,
  toggleNetVisible,
  toggleXrayMode,
  toggleMotorLabels,
  getMotorLabelsVisible,
} from './network-panel.js';
import { drawChemoPanel, toggleChemo, isChemoVisible } from './chemo-panel.js';
import { drawRadar, toggleRadar, isRadarVisible } from './radar-panel.js';
import * as chrome from './panel-chrome.js';
import * as magnifier from './magnifier.js';
// Bridge module — synchronizes the network panel's labels-toggle setter
// with the magnifier's labels-toggle setter so the 'l' key updates both.
import { setXrayLabelsVisible as setNetworkXrayLabelsVisible } from './network-panel.js';

// Register the static HTML overlays (#title-nav-wrap, #help) with
// panel-chrome so the dock can hide / reopen them like any other panel.
// The panel modules (network, chemo, radar, pca) register themselves at
// their own import time — these two don't have modules, so they're
// registered here in the entrypoint.
chrome.register({
  id: 'titleNav',
  glyph: '>><>><',
  label: 'worm title + nav',
  panelEl: document.getElementById('title-nav-wrap'),
});
chrome.register({
  id: 'help',
  glyph: '[ ? ]',
  label: 'keyboard legend',
  panelEl: document.getElementById('help'),
});

const hud = document.getElementById('hud');

// ---------------------------------------------------------------------------
// Food — yellow-ish bacterial spots. Modest emission so they bloom slightly.
// ---------------------------------------------------------------------------
const foodMaterial = new THREE.MeshBasicMaterial({
  color: 0xffd040,
  toneMapped: false,
  side: THREE.DoubleSide,
  depthTest: false,
});
const foodGeom = new THREE.SphereGeometry(11, 24, 16);
const foodGroup = new THREE.Group();
foodGroup.renderOrder = 10;
scene.add(foodGroup);

function setFood(items) {
  wordFoodMap.clear();
  while (foodGroup.children.length) foodGroup.remove(foodGroup.children[0]);
  for (const item of items) {
    if (item.word) {
      // Hamlet word food — tracked for text canvas
      wordFoodMap.set(`${item.line_id}_${item.word_idx}`, item);
    } else {
      // Manual click food — yellow 3D sphere
      const m = new THREE.Mesh(foodGeom, foodMaterial);
      m.position.set(item.x || item[0], item.y || item[1], 1);
      m.renderOrder = 10;
      foodGroup.add(m);
    }
  }
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
// Extract (flask, worm) from URL path. Both shapes are accepted:
//   /focus/<flask>/<worm>   ← multi-flask mode
//   /focus/<worm>           ← legacy single-group mode, flask='default'
const _pathParts = location.pathname.replace(/^\/focus\//, '').replace(/\/$/, '').split('/');
const FLASK_NAME = _pathParts.length >= 2 ? decodeURIComponent(_pathParts[0]) : 'default';
const WORM_NAME = decodeURIComponent(_pathParts[_pathParts.length - 1]);
const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/focus/${encodeURIComponent(FLASK_NAME)}/${encodeURIComponent(WORM_NAME)}`;

// Debug door: clients that have set localStorage.wormletDebugToken get to
// poke the sim (add food, pause, etc.). Public visitors don't, so their
// click handlers silently no-op.
function debugToken() { return localStorage.getItem('wormletDebugToken'); }
async function debugPost(path, body) {
  const tok = debugToken();
  if (!tok) return;
  try {
    await fetch(`/debug/${encodeURIComponent(WORM_NAME)}/${path}`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${tok}`, 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : '{}',
    });
  } catch (_e) {}
}
let ws = null;
function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => { hud.textContent = 'connected'; };
  ws.onclose = () => { hud.textContent = 'disconnected · retrying…'; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type !== 'state') return;
    setMidline(msg.midline);
    setFood(msg.food);
    smellsData = msg.smells || [];
    latestResidual = msg.residual || { pca: new Array(12).fill(0), words: [] };
    const wordImpact = computeWordImpact();
    // Each panel's own draw fn short-circuits when hidden, but we also guard
    // here so once-per-frame work (string building, etc.) doesn't fire when
    // the panel is hidden via the dock.
    if (isChemoVisible()) drawChemoPanel({ corpusPca, contributions: wordImpact.contributions });
    if (isRadarVisible()) drawRadar({ corpusPca, computeWordImpact });
    wormHeadPos = { x: msg.head[0], y: msg.head[1] };
    neuronActivity = msg.neurons || {};
    stimFlags = msg.stim;
    const stims = Object.entries(msg.stim).filter(([, v]) => v).map(([k]) => k).join(',');

    isPaused = msg.paused || false;
    window.__lastPaused = isPaused;
    const pausedStr = isPaused ? ' <span style="color:#fc6;">[PAUSED]</span>' : '';

    // Row 1: tight, just speed and motor.
    const row1 =
      `speed=${msg.speed.toFixed(2)}` +
      `  motor L=${msg.motor.L.toFixed(1)} R=${msg.motor.R.toFixed(1)}` +
      pausedStr;

    // Row 2: top words affecting the worm right now. "Affecting" =
    // total chemosensory contribution (sum of L+R activations across
    // all 12 pairs) for in-range smells, plus the per-PC residual ×
    // word-decay for recently eaten words.
    // (wordImpact computed once above for both drawChemoPanel + this HUD row.)
    let row2 = `<span style="opacity:0.55;">smelling:</span> `;
    if (wordImpact.entries.length === 0) {
      row2 += `<span style="opacity:0.45;">— nothing in range —</span>`;
    } else {
      const total = wordImpact.total || 1;
      const top = wordImpact.entries.slice(0, 5);
      row2 += top.map(e => {
        const pct = (e.value / total * 100) | 0;
        const color = e.eaten ? '#fc8' : '#cfc';
        const note = e.eaten ? `<span style="opacity:0.6;">·eaten</span>` : '';
        return `<span style="color:${color};">${escapeHTML(e.word)} ${pct}%${note}</span>`;
      }).join(' <span style="opacity:0.3;">·</span> ');
      row2 += ` <span style="opacity:0.4; font-size:10px;">[<span style="color:#cfc;">smelled</span> <span style="color:#fc8;">eaten</span>]</span>`;
    }

    hud.innerHTML = row1 + '<br>' + row2;
  };
}
connect();

function bar(percent, hue, w = 70) {
  const v = Math.max(0, Math.min(100, percent));
  return `<span style="display:inline-block; width:${w}px; height:8px; background:rgba(255,255,255,0.08); border-radius:2px; vertical-align:middle; overflow:hidden;">
    <span style="display:block; width:${v}%; height:100%; background:hsl(${hue},75%,55%); box-shadow:0 0 4px hsl(${hue},90%,60%);"></span>
  </span>`;
}

// Each PC gets a stable color hue spaced around the wheel so the eye can
// learn "blue = PC3" type associations over time.
function pcHue(i) { return Math.round((i * 360) / 12); }

// wordHue (used by chemo-panel's stacked bars) now lives in
// ./chemo-panel.js — index.js no longer needs it.

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// Compute per-(word, neuron-pair, side) contributions from current sensory
// state. Used by both the HUD (Item 1 ranked-words row) and the
// chemosensory panel (Item 2 stacked bars).
//
// Returns:
//   {
//     contributions: { neuron_name: [{word, value, eaten}, ...] },
//     entries: [{word, value, eaten}, ...]    sorted desc by total impact
//     total: <sum of all entries.value>
//   }
function computeWordImpact() {
  if (!corpusPca) return { contributions: {}, entries: [], total: 0 };
  const pairs = corpusPca.pc_neuron_pairs;

  // contributions[neuron] = list of {word, value, eaten}
  const contributions = {};
  for (const [L, R] of pairs) { contributions[L] = []; contributions[R] = []; }

  // 1) In-range smells: per-word PCA × distance_factor × direction-aware split.
  for (const smell of smellsData) {
    const dir = smell.direction_factor;
    const dirL = dir !== undefined ? dir : 0.5;
    const dirR = 1 - dirL;
    const distF = smell.distance_factor ?? 0;
    const pca = smell.pca || [];
    for (let i = 0; i < pairs.length; i++) {
      const [L, R] = pairs[i];
      const v = (pca[i] ?? 0) * distF;
      const cL = v * dirL;
      const cR = v * dirR;
      if (cL > 0) contributions[L].push({ word: smell.word, value: cL, eaten: false });
      if (cR > 0) contributions[R].push({ word: smell.word, value: cR, eaten: false });
    }
  }

  // 2) Eaten residual: each previously-eaten word contributes to both L and R
  // equally, weighted by its current decay factor. MUST use the same
  // sparse PCA the sim uses (pca12_sparse), otherwise eaten words would
  // saturate the bars at full raw-PCA strength while smelled words run
  // through the softmaxed values.
  if (latestResidual && latestResidual.words && corpusPca.words) {
    const idx = corpusPca._wordIdx ||
      (corpusPca._wordIdx = Object.fromEntries(corpusPca.words.map((w, i) => [w, i])));
    const pcaTable = corpusPca.pca12_sparse || corpusPca.pca12;
    for (const { word, decay } of latestResidual.words) {
      const k = word.toLowerCase().replace(/^'+|'+$/g, '');
      const wi = idx[k];
      if (wi === undefined) continue;
      const pca = pcaTable[wi];
      for (let i = 0; i < pairs.length; i++) {
        const v = (pca[i] ?? 0) * decay * 0.5;  // 0.5 to mirror sim's L/R equal split
        if (v <= 0) continue;
        const [L, R] = pairs[i];
        contributions[L].push({ word, value: v, eaten: true });
        contributions[R].push({ word, value: v, eaten: true });
      }
    }
  }

  // Aggregate per-word total impact (used by HUD row 2).
  const byWord = {};
  for (const list of Object.values(contributions)) {
    for (const c of list) {
      if (!byWord[c.word]) byWord[c.word] = { word: c.word, value: 0, eaten: c.eaten };
      byWord[c.word].value += c.value;
      // If a word appears as both sensed and eaten, prefer "sensed" labelling.
      if (!c.eaten) byWord[c.word].eaten = false;
    }
  }
  const entries = Object.values(byWord).sort((a, b) => b.value - a.value);
  const total = entries.reduce((s, e) => s + e.value, 0);
  return { contributions, entries, total };
}

// stackedBar (and its wordHue helper) now live in ./chemo-panel.js — they
// were only used by the chemosensory panel's bar rendering.

// Emotion radar panel now lives in ./radar-panel.js as drawRadar.
// updateChemosensoryPanel now lives in ./chemo-panel.js as drawChemoPanel.

// ---------------------------------------------------------------------------
// Text rendering functions live in ./text-canvas.js; the drifting Hamlet
// words, smell lines, hover PCA popup, and world→screen projection are all
// there. Hex-bin computation for the PCA popup remains here because it's
// invoked once during corpus-load (next block) rather than each frame.
// ---------------------------------------------------------------------------

// Corpus embeddings: every Hamlet word's 12-d chemosensory vector (still PCA,
// drives the sim) + a 2-d projection for the hover scatter (now UMAP, since
// UMAP preserves local semantic structure better in 2D). Fetched once on
// load; the sim will swap to UMAP-12 when generational evolution starts.
let corpusPca = null;     // {words, pca12, pca2, pc_neuron_pairs, ...}
let pcaData = null;       // legacy shape kept for the hover popup: {tokens, pca, token_to_idx}
async function initCorpusEmbeddings() {
  // Fetch the PCA artifact first for its scaffolding (pc_neuron_pairs,
  // emotions, words). If a UMAP artifact is also available AND the sim is
  // running on UMAP, overlay the UMAP dim-reduction outputs into the same
  // fields so downstream code stays unchanged.
  try {
    const data = await (await fetch('/api/corpus_pca')).json();
    corpusPca = data;
    corpusPca.embeddingName = 'PCA';
    const idx = {};
    for (let i = 0; i < data.words.length; i++) idx[data.words[i]] = i;
    pcaData = { tokens: data.words, pca: data.pca2, token_to_idx: idx, projection: 'pca' };
  } catch (e) {
    console.warn('corpus PCA not available', e);
    return;
  }
  let simEmbedding = 'pca';
  try {
    const hz = await (await fetch('/healthz')).json();
    simEmbedding = (hz.embedding || 'pca').toLowerCase();
  } catch (_e) { /* fall back to pca */ }
  try {
    const umap = await (await fetch('/api/corpus_umap')).json();
    // Word order in the UMAP cache matches the PCA cache (same dedup pipeline).
    pcaData.pca = umap.umap2;
    pcaData.projection = 'umap';
    if (simEmbedding === 'umap') {
      // Swap the 12-d chemosensory bars to read UMAP values too, so the
      // viewer matches what the sim is actually firing.
      corpusPca.pca12 = umap.umap12;
      corpusPca.pca12_sparse = umap.umap12_sparse;
      corpusPca.embeddingName = 'UMAP';
    }
  } catch (e) {
    console.warn('corpus UMAP not available; hover scatter falling back to PCA-2D', e);
  }
  // Precompute hex bins for the hover-popup density plot. Done once here so
  // we don't recompute 4500+ nearest-hex lookups every frame.
  computeHexBins(pcaData);
}
initCorpusEmbeddings();

// Hex-binning of the projected coords. Pointy-top hexagons tiled across
// [0,1] × [0,1]. Each word is binned to its nearest hex center; the popup
// renders each hex with grayscale intensity proportional to log(count+1),
// so white = densest, black = empty. We use log to keep mid-density hexes
// visible against the few super-dense cluster cores.
const HEX_NCOLS = 18;
function computeHexBins(p) {
  if (!p || !p.pca || !p.pca.length) return;
  const rUnit = 1 / (HEX_NCOLS * Math.sqrt(3));  // hex "radius" (center to vertex)
  const hSpacing = rUnit * Math.sqrt(3);          // horizontal step between centers
  const vSpacing = rUnit * 1.5;                   // vertical step (rows interleave x-offset)
  const nCols = HEX_NCOLS + 2;                    // padding so the plot edges are covered
  const nRows = Math.ceil(1 / vSpacing) + 2;
  const centers = [];
  for (let row = 0; row < nRows; row++) {
    const yc = row * vSpacing;
    const xOffset = (row % 2) ? hSpacing / 2 : 0;
    for (let col = 0; col < nCols; col++) {
      centers.push([col * hSpacing + xOffset, yc]);
    }
  }
  const counts = new Array(centers.length).fill(0);
  const { pca } = p;
  for (let i = 0; i < pca.length; i++) {
    const x = pca[i][0], y = pca[i][1];
    let bestIdx = -1, bestD2 = Infinity;
    for (let j = 0; j < centers.length; j++) {
      const dx = centers[j][0] - x;
      const dy = centers[j][1] - y;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) { bestD2 = d2; bestIdx = j; }
    }
    if (bestIdx >= 0) counts[bestIdx]++;
  }
  let maxCount = 0;
  for (const c of counts) if (c > maxCount) maxCount = c;
  p.hexBins = { centers, counts, maxCount, radius: rUnit };
}

// drawTextCanvas / drawPcaPopup / drawSmells now live in ./text-canvas.js.

// ---------------------------------------------------------------------------
// Network panel (#netcanvas) — owned by ./network-panel.js. It manages
// the canvas DOM ref + DPR + the x-ray/legacy-graph dispatch. This file
// just feeds it live state each frame and forwards keypresses.
// ---------------------------------------------------------------------------

// Text canvas (textcanvas/tctx/resize handler) lives in ./text-canvas.js;
// the mousemove handler below still needs `textcanvas` for cursor → world
// coordinate translation.

// Track mouse position in screen coordinates
let mouseScreenPos = { x: 0, y: 0 };

let graph = null;
let neuronActivity = {};
let stimFlags = { hunger: false, nose_touch: false, food_sense: false };

// Per-neuron 2D body-plan coords: {axial: [0,1], lateral: [-1,+1], dv: [-1,+1]}
// Fetched once on load; static.
let neuronBodyCoords = null;
(async () => {
  try {
    const data = await (await fetch('/api/neuron_body_coords')).json();
    neuronBodyCoords = data;
  } catch (e) {
    console.warn('neuron body coords not available', e);
  }
})();

// Text food and word embedding data
let wordFoodMap = new Map();  // key: `${line_id}_${word_idx}` → {x, y, word}
// pcaData declared above via initCorpusPca()
// nearestWord (the cursor-nearest drifting word) lives in ./text-canvas.js.
let wormHeadPos = { x: 800, y: 500 };  // updated from snapshot
let mouseWorldPos = { x: WORLD_W / 2, y: WORLD_H / 2 };  // mouse position in world coords

// Smell visualization
let smellsData = [];          // list of sensed smells from snapshot
let latestResidual = { pca: new Array(12).fill(0), words: [] };
let smellsVisible = true;     // toggle with 'o' key

// X-ray neuron-label visibility ('l' key). Single source of truth — the
// network panel and the magnifier are both notified on every toggle. This
// replaces the network panel's internal flag as authoritative state.
let xrayLabelsVisible = false;

// Simulation state
let isPaused = false;         // toggled with spacebar

// Chemosensory panel state + DOM ref live in ./chemo-panel.js.
// Emotion radar panel state + DOM ref live in ./radar-panel.js.

// Track mouse movement
document.addEventListener('mousemove', (ev) => {
  const rect = textcanvas.getBoundingClientRect();
  const screenX = ev.clientX - rect.left;
  const screenY = ev.clientY - rect.top;
  mouseScreenPos.x = screenX;
  mouseScreenPos.y = screenY;
  // Transform screen coords to world coords. Use CSS-pixel viewport dims
  // because `textcanvas.width` is now DPR-scaled (physical pixels) after the
  // text-canvas DPR fix, while `screenX/Y` from clientX/Y are CSS pixels.
  const normX = screenX / window.innerWidth;
  const normY = screenY / window.innerHeight;
  mouseWorldPos.x = camera.left + normX * (camera.right - camera.left);
  mouseWorldPos.y = camera.top + normY * (camera.bottom - camera.top);
});

window.addEventListener('keydown', ev => {
  if (ev.key === 'n' || ev.key === 'N') {
    toggleNetVisible();
  }
  if (ev.key === 'x' || ev.key === 'X') {
    toggleXrayMode();
  }
  if (ev.key === 'l' || ev.key === 'L') {
    // Toggle our local copy and broadcast to both surfaces so the network
    // panel and the magnifier stay in sync. Using a local flag here (rather
    // than reading network-panel's getXrayLabelsVisible after a toggle)
    // means the flow stays one-directional: one source of truth → two
    // consumers.
    xrayLabelsVisible = !xrayLabelsVisible;
    setNetworkXrayLabelsVisible(xrayLabelsVisible);
    magnifier.setXrayLabelsVisible(xrayLabelsVisible);
  }
  if (ev.key === 'm' || ev.key === 'M') {
    toggleMotorLabels();
  }
  if (ev.key === 'o' || ev.key === 'O') {
    smellsVisible = !smellsVisible;
  }
  if (ev.key === 'c' || ev.key === 'C') {
    toggleChemo();
  }
  if (ev.key === 'e' || ev.key === 'E') {
    toggleRadar();
    // Paint immediately on open, don't wait for the next WS frame.
    if (isRadarVisible()) drawRadar({ corpusPca, computeWordImpact });
  }
  if (ev.key === ' ') {
    ev.preventDefault();
    // Debug-only: toggle pause via HTTP. Public visitors silently no-op.
    debugPost('pause', { paused: !window.__lastPaused });
  }
});

async function initGraph() {
  const data = await (await fetch('/api/graph')).json();
  const N = data.neurons.length;
  const pos = buildPositions(data.neurons, data.positions);
  const adjOut = Array.from({length: N}, () => []);
  for (const [pi, qi] of data.edges) adjOut[pi].push(qi);

  graph = {
    neurons:          data.neurons,
    fireThreshold:    data.fire_threshold,
    muscleSet:        new Set(data.muscle_indices),
    sensorySet:       new Set(data.sensory_indices),
    chemosensorySet:  new Set(data.chemosensory_indices),
    motorSet:         new Set(data.motor_indices),
    foodSet:          new Set(data.food_indices),
    noseSet:          new Set(data.nose_indices),
    hungerSet:        new Set(data.hunger_indices),
    pos, adjOut, N,
  };
}
initGraph();

// X-ray and legacy graph rendering live in ./xray-render.js and
// ./network-panel.js respectively. The render loop below dispatches via
// `drawNetworkPanel(state)`; keyboard toggles ('n','x','l','m') are
// forwarded from the keydown handler above.

// ---------------------------------------------------------------------------
// Click → world coords → drop food
// ---------------------------------------------------------------------------
function eventToWorld(ev) {
  const r = canvas.getBoundingClientRect();
  const nx = (ev.clientX - r.left) / r.width;
  const ny = (ev.clientY - r.top) / r.height;
  const x = camera.left + nx * (camera.right - camera.left);
  const y = camera.top + ny * (camera.bottom - camera.top);
  return [x, y];
}
canvas.addEventListener('mousedown', (ev) => {
  // Debug-only: drop food at click position. Public visitors silently no-op.
  if (!debugToken()) return;
  if (ev.shiftKey) {
    // Clear food not exposed as a debug route — would need reset; skip.
    return;
  }
  const [x, y] = eventToWorld(ev);
  debugPost('add_food', { x, y });
});

// ---------------------------------------------------------------------------
// Debug handles + render loop
// ---------------------------------------------------------------------------
window.__sim = {
  THREE, scene, camera, renderer, composer, bloom,
  foodGroup, bodyMaterial,
  get wormMesh() { return getWormMesh(); },
  get graph() { return graph; },
  get neuronActivity() { return neuronActivity; },
  get motorLabelsVisible() { return getMotorLabelsVisible(); },
};

const clock = new THREE.Clock();
function render() {
  requestAnimationFrame(render);
  const t = clock.getElapsedTime();
  bodyMaterial.uniforms.uTime.value = t;
  organMaterial.uniforms.uTime.value = t;
  composer.render();
  if (isNetVisible()) {
    drawNetworkPanel({ neuronBodyCoords, graph, neuronActivity, stimFlags });
  }
  // Magnifier runs after the network panel so they share the same per-frame
  // sim state. It internally no-ops when hidden — cheap to call every frame.
  magnifier.setState({ neuronBodyCoords, graph, neuronActivity });
  magnifier.render();
  drawTextCanvas({
    wordFoodMap,
    mouseWorldPos,
    mouseScreenPos,
    smellsData,
    smellsVisible,
    wormHeadPos,
    pcaData,
  });
}
render();

// --- Generation rollover overlay ---
// Same banner as the overview page; shows while the sim is frozen for
// end-of-generation processing (LLM scoring, NES update, git commit).
(function setupGenOverlay() {
  const style = document.createElement('style');
  style.textContent = `
    #gen-overlay {
      position: fixed; inset: 0; display: none;
      background: rgba(0, 0, 0, 0.78); z-index: 9999;
      align-items: center; justify-content: center;
      font: 13px ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #6f9;
    }
    #gen-overlay.visible { display: flex; }
    #gen-overlay .panel {
      min-width: 360px; max-width: 560px; padding: 22px 28px;
      background: rgba(0, 20, 10, 0.92);
      border: 1px solid rgba(100, 255, 200, 0.4);
      border-radius: 6px;
      box-shadow: 0 8px 40px rgba(0, 0, 0, 0.6);
    }
    #gen-overlay h2 { margin: 0 0 14px 0; font-size: 15px; color: #6f9; }
    #gen-overlay .phase { margin-bottom: 12px; color: #cfd; font-size: 13px; }
    #gen-overlay .bar { height: 6px; background: rgba(100, 255, 200, 0.12); border-radius: 3px; overflow: hidden; margin-bottom: 8px; }
    #gen-overlay .bar > div { height: 100%; background: #6f9; transition: width 250ms ease; }
    #gen-overlay .meta { color: #5a5; font-size: 11px; margin-top: 8px; }
    #gen-overlay .err { color: #f88; margin-top: 10px; font-size: 12px; }
  `;
  document.head.appendChild(style);
  const overlay = document.createElement('div');
  overlay.id = 'gen-overlay';
  overlay.innerHTML =
    `<div class="panel"><h2>generation rollover</h2>` +
    `<div class="phase">…</div>` +
    `<div class="bar"><div style="width:0%"></div></div>` +
    `<div class="meta"></div><div class="err"></div></div>`;
  document.body.appendChild(overlay);
  const $phase = overlay.querySelector('.phase');
  const $bar = overlay.querySelector('.bar > div');
  const $meta = overlay.querySelector('.meta');
  const $err = overlay.querySelector('.err');
  const PHASE_LABELS = {
    running: 'simulation running',
    corpus_draining: 'last words drifting off-screen…',
    judging: 'LLM is reading the poems',
    evolving: 'computing NES gradient',
    committing: 'committing generation to git',
    respawning: 'spawning next generation',
  };
  async function tick() {
    try {
      const res = await fetch('/api/generation_status');
      if (!res.ok) return;
      const s = await res.json();
      const active = s.enabled && s.phase && s.phase !== 'running';
      overlay.classList.toggle('visible', active);
      if (!active) return;
      $phase.textContent = PHASE_LABELS[s.phase] || s.phase;
      const pct = s.worms_total > 0 ? Math.round(100 * s.worms_done / s.worms_total) : 0;
      $bar.style.width = (s.phase === 'judging' ? pct : 100) + '%';
      $meta.textContent =
        `generation ${s.generation} · group ${s.group || '—'} · ` +
        `${s.worms_done}/${s.worms_total} worms scored · ${s.elapsed_s}s elapsed`;
      $err.textContent = s.error || '';
    } catch (_e) {}
  }
  setInterval(tick, 500);
  tick();
})();
