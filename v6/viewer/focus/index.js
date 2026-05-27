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
  toggleXrayLabels,
  toggleMotorLabels,
  getMotorLabelsVisible,
} from './network-panel.js';

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
    updateChemosensoryPanel();
    updateRadarPanel();
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
    const wordImpact = computeWordImpact();
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

// Hash-based color per word so the same word draws the same color in every
// bar segment it appears in, every frame. Stable across reloads.
function wordHue(word) {
  let h = 0;
  const s = word.toLowerCase();
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return ((h % 360) + 360) % 360;
}

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

// Render a single horizontal bar composed of stacked colored segments,
// one per contributing word. Segments are sorted ALPHABETICALLY (stable
// positions across frames), colored by per-word hash hue, and labeled
// inline if the segment is wide enough. Eaten words get a dashed border
// so you can see at a glance which contributions are residual.
//
// `entries` is a list of {word, value, eaten}. `barMax` is the bar's
// pixel width. `totalScale` is the value that should fill the full bar
// (we use 1.0 as "saturated", same as the sim's per-neuron cap).
function stackedBar(entries, barMax, totalScale = 1.0) {
  const sorted = entries.slice().sort((a, b) =>
    a.word.toLowerCase().localeCompare(b.word.toLowerCase())
  );
  const total = sorted.reduce((s, e) => s + e.value, 0);
  const fillFrac = Math.max(0, Math.min(1, total / totalScale));
  const fillPx = fillFrac * barMax;

  let html = `<span style="display:inline-block; position:relative; width:${barMax}px; height:14px; background:rgba(255,255,255,0.07); border-radius:2px; vertical-align:middle; overflow:hidden;">`;

  if (total > 0) {
    let x = 0;
    for (const e of sorted) {
      const segPx = (e.value / total) * fillPx;
      const hue = wordHue(e.word);
      const bg = `hsl(${hue},75%,55%)`;
      const borderStyle = e.eaten ? `outline:1px dashed hsl(${hue},85%,80%); outline-offset:-1px;` : '';
      const labelFits = segPx > 28;  // ~3 chars of monospace at 9px
      const label = labelFits
        ? `<span style="position:absolute; left:3px; top:1px; color:#000a; font-size:9px; mix-blend-mode:luminosity; pointer-events:none; max-width:${segPx - 4}px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHTML(e.word)}</span>`
        : '';
      html += `<span title="${escapeHTML(e.word)}${e.eaten ? ' · eaten' : ''}: ${(e.value*100|0)}%" style="position:absolute; left:${x}px; top:0; width:${segPx}px; height:100%; background:${bg}; box-shadow:0 0 4px hsl(${hue},90%,60%); ${borderStyle}">${label}</span>`;
      x += segPx;
    }
  }
  html += `</span>`;
  return html;
}

// -------- Emotion radar panel (Item 3) --------
// 24 small radar charts, one per chemosensory neuron. For each neuron, the
// 8 NRC emotions are axes; the polygon shows the contribution-weighted
// emotional shape of words currently driving the neuron. Useful as an
// interpretability lens onto what each PC has "absorbed" from the corpus.

const EMOTION_AXIS_COLORS = {
  joy: '#fc6', trust: '#6f9', anticipation: '#9cf', surprise: '#cfc',
  fear: '#c6f', disgust: '#9c6', sadness: '#69c', anger: '#f66',
};

function _radarLayout() {
  // 4 cols × 3 rows = 12 charts, one per PC / neuron pair. (L and R of the
  // same pair have identical emotion content — only their magnitudes
  // differ — so 24 separate charts were redundant.)
  const COLS = 4, ROWS = 3;
  const W = radarcanvas.width / (window.devicePixelRatio || 1);
  const H = radarcanvas.height / (window.devicePixelRatio || 1);
  const padTop = 28, padBottom = 6, padX = 6;
  const cellW = (W - padX * 2) / COLS;
  const cellH = (H - padTop - padBottom) / ROWS;
  return { COLS, ROWS, W, H, padTop, padX, cellW, cellH };
}

function updateRadarPanel() {
  if (!radarVisible || !corpusPca) return;
  if (!corpusPca.emotion_keys || !corpusPca.emotions) {
    // Corpus PCA cache predates the emotion fields — rebuild needed.
    radarctx.clearRect(0, 0, radarcanvas.width, radarcanvas.height);
    radarctx.fillStyle = '#fc6';
    radarctx.font = '11px ui-monospace, monospace';
    radarctx.fillText('rebuild corpus_pca.json with emotions', 10, 20);
    return;
  }
  const emotionKeys = corpusPca.emotion_keys;
  const wordIdx = corpusPca._wordIdx ||
    (corpusPca._wordIdx = Object.fromEntries(corpusPca.words.map((w, i) => [w, i])));
  const { contributions } = computeWordImpact();

  // For each PC / neuron pair, aggregate emotion-weighted contributions
  // across BOTH L and R members. (Same words contribute to both with
  // different magnitudes; we want the pair-level emotional shape.)
  const pairs = corpusPca.pc_neuron_pairs;
  const weightedByPair = [];
  let globalMax = 0;
  for (let i = 0; i < pairs.length; i++) {
    const [L, R] = pairs[i];
    const w = new Array(emotionKeys.length).fill(0);
    const merged = [...(contributions[L] || []), ...(contributions[R] || [])];
    for (const c of merged) {
      const k = c.word.toLowerCase().replace(/^'+|'+$/g, '');
      const wi = wordIdx[k];
      if (wi === undefined) continue;
      const ev = corpusPca.emotions[wi];
      for (let e = 0; e < emotionKeys.length; e++) w[e] += c.value * ev[e];
    }
    weightedByPair.push(w);
    for (const v of w) if (v > globalMax) globalMax = v;
  }
  const hasEmotionalSignal = globalMax > 0;
  if (!hasEmotionalSignal) globalMax = 1;

  // Render
  const ctx = radarctx;
  const { COLS, ROWS, W, H, padTop, padX, cellW, cellH } = _radarLayout();
  ctx.clearRect(0, 0, W, H);

  // Title
  ctx.fillStyle = '#8f8';
  ctx.font = 'bold 11px ui-monospace, monospace';
  ctx.textBaseline = 'top';
  ctx.fillText('● EMOTION RADAR — what each PC is absorbing', 8, 4);
  ctx.fillStyle = '#9c9';
  ctx.font = '9px ui-monospace, monospace';
  if (hasEmotionalSignal) {
    ctx.fillText('contribution-weighted NRC emotions per PC pair', 8, 16);
  } else {
    ctx.fillStyle = '#fc6';
    ctx.fillText('no words with NRC emotional content nearby', 8, 16);
  }

  const axes = emotionKeys.length;
  for (let idx = 0; idx < pairs.length; idx++) {
    const [L] = pairs[idx];
    const pairBase = L.slice(0, -1);
    const col = idx % COLS;
    const row = (idx / COLS) | 0;
    const cx = padX + col * cellW + cellW / 2;
    const cy = padTop + row * cellH + cellH / 2;
    const radius = Math.min(cellW, cellH) * 0.32;

    // Background rings
    ctx.strokeStyle = 'rgba(150,200,255,0.10)';
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = 'rgba(150,200,255,0.07)';
    ctx.beginPath(); ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2); ctx.stroke();

    // Axis spokes + (on top-right cell only) emotion abbreviations
    ctx.strokeStyle = 'rgba(150,200,255,0.10)';
    ctx.beginPath();
    for (let a = 0; a < axes; a++) {
      const ang = -Math.PI / 2 + (a / axes) * Math.PI * 2;
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(ang) * radius, cy + Math.sin(ang) * radius);
    }
    ctx.stroke();

    // Polygon
    const w = weightedByPair[idx];
    const hue = pcHue(idx);
    ctx.beginPath();
    for (let a = 0; a < axes; a++) {
      const ang = -Math.PI / 2 + (a / axes) * Math.PI * 2;
      const r = (w[a] / globalMax) * radius;
      const x = cx + Math.cos(ang) * r;
      const y = cy + Math.sin(ang) * r;
      if (a === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = `hsla(${hue}, 75%, 60%, 0.32)`;
    ctx.fill();
    ctx.strokeStyle = `hsl(${hue}, 80%, 70%)`;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // Emotion dots at non-trivial axes (helps read the polygon shape)
    for (let a = 0; a < axes; a++) {
      const r = (w[a] / globalMax) * radius;
      if (r < 1) continue;
      const ang = -Math.PI / 2 + (a / axes) * Math.PI * 2;
      ctx.fillStyle = EMOTION_AXIS_COLORS[emotionKeys[a]] || '#fff';
      ctx.beginPath();
      ctx.arc(cx + Math.cos(ang) * r, cy + Math.sin(ang) * r, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Label below
    ctx.fillStyle = `hsl(${hue}, 55%, 78%)`;
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`PC${idx} · ${pairBase}`, cx, cy + radius + 2);

    // Axis legend only on the top-right cell.
    if (idx === COLS - 1) {
      ctx.textAlign = 'left';
      ctx.font = '8px ui-monospace, monospace';
      for (let a = 0; a < axes; a++) {
        const ang = -Math.PI / 2 + (a / axes) * Math.PI * 2;
        const lx = cx + Math.cos(ang) * (radius + 3);
        const ly = cy + Math.sin(ang) * (radius + 3);
        ctx.fillStyle = EMOTION_AXIS_COLORS[emotionKeys[a]] || '#fff';
        ctx.fillText(emotionKeys[a].slice(0, 3), lx, ly);
      }
    }
  }
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
}

function updateChemosensoryPanel() {
  if (!chemosensoryVisible) return;
  if (!corpusPca) {
    chemosensoryPanel.innerHTML = `<div style="opacity:0.4; padding:6px 0; font-size:10px;">loading corpus PCA…</div>`;
    return;
  }

  const pairs = corpusPca.pc_neuron_pairs;
  const { contributions } = computeWordImpact();

  // Header
  const anyFiring = pairs.some(([L, R]) =>
    (contributions[L] && contributions[L].length) || (contributions[R] && contributions[R].length)
  );
  let html = `<div style="font-weight:bold; margin-bottom:8px; color:${anyFiring ? '#8f8' : '#6f6'}; opacity:${anyFiring ? 1 : 0.35};">● CHEMOSENSORY STATE (${corpusPca.embeddingName || 'PCA'})</div>`;

  // Column headers
  const labelW = 60, barW = 110;
  html += `<div style="display:grid; grid-template-columns:${labelW}px ${barW}px ${barW}px; gap:6px; align-items:center; font-size:9px; opacity:0.45; margin-bottom:3px;">
    <span>pair</span><span style="text-align:center;">L</span><span style="text-align:center;">R</span>
  </div>`;

  // 12 rows — one per neuron pair.
  for (let i = 0; i < pairs.length; i++) {
    const [L, R] = pairs[i];
    const Lcontribs = contributions[L] || [];
    const Rcontribs = contributions[R] || [];
    const pairBase = L.slice(0, -1);
    const pcHueValue = pcHue(i);
    html += `<div style="display:grid; grid-template-columns:${labelW}px ${barW}px ${barW}px; gap:6px; align-items:center; margin:2px 0;">
      <span style="font-size:9px; color:hsl(${pcHueValue},55%,72%); opacity:0.85;">PC${i} · ${pairBase}</span>
      ${stackedBar(Lcontribs, barW)}
      ${stackedBar(Rcontribs, barW)}
    </div>`;
  }

  // Footer legend
  html += `<div style="margin-top:10px; font-size:9px; opacity:0.5; line-height:1.5;">
    bars sum per-word contributions · colors are stable per word ·
    <span style="display:inline-block; width:8px; height:8px; background:#6f9; vertical-align:middle;"></span> sensed ·
    <span style="display:inline-block; width:8px; height:8px; outline:1px dashed #ccc; outline-offset:-1px; vertical-align:middle;"></span> eaten (residual)
  </div>`;

  chemosensoryPanel.innerHTML = html;
}

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

// Simulation state
let isPaused = false;         // toggled with spacebar

// Chemosensory panel
let chemosensoryVisible = true;  // toggle with 'c' key
const chemosensoryPanel = document.getElementById('chemosensoryPanel');

let radarVisible = false;        // toggle with 'e' key
const radarcanvas = document.getElementById('radarcanvas');
const radarctx = radarcanvas.getContext('2d');
// Hardcode the canvas buffer size to match the CSS box (the element is
// display:none at load, so getBoundingClientRect returns 0×0 — using it
// would size the buffer to 1×1 and make all drawing invisible).
const RADAR_W = 460, RADAR_H = 280;
{
  const _dpr = Math.min(window.devicePixelRatio || 1, 2);
  radarcanvas.width = RADAR_W * _dpr;
  radarcanvas.height = RADAR_H * _dpr;
  radarctx.scale(_dpr, _dpr);
}

// Neuron type to emotion mapping (for display)
const neuronEmotionMap = {
  'ASEL': 'salt/attract (L)',
  'ASER': 'salt/repel (R)',
  'AWAL': 'food-odor (L)',
  'AWAR': 'food-odor (R)',
  'AWBL': 'approach (L)',
  'AWBR': 'approach (R)',
  'AWCL': 'CO2/safety (L)',
  'AWCR': 'CO2/safety (R)',
  'ASIL': 'hunger/arousal (L)',
  'ASIR': 'hunger/arousal (R)',
  'ASJL': 'taste/novel (L)',
  'ASJR': 'taste/novel (R)',
  'ASHL': 'pain/avoid (L)',
  'ASHR': 'pain/avoid (R)',
  'ASKL': 'protect (L)',
  'ASKR': 'protect (R)',
  'ASGL': 'integrate (L)',
  'ASGR': 'integrate (R)',
  'ADFL': 'food-chemo (L)',
  'ADFR': 'food-chemo (R)',
  'ADLL': 'polymodal (L)',
  'ADLR': 'polymodal (R)',
};

// Track mouse movement
document.addEventListener('mousemove', (ev) => {
  const rect = textcanvas.getBoundingClientRect();
  const screenX = ev.clientX - rect.left;
  const screenY = ev.clientY - rect.top;
  mouseScreenPos.x = screenX;
  mouseScreenPos.y = screenY;
  // Transform screen coords to world coords
  const normX = screenX / textcanvas.width;
  const normY = screenY / textcanvas.height;
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
    toggleXrayLabels();
  }
  if (ev.key === 'm' || ev.key === 'M') {
    toggleMotorLabels();
  }
  if (ev.key === 'o' || ev.key === 'O') {
    smellsVisible = !smellsVisible;
  }
  if (ev.key === 'c' || ev.key === 'C') {
    chemosensoryVisible = !chemosensoryVisible;
    chemosensoryPanel.style.display = chemosensoryVisible ? 'block' : 'none';
  }
  if (ev.key === 'e' || ev.key === 'E') {
    radarVisible = !radarVisible;
    radarcanvas.style.display = radarVisible ? 'block' : 'none';
    if (radarVisible) updateRadarPanel();  // paint immediately, don't wait for next WS frame
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
