// ---------------------------------------------------------------------------
// Emotion radar panel — fixed-position canvas overlay (#radarcanvas).
// ---------------------------------------------------------------------------
// 12 small radar charts, one per PC / chemosensory-neuron pair. For each
// pair, the 8 NRC emotion axes show the contribution-weighted emotional
// shape of words currently driving that PC. Useful as an interpretability
// lens onto what each PC has "absorbed" from the corpus.
//
// Toggle:
//   'e' — show/hide the radar panel
//
// Live app state (corpusPca + computed contributions) is owned by index.js
// and passed in each call via the `state` parameter; the panel only owns the
// DOM ref and the visibility flag. The expensive computeWordImpact() call
// stays inside drawRadar so it's short-circuited by the visibility check —
// hiding the radar must not pay its cost each WebSocket frame.

import * as chrome from './panel-chrome.js';

// ---------------------------------------------------------------------------
// DOM ref + DPR scaling setup
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// View-mode flag (module-local)
// ---------------------------------------------------------------------------
// radarVisible is now synced via panel-chrome's onShow/onHide callbacks
// at the bottom of this file. Default = hidden until user opens it.
let radarVisible = false;

export function isRadarVisible() { return radarVisible; }
// toggleRadar delegates to panel-chrome so dock + saved state stay in sync.
export function toggleRadar() {
  if (radarVisible) chrome.hidePanel('radar');
  else chrome.showPanel('radar');
}

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

// Each PC gets a stable color hue spaced around the wheel so the eye can
// learn "blue = PC3" type associations over time. (Duplicated from index.js
// and chemo-panel.js — the radar's only dependency on it is local, and the
// formula is one line.)
function pcHue(i) { return Math.round((i * 360) / 12); }

const EMOTION_AXIS_COLORS = {
  joy: '#cfa348', trust: '#cfa348', anticipation: '#cd7f5d', surprise: '#ece2cd',
  fear: '#8a7a9d', disgust: '#7d9d7f', sadness: '#7a93a8', anger: '#cd5d4a',
};

function _radarLayout() {
  // 4 cols × 3 rows = 12 charts, one per PC / neuron pair. (L and R of the
  // same pair have identical emotion content — only their magnitudes
  // differ — so 24 separate charts were redundant.)
  // W/H are the LOGICAL (CSS-pixel) drawing dims. Use the constants, not
  // canvas.width/devicePixelRatio: the backing store is sized with a capped
  // dpr (min(dpr,2)), so dividing the raw width by an uncapped devicePixelRatio
  // (e.g. 3 on many phones) yielded a too-small layout and clipped the charts.
  const COLS = 4, ROWS = 3;
  const W = RADAR_W;
  const H = RADAR_H;
  const padTop = 28, padBottom = 6, padX = 6;
  const cellW = (W - padX * 2) / COLS;
  const cellH = (H - padTop - padBottom) / ROWS;
  return { COLS, ROWS, W, H, padTop, padX, cellW, cellH };
}

// ---------------------------------------------------------------------------
// Renderer — called from the main render loop / WebSocket frame handler.
// `state` carries the live data the panel doesn't own:
//   { corpusPca, computeWordImpact }
// `computeWordImpact` is a zero-arg closure provided by index.js (it closes
// over smellsData + latestResidual). We invoke it lazily so the cost is
// only paid when the radar is actually visible.
// ---------------------------------------------------------------------------
export function drawRadar(state) {
  if (!radarVisible || !state || !state.corpusPca) return;
  const { corpusPca, computeWordImpact } = state;
  if (!corpusPca.emotion_keys || !corpusPca.emotions) {
    // Corpus PCA cache predates the emotion fields — rebuild needed.
    radarctx.clearRect(0, 0, radarcanvas.width, radarcanvas.height);
    radarctx.fillStyle = '#cfa348';
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
  ctx.fillStyle = '#e0c48f';
  ctx.font = 'bold 11px ui-monospace, monospace';
  ctx.textBaseline = 'top';
  ctx.fillText('● EMOTION RADAR — what each PC is absorbing', 8, 4);
  ctx.fillStyle = '#b3a789';
  ctx.font = '9px ui-monospace, monospace';
  if (hasEmotionalSignal) {
    ctx.fillText('contribution-weighted NRC emotions per PC pair', 8, 16);
  } else {
    ctx.fillStyle = '#cfa348';
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
    ctx.strokeStyle = 'rgba(205,127,93,0.10)';
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = 'rgba(205,127,93,0.07)';
    ctx.beginPath(); ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2); ctx.stroke();

    // Axis spokes + (on top-right cell only) emotion abbreviations
    ctx.strokeStyle = 'rgba(205,127,93,0.10)';
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
      ctx.fillStyle = EMOTION_AXIS_COLORS[emotionKeys[a]] || '#ece2cd';
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
        ctx.fillStyle = EMOTION_AXIS_COLORS[emotionKeys[a]] || '#ece2cd';
        ctx.fillText(emotionKeys[a].slice(0, 3), lx, ly);
      }
    }
  }
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
}

// Register with panel-chrome AFTER the toggle/flag declarations so
// onShow/onHide can flip radarVisible. Saved visibility applied immediately.
chrome.register({
  id: 'radar',
  label: 'emotion compass',
  panelEl: document.getElementById('radarpanel'),
  onShow: () => { radarVisible = true; },
  onHide: () => { radarVisible = false; },
});

export { radarcanvas };
