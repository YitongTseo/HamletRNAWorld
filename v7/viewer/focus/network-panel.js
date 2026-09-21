// ---------------------------------------------------------------------------
// Network panel — bottom-right 500×360 canvas (#netcanvas).
// ---------------------------------------------------------------------------
// Owns the panel DOM ref + 2D context + DPR setup, and dispatches between
// two views:
//   - x-ray (default): neurons painted on the moving worm silhouette
//     (delegates to ./xray-render.js)
//   - legacy graph: static anatomical layout with edges from firing nodes
//
// Toggles:
//   'n' — show/hide the panel
//   'x' — flip between x-ray and legacy graph view
//   'l' — show/hide neuron name labels (x-ray view only)
//   'm' — show/hide motor labels (legacy graph view only)

import { drawXRay, NEURON_CLASS_PALETTE, drawNeuronLegend } from './xray-render.js';
import * as chrome from './panel-chrome.js';

// ---------------------------------------------------------------------------
// Canvas / DPR setup
// ---------------------------------------------------------------------------
const netcanvas = document.getElementById('netcanvas');
const ctx = netcanvas.getContext('2d');
const NET_W = 500, NET_H = 360;
const LEGEND_H = 72;
const NEURO_TOP = LEGEND_H + 2;
const NEURO_H = NET_H - LEGEND_H;
const PAD = 12;
{
  const dpr = window.devicePixelRatio || 1;
  netcanvas.width  = NET_W * dpr;
  netcanvas.height = NET_H * dpr;
  ctx.scale(dpr, dpr);
}

// ---------------------------------------------------------------------------
// Static anatomical layout for the legacy graph view.
// Map OpenWorm (AP, DV) coords to canvas coords once at graph-init time;
// `drawNetCanvas` then reads from the resulting `pos` Float32Array.
// ---------------------------------------------------------------------------
const AP_MIN = -290, AP_MAX = 420;
const DV_MIN = -90,  DV_MAX = 65;

export function buildPositions(neurons, rawPositions) {
  const N = neurons.length;
  const pos = new Float32Array(N * 2);
  const apRange = AP_MAX - AP_MIN;
  const dvRange = DV_MAX - DV_MIN;
  const neuroW = NET_W - PAD * 2;
  let fallbackX = PAD + neuroW * 0.6;

  for (let i = 0; i < N; i++) {
    const xyz = rawPositions[i];
    let cx, cy;
    if (xyz) {
      cx = PAD + (xyz[1] - AP_MIN) / apRange * neuroW;
      cy = NEURO_TOP + PAD + (1 - (xyz[2] - DV_MIN) / dvRange) * (NEURO_H - PAD * 2);
    } else {
      cx = fallbackX;
      cy = NEURO_TOP + NEURO_H / 2 + (Math.random() - 0.5) * 40;
      fallbackX = PAD + neuroW * 0.6 + ((fallbackX + 3 - PAD) % (neuroW * 0.4));
    }
    pos[i * 2]     = cx;
    pos[i * 2 + 1] = cy;
  }
  return pos;
}

// ---------------------------------------------------------------------------
// View-mode flags (module-local)
// ---------------------------------------------------------------------------
// netVisible is now synced via panel-chrome's onShow/onHide callbacks below.
// chrome.register applies the saved visibility (default = hidden) and fires
// the appropriate callback, so the flag starts coherent with the DOM.
let netVisible = false;
let xrayMode = false;          // default = labelled anatomical graph; 'x' flips to the live body overlay
let xrayLabelsVisible = false; // neuron-name labels OFF by default in the live body view — press 'l' to show, or read them in the connectome graph ('x')
let motorLabelsVisible = true; // motor labels on by default in the anatomical graph

export function isNetVisible() { return netVisible; }
// toggleNetVisible delegates to panel-chrome so the dock button + saved
// state stay in sync with the keyboard shortcut. The onShow/onHide
// callbacks below flip `netVisible` so subsequent reads remain coherent.
export function toggleNetVisible() {
  if (netVisible) chrome.hidePanel('network');
  else chrome.showPanel('network');
}
export function toggleXrayMode() { xrayMode = !xrayMode; }
// Exported for Task 17: the magnifier also needs to react to the 'l' key
// so both surfaces stay in sync.
export function setXrayLabelsVisible(v) { xrayLabelsVisible = !!v; }
export function getXrayLabelsVisible() { return xrayLabelsVisible; }
export function toggleXrayLabels() { xrayLabelsVisible = !xrayLabelsVisible; }
export function toggleMotorLabels() { motorLabelsVisible = !motorLabelsVisible; }
export function getMotorLabelsVisible() { return motorLabelsVisible; }

// ---------------------------------------------------------------------------
// Legacy graph view (static anatomical layout)
// ---------------------------------------------------------------------------
function drawNetCanvas(graph, neuronActivity, stimFlags) {
  if (!graph || !netVisible) return;
  const { neurons, fireThreshold, muscleSet, sensorySet, chemosensorySet,
          motorSet, foodSet, noseSet, hungerSet, pos, adjOut, N } = graph;

  ctx.clearRect(0, 0, NET_W, NET_H);

  // ── Header ──────────────────────────────────────────────────────────────
  ctx.textBaseline = 'top';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#8f8';
  ctx.font = 'bold 11px ui-monospace, monospace';
  ctx.fillText('● CONNECTOME (anatomical layout)', 8, 6);
  ctx.fillStyle = '#9c9';
  ctx.font = '9px ui-monospace, monospace';
  ctx.textAlign = 'right';
  ctx.fillText("'x' → live body connectome", NET_W - 8, 19);
  ctx.textAlign = 'left';

  drawNeuronLegend(ctx, 0, 36);

  ctx.fillStyle = 'rgba(68,255,119,0.35)';
  ctx.font = '8px ui-monospace,monospace';
  ctx.fillText('[n] toggle panel  [m] motor labels' + (motorLabelsVisible ? '  ✓' : ''), 8, 48);

  ctx.fillStyle = 'rgba(68,255,119,0.2)';
  ctx.fillText('← head', PAD, LEGEND_H - 2);
  ctx.fillText('tail →', NET_W - 40, LEGEND_H - 2);
  ctx.fillText('dorsal', PAD, NEURO_TOP + 10);
  ctx.fillText('ventral', PAD, NET_H - 4);

  // ── Build activity array ────────────────────────────────────────────────
  const activity = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const v = neuronActivity[neurons[i]];
    if (v) activity[i] = v;
  }

  // ── Edges: from firing neurons only ─────────────────────────────────────
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(68,255,119,0.15)';
  ctx.lineWidth = 0.4;
  for (let pi = 0; pi < N; pi++) {
    if (activity[pi] <= fireThreshold) continue;
    const px = pos[pi*2], py = pos[pi*2+1];
    for (const qi of adjOut[pi]) {
      ctx.moveTo(px, py);
      ctx.lineTo(pos[qi*2], pos[qi*2+1]);
    }
  }
  ctx.stroke();

  // ── Nodes ───────────────────────────────────────────────────────────────
  const hungerOn = stimFlags.hunger;
  const noseOn   = stimFlags.nose_touch;
  const foodOn   = stimFlags.food_sense;
  const pendingLabels = [];

  for (let i = 0; i < N; i++) {
    const x = pos[i*2], y = pos[i*2+1];
    const v = activity[i];
    const firing  = v > fireThreshold;
    const charged = v > 0;
    const t = firing ? 1 : (charged ? Math.min(v / fireThreshold, 1) : 0);

    const stimulated =
      (foodSet.has(i)   && foodOn) ||
      (noseSet.has(i)   && noseOn) ||
      (hungerSet.has(i) && hungerOn);

    const isChemo   = chemosensorySet.has(i);
    const isSensory = sensorySet.has(i);
    const isMotor   = motorSet.has(i);
    const isMuscle  = muscleSet.has(i);

    let baseColor, fireColor, r, haloColor, hr;
    if (isChemo) {
      baseColor  = `rgba(40,200,255,${(0.25 + t * 0.7).toFixed(2)})`;
      fireColor  = 'rgba(40,255,255,0.95)';
      haloColor  = 'rgba(40,220,255,0.22)';
      r = 1.8 + t;  hr = 7;
    } else if (isSensory) {
      baseColor  = `rgba(100,160,255,${(0.2 + t * 0.75).toFixed(2)})`;
      fireColor  = 'rgba(120,180,255,0.95)';
      haloColor  = 'rgba(100,160,255,0.2)';
      r = 1.5 + t;  hr = 6;
    } else if (isMotor) {
      baseColor  = `rgba(255,150,40,${(0.2 + t * 0.75).toFixed(2)})`;
      fireColor  = 'rgba(255,180,40,0.95)';
      haloColor  = 'rgba(255,150,40,0.2)';
      r = 1.5 + t;  hr = 6;
    } else if (isMuscle) {
      baseColor  = `rgba(255,220,40,${(0.12 + t * 0.7).toFixed(2)})`;
      fireColor  = 'rgba(255,240,80,0.9)';
      haloColor  = 'rgba(255,200,40,0.18)';
      r = 1.2 + t * 0.8;  hr = 5;
    } else {
      baseColor  = `rgba(68,180,80,${(0.15 + t * 0.8).toFixed(2)})`;
      fireColor  = '#44ff77';
      haloColor  = 'rgba(68,255,119,0.18)';
      r = 1.3 + t * 0.9;  hr = 6;
    }

    const useCyan = stimulated && !firing;
    const color = useCyan ? 'rgba(40,255,255,0.9)' : (firing ? fireColor : baseColor);
    const halo  = (firing || stimulated || t > 0.5) ? haloColor : null;
    const haloR = (firing || stimulated) ? hr : hr * t;

    if (halo && haloR > 1) {
      ctx.beginPath();
      ctx.arc(x, y, haloR, 0, Math.PI * 2);
      ctx.fillStyle = firing ? haloColor : (useCyan ? 'rgba(40,255,255,0.15)' : haloColor);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(x, y, Math.max(0.8, r), 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    const shouldLabel =
      firing || stimulated ||
      (isMotor && motorLabelsVisible);
    if (shouldLabel && (isSensory || isChemo || isMotor)) {
      pendingLabels.push({ x, y, name: neurons[i], firing, isChemo, isSensory, isMotor });
    }
  }

  // ── Labels (second pass) ────────────────────────────────────────────────
  ctx.font = '6.5px ui-monospace,monospace';
  for (const {x, y, name, firing, isChemo, isSensory, isMotor} of pendingLabels) {
    let lc;
    if (isChemo)   lc = firing ? 'rgba(40,255,255,0.95)' : 'rgba(40,200,255,0.65)';
    else if (isSensory) lc = firing ? 'rgba(140,200,255,0.95)' : 'rgba(100,160,255,0.55)';
    else           lc = firing ? 'rgba(255,200,80,0.95)' : 'rgba(255,150,40,0.5)';
    ctx.fillStyle = lc;
    ctx.fillText(name, x + 3, y - 3);
  }
}

// ---------------------------------------------------------------------------
// Dispatcher — called once per frame from the main render loop.
// `state` carries the live data the panel doesn't own:
//   { neuronBodyCoords, graph, neuronActivity, stimFlags }
// ---------------------------------------------------------------------------
export function drawNetworkPanel(state) {
  if (!netVisible) return;
  const { neuronBodyCoords, graph, neuronActivity, stimFlags } = state;
  if (xrayMode) {
    drawXRay(ctx, { x: 0, y: 0, w: NET_W, h: NET_H }, {
      xrayLabelsVisible,
      neuronBodyCoords,
      graph,
      neuronActivity,
    });
  } else {
    drawNetCanvas(graph, neuronActivity, stimFlags);
  }
}

// Register with panel-chrome AFTER the toggle/flag declarations so the
// onShow/onHide callbacks can flip netVisible. chrome.register applies the
// saved visibility immediately, so on a fresh visit the panel is hidden.
chrome.register({
  id: 'network',
  label: 'connectome',
  panelEl: document.getElementById('netpanel'),
  onShow: () => { netVisible = true; },
  onHide: () => { netVisible = false; },
});

export { netcanvas, ctx, NET_W, NET_H, NEURON_CLASS_PALETTE };
