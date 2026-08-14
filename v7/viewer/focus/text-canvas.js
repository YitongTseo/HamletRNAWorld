// Text overlay canvas: drifting Hamlet words, smell lines, PCA hover popup.
// Extracted from focus/index.js — behavior unchanged.
//
// DPR-correctness: the canvas backing store is sized to physical pixels
// (CSS px × devicePixelRatio) while the CSS style keeps the canvas at the
// viewport's CSS size. `setTransform(dpr, ...)` scales the 2D context so
// callers can keep drawing in CSS pixel coordinates. This makes the
// drifting Hamlet text render crisply on Retina/HiDPI displays.
//
// Consequence: `textcanvas.width` / `textcanvas.height` are now in PHYSICAL
// pixels, NOT CSS pixels. Any code computing screen-space positions in CSS
// coords must read `window.innerWidth` / `window.innerHeight` instead.
import { camera, canvas } from './three-scene.js';
import { isMobile } from './responsive.js';
import { drawPcaPopup, isPcaVisible } from './pca-popup.js';

// ---------------------------------------------------------------------------
// Canvas + 2D context
// ---------------------------------------------------------------------------
const textcanvas = document.getElementById('textcanvas');
const tctx = textcanvas.getContext('2d');
function resizeTextCanvas() {
  const dpr = window.devicePixelRatio || 1;
  textcanvas.width  = window.innerWidth  * dpr;
  textcanvas.height = window.innerHeight * dpr;
  textcanvas.style.width  = window.innerWidth  + 'px';
  textcanvas.style.height = window.innerHeight + 'px';
  // setTransform must be re-applied after each resize because changing the
  // canvas backing-store size implicitly resets the 2D context's transform.
  tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
resizeTextCanvas();
window.addEventListener('resize', resizeTextCanvas);

// ---------------------------------------------------------------------------
// World → screen projection (used by text + smell drawing, and later by the
// magnifier). Returns CSS pixel coords; the context's DPR transform handles
// the upscale to physical pixels.
// ---------------------------------------------------------------------------
// Project against the #c canvas's ACTUAL on-screen box, not window.innerWidth/
// innerHeight. On mobile the canvas is CSS-sized 100vh (the large viewport,
// incl. the area behind the address bar) while innerHeight is the small
// viewport, so dividing by innerHeight placed overlays (notably the magnifier
// x-ray) higher than the worm three.js actually renders. The rect is cached
// and refreshed only on viewport changes, so this stays cheap per-frame.
let _canvasRect = canvas.getBoundingClientRect();
function refreshCanvasRect() { _canvasRect = canvas.getBoundingClientRect(); }
window.addEventListener('resize', refreshCanvasRect);
window.addEventListener('scroll', refreshCanvasRect, true);
if (window.visualViewport) {
  window.visualViewport.addEventListener('resize', refreshCanvasRect);
  window.visualViewport.addEventListener('scroll', refreshCanvasRect);
}

function worldToScreen(wx, wy) {
  const r = _canvasRect;
  const nx = (wx - camera.left) / (camera.right - camera.left);
  const ny = (wy - camera.top) / (camera.bottom - camera.top);
  return [r.left + nx * r.width, r.top + ny * r.height];
}

function screenScale() {
  return window.innerWidth / (camera.right - camera.left);
}

// ---------------------------------------------------------------------------
// Hovered-word tracking (computed inside drawTextCanvas each frame).
// ---------------------------------------------------------------------------
let nearestWord = null;

// ---------------------------------------------------------------------------
// Main draw entry point. `state` carries the per-frame inputs sourced from
// the websocket + mouse handlers in index.js:
//   {
//     wordFoodMap,    // Map<string, {x,y,word,...}>
//     mouseWorldPos,  // {x,y}
//     mouseScreenPos, // {x,y}
//     smellsData,     // [{x,y,neurons:{...}}]
//     smellsVisible,  // boolean
//     wormHeadPos,    // {x,y}
//     pcaData,        // {tokens, pca, token_to_idx, hexBins, projection} | null
//   }
// ---------------------------------------------------------------------------
function drawTextCanvas(state) {
  const { wordFoodMap, mouseWorldPos, mouseScreenPos, pcaData } = state;
  // CSS pixel coords — context is DPR-scaled, so clearing in CSS units
  // wipes the full physical backing store.
  const w = window.innerWidth, h = window.innerHeight;
  tctx.clearRect(0, 0, w, h);

  // Draw smells first (so they appear behind words)
  drawSmells(state);

  // Desire layer ('d' key): per-word pull = the summed chemosensory
  // activation this word currently produces in the worm's nose — the sim's
  // own numbers (distance × direction × meaning × hunger gain), not a
  // recomputation. Salience, not steering: the turn decision uses the L/R
  // split, but "which word does it want most" is honestly this sum.
  // Keyed by rounded coords — snapshot food and smells round identically.
  let desireByPos = null, desireMax = 0, desireTop = null;
  if (state.desireVisible && state.smellsData && state.smellsData.length) {
    desireByPos = new Map();
    for (const smell of state.smellsData) {
      let pull = 0;
      for (const v of Object.values(smell.neurons || {})) pull += v;
      if (pull <= 0) continue;
      desireByPos.set(`${smell.x},${smell.y}`, pull);
      if (pull > desireMax) { desireMax = pull; desireTop = smell; }
    }
  }

  // Find nearest word to mouse cursor
  let minDist = Infinity;
  nearestWord = null;
  for (const [key, item] of wordFoodMap) {
    const dx = item.x - mouseWorldPos.x, dy = item.y - mouseWorldPos.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < minDist) {
      minDist = d;
      nearestWord = item;
    }
  }

  // Draw each word. Smaller on mobile so the lines don't overlap when the
  // camera frustum is zoomed in (#9).
  tctx.font = (isMobile() ? '13px' : '18px') + ' ui-monospace, monospace';
  for (const [key, item] of wordFoodMap) {
    const [sx, sy] = worldToScreen(item.x, item.y);
    const isHovered = nearestWord === item && minDist < 80;
    // Edible words are bright white (the worm can eat these). Inedible
    // set-dressing (speaker names, stage cues, directions) is drawn in a dim
    // blue-grey so it recedes — echoes the bluish word color in the overview
    // view. `edible !== false` so older payloads (no flag) default to edible.
    const edible = item.edible !== false;
    // Desire tint: cold white → warm amber → hot orange as the worm's pull
    // toward this word rises, glow strength following. Words outside smell
    // range (no entry) stay stock.
    const pull = desireByPos ? desireByPos.get(`${item.x},${item.y}`) : undefined;
    if (edible && pull !== undefined && desireMax > 0) {
      const t = pull / desireMax;                    // 0..1 this frame
      const hue = 55 - 30 * t;                       // amber → orange
      tctx.fillStyle = isHovered
        ? `hsla(${hue}, 100%, 70%, 1.0)`
        : `hsla(${hue}, ${40 + 60 * t}%, ${70 + 15 * t}%, ${0.7 + 0.3 * t})`;
      tctx.shadowColor = `hsla(${hue}, 100%, 60%, ${0.9 * t})`;
      tctx.shadowBlur = 14 * t;
    } else if (edible) {
      tctx.fillStyle = isHovered ? 'rgba(255,255,255,1.0)' : 'rgba(255,255,255,0.7)';
    } else {
      tctx.fillStyle = isHovered ? 'rgba(170,190,225,0.85)' : 'rgba(150,170,205,0.45)';
    }
    tctx.textAlign = 'center';
    tctx.textBaseline = 'middle';
    tctx.fillText(item.word, sx, sy);
    tctx.shadowBlur = 0;
  }

  // Ring the single strongest pull — the word the worm most wants right now.
  if (desireTop) {
    const [dx, dy] = worldToScreen(desireTop.x, desireTop.y);
    const r = 16 + 4 * Math.sin(performance.now() / 300);  // slow breathe
    tctx.strokeStyle = 'hsla(25, 100%, 60%, 0.9)';
    tctx.lineWidth = 2;
    tctx.beginPath();
    tctx.arc(dx, dy, r, 0, Math.PI * 2);
    tctx.stroke();
  }

  // PCA popup for nearest hovered word — only when the user has opened it
  // via the dock (glyph [x,y]). Default = closed.
  if (isPcaVisible() && nearestWord && minDist < 80 && pcaData) {
    drawPcaPopup(nearestWord.word, mouseScreenPos, pcaData);
  }
}

function drawSmells(state) {
  const { smellsData, smellsVisible, wormHeadPos } = state;
  if (!smellsVisible || smellsData.length === 0) return;

  // Neuron type colors
  const neuronColors = {
    ASE: { h: 240, s: 100, l: 50 },  // Blue - valence
    AWA: { h: 120, s: 100, l: 50 },  // Green - appetitive
    AWB: { h: 150, s: 100, l: 50 },  // Cyan - approach
    AWC: { h: 180, s: 100, l: 50 },  // Turquoise - CO2
    ASI: { h: 60, s: 100, l: 50 },   // Yellow - intensity
    ASJ: { h: 30, s: 100, l: 50 },   // Orange - feeding
    ASH: { h: 0, s: 100, l: 50 },    // Red - protective
  };

  const [headsx, headsy] = worldToScreen(wormHeadPos.x, wormHeadPos.y);
  const headRadius = 12;  // Worm head radius for offset calculation

  for (const smell of smellsData) {
    if (!smell.neurons || Object.keys(smell.neurons).length === 0) continue;

    const [wordsx, wordsy] = worldToScreen(smell.x, smell.y);

    // Draw separate lines for each active chemosensory neuron
    for (const [neuronName, activation] of Object.entries(smell.neurons)) {
      if (activation === 0) continue;

      // Get neuron type (first 3 chars: ASE, AWA, AWB, etc.)
      const neuronType = neuronName.substring(0, 3);
      const neuronSide = neuronName.endsWith('L') ? 'L' :
                         neuronName.endsWith('R') ? 'R' : 'C';

      const color = neuronColors[neuronType] || { h: 210, s: 100, l: 50 };
      const intensity = Math.min(1.0, activation);
      const lineWidth = 1 + intensity * 3;

      // Offset line origin based on neuron side (bilateral asymmetry)
      let startX = headsx;
      let startY = headsy;

      if (neuronSide === 'L') {
        startX -= headRadius * 0.6;  // Left side neurons start left
      } else if (neuronSide === 'R') {
        startX += headRadius * 0.6;  // Right side neurons start right
      }
      // Center neurons start from head center

      tctx.strokeStyle = `hsla(${color.h}, ${color.s}%, ${color.l}%, ${intensity * 0.8})`;
      tctx.lineWidth = lineWidth;
      tctx.setLineDash([5, 5]);
      tctx.lineCap = 'round';
      tctx.lineJoin = 'round';

      tctx.beginPath();
      tctx.moveTo(startX, startY);
      tctx.lineTo(wordsx, wordsy);
      tctx.stroke();

      tctx.setLineDash([]);
    }

    // Small circle at smell source
    let maxActivation = 0;
    for (const activation of Object.values(smell.neurons)) {
      maxActivation = Math.max(maxActivation, activation);
    }
    if (maxActivation > 0) {
      const radius = 2 + maxActivation * 3;
      tctx.fillStyle = `hsla(210, 100%, 50%, ${maxActivation * 0.6})`;
      tctx.beginPath();
      tctx.arc(wordsx, wordsy, radius, 0, Math.PI * 2);
      tctx.fill();
    }
  }
}

export { textcanvas, tctx, resizeTextCanvas, drawTextCanvas, worldToScreen };
