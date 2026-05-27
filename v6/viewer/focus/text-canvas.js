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
import { camera } from './three-scene.js';
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
function worldToScreen(wx, wy) {
  const nx = (wx - camera.left) / (camera.right - camera.left);
  const ny = (wy - camera.top) / (camera.bottom - camera.top);
  return [nx * window.innerWidth, ny * window.innerHeight];
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

  // Draw each word
  tctx.font = '18px ui-monospace, monospace';
  for (const [key, item] of wordFoodMap) {
    const [sx, sy] = worldToScreen(item.x, item.y);
    const isHovered = nearestWord === item && minDist < 80;
    tctx.fillStyle = isHovered ? 'rgba(255,255,255,1.0)' : 'rgba(255,255,255,0.7)';
    tctx.textAlign = 'center';
    tctx.textBaseline = 'middle';
    tctx.fillText(item.word, sx, sy);
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
