// Text overlay canvas: drifting Hamlet words, smell lines, PCA hover popup.
// Extracted from focus/index.js — behavior unchanged.
//
// NOTE: this module is currently a pure extraction. The DPR-correctness fix
// for fuzzy text on Retina displays is Task 10 of the redesign plan.
import { camera } from './three-scene.js';

// ---------------------------------------------------------------------------
// Canvas + 2D context
// ---------------------------------------------------------------------------
const textcanvas = document.getElementById('textcanvas');
const tctx = textcanvas.getContext('2d');
function resizeTextCanvas() {
  textcanvas.width = window.innerWidth;
  textcanvas.height = window.innerHeight;
}
resizeTextCanvas();
window.addEventListener('resize', resizeTextCanvas);

// ---------------------------------------------------------------------------
// World → screen projection (used by text + smell drawing, and later by the
// magnifier).
// ---------------------------------------------------------------------------
function worldToScreen(wx, wy) {
  const nx = (wx - camera.left) / (camera.right - camera.left);
  const ny = (wy - camera.top) / (camera.bottom - camera.top);
  return [nx * textcanvas.width, ny * textcanvas.height];
}

function screenScale() {
  return textcanvas.width / (camera.right - camera.left);
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
  const w = textcanvas.width, h = textcanvas.height;
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

  // PCA popup for nearest hovered word
  if (nearestWord && minDist < 80 && pcaData) {
    drawPcaPopup(nearestWord.word, mouseScreenPos, pcaData);
  }
}

function drawPcaPopup(word, mouseScreenPos, pcaData) {
  const PW = 200, PH = 200, PAD_X = 20, PAD_Y = 20;
  // Position popup at mouse cursor with offset
  let px = mouseScreenPos.x + PAD_X;
  let py = mouseScreenPos.y + PAD_Y;

  // Clamp to viewport
  px = Math.min(px, textcanvas.width - PW - 4);
  py = Math.min(py, textcanvas.height - PH - 4);
  px = Math.max(px, 4);
  py = Math.max(py, 4);

  // Background
  tctx.fillStyle = 'rgba(0,0,0,0.9)';
  tctx.fillRect(px, py, PW, PH);
  tctx.strokeStyle = 'rgba(255,255,255,0.3)';
  tctx.lineWidth = 1;
  tctx.strokeRect(px, py, PW, PH);

  // Crosshairs
  tctx.strokeStyle = 'rgba(255,255,255,0.08)';
  tctx.beginPath();
  tctx.moveTo(px + PW / 2, py);
  tctx.lineTo(px + PW / 2, py + PH);
  tctx.moveTo(px, py + PH / 2);
  tctx.lineTo(px + PW, py + PH / 2);
  tctx.stroke();

  // Tiny projection label (top-right corner)
  tctx.fillStyle = 'rgba(180,180,180,0.5)';
  tctx.font = '8px ui-monospace, monospace';
  tctx.textAlign = 'right';
  tctx.fillText(pcaData.projection || 'pca', px + PW - 4, py + 10);

  const { tokens, pca, token_to_idx, hexBins } = pcaData;
  const margin = 16;
  const innerW = PW - 2 * margin;
  const innerH = PH - 2 * margin;
  const ox = px + margin;
  const oy = py + margin;

  // Hex density plot. Each filled hexagon's intensity is log-scaled by the
  // count of words that fell into it — white = densest cluster cores,
  // black = empty regions. Plot extents are letterboxed into the popup so
  // x and y get the same per-unit pixel scale.
  if (hexBins) {
    const plotS = Math.min(innerW, innerH);
    const plotOx = ox + (innerW - plotS) / 2;
    const plotOy = oy + (innerH - plotS) / 2;
    const rPx = hexBins.radius * plotS;
    const logMax = Math.log(1 + hexBins.maxCount);
    // Pre-compute the 6 hexagon vertices once (pointy-top: vertices at 30°,
    // 90°, 150°, 210°, 270°, 330° — angle = π/6 + i·π/3).
    const cos = [], sin = [];
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 6 + i * Math.PI / 3;
      cos.push(Math.cos(a) * rPx);
      sin.push(Math.sin(a) * rPx);
    }
    for (let i = 0; i < hexBins.centers.length; i++) {
      const c = hexBins.counts[i];
      if (c === 0) continue;
      const t = Math.log(1 + c) / logMax;
      const g = Math.round(t * 255);
      tctx.fillStyle = `rgb(${g},${g},${g})`;
      const cx = plotOx + hexBins.centers[i][0] * plotS;
      const cy = plotOy + hexBins.centers[i][1] * plotS;
      tctx.beginPath();
      tctx.moveTo(cx + cos[0], cy + sin[0]);
      for (let k = 1; k < 6; k++) tctx.lineTo(cx + cos[k], cy + sin[k]);
      tctx.closePath();
      tctx.fill();
    }
  } else {
    // Fallback: hex bins haven't been computed yet (shouldn't normally
    // happen since they're populated in initCorpusEmbeddings, but a
    // mid-load hover would hit this path).
    tctx.fillStyle = 'rgba(180,180,180,0.35)';
    for (let i = 0; i < tokens.length; i++) {
      const cx = ox + pca[i][0] * innerW;
      const cy = oy + pca[i][1] * innerH;
      tctx.beginPath();
      tctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
      tctx.fill();
    }
  }

  // Highlighted word — small contrasting marker drawn on top of the
  // density plot. Cyan reads well against the grayscale background.
  const idx = token_to_idx[word] ?? token_to_idx[word.toLowerCase()];
  if (idx !== undefined) {
    const [cx01, cy01] = pca[idx];
    let hx, hy;
    if (hexBins) {
      const plotS = Math.min(innerW, innerH);
      const plotOx = ox + (innerW - plotS) / 2;
      const plotOy = oy + (innerH - plotS) / 2;
      hx = plotOx + cx01 * plotS;
      hy = plotOy + cy01 * plotS;
    } else {
      hx = ox + cx01 * innerW;
      hy = oy + cy01 * innerH;
    }
    // Outer dark halo + cyan dot for contrast against both white-dense and
    // black-empty hexes.
    tctx.fillStyle = 'rgba(0,0,0,0.85)';
    tctx.beginPath();
    tctx.arc(hx, hy, 5, 0, Math.PI * 2);
    tctx.fill();
    tctx.fillStyle = 'rgba(120,220,255,1)';
    tctx.beginPath();
    tctx.arc(hx, hy, 3, 0, Math.PI * 2);
    tctx.fill();
    // Label with its own dark backdrop.
    tctx.font = '9px ui-monospace, monospace';
    tctx.textAlign = 'left';
    const tw = tctx.measureText(word).width;
    tctx.fillStyle = 'rgba(0,0,0,0.7)';
    tctx.fillRect(hx + 5, hy - 5, tw + 4, 11);
    tctx.fillStyle = 'rgba(255,255,255,0.95)';
    tctx.fillText(word, hx + 7, hy + 3);
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
