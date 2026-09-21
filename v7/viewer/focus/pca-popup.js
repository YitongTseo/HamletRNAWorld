// PCA hover popup: small density-plot panel drawn next to the mouse cursor
// when hovering near a Hamlet word. Extracted from text-canvas.js — behavior
// unchanged.
//
// `drawTextCanvas` in text-canvas.js owns the hover-detection (finding the
// nearest word and distance threshold). This module only handles the
// actual popup rendering once the hover decision has been made.
import { tctx } from './text-canvas.js';

// ---------------------------------------------------------------------------
// Visibility flag. The UMAP/PCA hover popup (hex-density plot) draws into
// #textcanvas; it's ON by default and is no longer a toggle-bar entry — it
// just appears when you hover near a word. text-canvas.js gates the actual
// drawPcaPopup call on isPcaVisible().
// ---------------------------------------------------------------------------
let pcaVisible = true;
export function setPcaVisible(v) { pcaVisible = v; }
export function isPcaVisible() { return pcaVisible; }

function drawPcaPopup(word, mouseScreenPos, pcaData) {
  const PW = 200, PH = 200, PAD_X = 20, PAD_Y = 20;
  // Position popup at mouse cursor with offset
  let px = mouseScreenPos.x + PAD_X;
  let py = mouseScreenPos.y + PAD_Y;

  // Clamp to viewport. Read CSS-pixel viewport dims here — `textcanvas.width`
  // is now in physical pixels (DPR-scaled), so it would give wrong clamps on
  // Retina. The 2D context is DPR-scaled, so drawing math is all CSS px.
  px = Math.min(px, window.innerWidth - PW - 4);
  py = Math.min(py, window.innerHeight - PH - 4);
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

export { drawPcaPopup };
