// viewer/focus/magnifier.js
// ---------------------------------------------------------------------------
// Draggable circular lens that overlays a live neural x-ray on top of the
// worm body underneath. The lens has its own DPR-correct 2D canvas; the
// x-ray drawing routine (./xray-render.js) is called with a custom
// `worldToScreen` so the worm body / neurons appear in their actual page-
// space position (translated into the lens's own canvas coords), giving
// the "peek through the body" effect rather than "fit-the-worm-in-a-panel".
//
// Lifecycle:
//   - At import time: build DOM, register with panel-chrome (default hidden),
//     listen for viewport changes to resize.
//   - Each frame: `render()` is called from index.js's main render loop.
//     It skips work when hidden.
//
// Persistence: lens position is saved to localStorage via state.js.
//
// Drag: Pointer Events API — unified mouse/touch with setPointerCapture.

import { drawXRay } from './xray-render.js';
import { worldToScreen as pageWorldToScreen } from './text-canvas.js';
import * as chrome from './panel-chrome.js';
import { loadMagnifierPos, saveMagnifierPos } from './state.js';
import { onViewportChange } from './responsive.js';

let lensEl, lensCanvas, lensCtx, closeBtn;
let xrayLabelsVisible = false;

// Module-local handle on live sim data; set by `setState` each frame from
// index.js so we can forward it into drawXRay. Avoids each-frame allocation.
let liveState = null;

// ---------------------------------------------------------------------------
// Sizing — diameter scales with viewport; clamped so it stays usable on both
// phones (~120 px) and big desktops (~360 px).
// ---------------------------------------------------------------------------
function diameter() {
  const d = Math.round(Math.min(window.innerWidth, window.innerHeight) * 0.30);
  return Math.max(120, Math.min(360, d));
}

// ---------------------------------------------------------------------------
// DOM construction. Called once at module load.
// ---------------------------------------------------------------------------
function build() {
  lensEl = document.createElement('div');
  lensEl.className = 'magnifier';
  lensCanvas = document.createElement('canvas');
  closeBtn = document.createElement('button');
  closeBtn.className = 'magnifier-close';
  closeBtn.textContent = '✕'; // ✕
  closeBtn.setAttribute('aria-label', 'close magnifier');
  closeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    chrome.hidePanel('magnifier');
  });
  // Also stop pointerdown on the close button from initiating a drag —
  // otherwise the lens gobbles the click via setPointerCapture before
  // the close button's click fires.
  closeBtn.addEventListener('pointerdown', (e) => e.stopPropagation());
  lensEl.appendChild(lensCanvas);
  lensEl.appendChild(closeBtn);
  document.body.appendChild(lensEl);

  applySize();
  applyInitialPosition();
  wireDrag();
}

// ---------------------------------------------------------------------------
// Size the lens div + DPR-correct the backing canvas. Re-applied on
// viewport changes so the lens stays appropriate on rotate/resize.
// ---------------------------------------------------------------------------
function applySize() {
  const d = diameter();
  lensEl.style.width = d + 'px';
  lensEl.style.height = d + 'px';
  const dpr = window.devicePixelRatio || 1;
  lensCanvas.width  = d * dpr;
  lensCanvas.height = d * dpr;
  // Re-fetch the context after a backing-store resize and re-apply the DPR
  // transform — changing canvas.width resets the 2D context's transform.
  lensCtx = lensCanvas.getContext('2d');
  lensCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ---------------------------------------------------------------------------
// Initial position: saved-from-last-session if available, otherwise centered.
// Clamped to viewport so a saved off-screen position from a smaller viewport
// can't strand the lens.
// ---------------------------------------------------------------------------
function applyInitialPosition() {
  const d = diameter();
  const saved = loadMagnifierPos();
  let x, y;
  if (saved && saved.x != null && saved.y != null) {
    x = saved.x; y = saved.y;
  } else {
    x = (window.innerWidth  - d) / 2;
    y = (window.innerHeight - d) / 2;
  }
  x = Math.max(0, Math.min(window.innerWidth  - d, x));
  y = Math.max(0, Math.min(window.innerHeight - d, y));
  lensEl.style.left = x + 'px';
  lensEl.style.top  = y + 'px';
}

// ---------------------------------------------------------------------------
// Pointer-events drag: works for mouse + touch + pen uniformly. Capturing
// the pointer means we still get move events even if the cursor leaves the
// lens during a drag. `touch-action: none` (in focus.css) prevents the
// browser from intercepting two-finger pan/zoom.
// ---------------------------------------------------------------------------
function wireDrag() {
  let dragging = false;
  let offsetX = 0, offsetY = 0;
  lensEl.addEventListener('pointerdown', (e) => {
    if (e.target === closeBtn || closeBtn.contains(e.target)) return;
    dragging = true;
    lensEl.classList.add('dragging');
    try { lensEl.setPointerCapture(e.pointerId); } catch {}
    const rect = lensEl.getBoundingClientRect();
    offsetX = e.clientX - rect.left;
    offsetY = e.clientY - rect.top;
    e.preventDefault();
  });
  lensEl.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const d = diameter();
    let x = e.clientX - offsetX;
    let y = e.clientY - offsetY;
    x = Math.max(0, Math.min(window.innerWidth  - d, x));
    y = Math.max(0, Math.min(window.innerHeight - d, y));
    lensEl.style.left = x + 'px';
    lensEl.style.top  = y + 'px';
  });
  const release = (e) => {
    if (!dragging) return;
    dragging = false;
    lensEl.classList.remove('dragging');
    try { lensEl.releasePointerCapture(e.pointerId); } catch {}
    const x = parseFloat(lensEl.style.left) || 0;
    const y = parseFloat(lensEl.style.top)  || 0;
    saveMagnifierPos({ x, y, diameter: diameter() });
  };
  lensEl.addEventListener('pointerup', release);
  lensEl.addEventListener('pointercancel', release);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Caller (index.js) hands in the same live sim data it passes to the
// network panel — drawXRay needs `neuronBodyCoords`, `graph`, `neuronActivity`.
export function setState(state) { liveState = state; }

// Called once per frame from the main render loop. Cheap no-op when the
// lens is hidden (default state on first visit).
export function render() {
  if (!lensEl || lensEl.style.display === 'none' || !liveState) return;
  const rect = lensEl.getBoundingClientRect();
  // Build a worldToScreen that maps world coords → THIS canvas's local
  // coords by translating the page-space projection by the lens's
  // top-left. The xray-render's `lensMode` path uses this to project the
  // live midline directly, so the silhouette appears under the lens at
  // the actual worm's on-screen position.
  const lensWorldToScreen = (wx, wy) => {
    const [px, py] = pageWorldToScreen(wx, wy);
    return [px - rect.left, py - rect.top];
  };
  // Lens canvas is sized to `rect.width` × `rect.height` CSS px (DPR
  // transform applied in applySize). drawXRay clears the rect itself.
  drawXRay(lensCtx, { x: 0, y: 0, w: rect.width, h: rect.height }, {
    xrayLabelsVisible,
    neuronBodyCoords: liveState.neuronBodyCoords,
    graph: liveState.graph,
    neuronActivity: liveState.neuronActivity,
    worldToScreen: lensWorldToScreen,
  });
}

// Synced from index.js when the user presses 'l' so the lens labels stay
// in lockstep with the network panel's labels.
export function setXrayLabelsVisible(v) { xrayLabelsVisible = !!v; }

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
build();
chrome.register({
  id: 'magnifier',
  glyph: '(°o°)',  // (°o°)
  label: 'neural x-ray magnifier',
  panelEl: lensEl,
});
onViewportChange(() => { applySize(); applyInitialPosition(); });
