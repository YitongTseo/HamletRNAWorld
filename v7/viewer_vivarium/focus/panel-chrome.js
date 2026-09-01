// viewer/focus/panel-chrome.js
// Gives each panel: a ✕ close button, drag-to-move, and corner-resize, plus
// a labelled toggle button in the bottom-right bar (dock.js). panelEl MUST be
// a normal element (a <div> wrapper) — a ✕ appended to a bare <canvas> never
// renders, which is why the canvas panels used to look un-closeable.
import { saveVisibility, loadVisibility, savePanelGeom, loadPanelGeom } from './state.js';
import * as dock from './dock.js';
import { isMobile, onViewportChange } from './responsive.js';

const registered = new Map(); // id -> { panelEl, onShow, onHide, visible }

export function register({ id, label, panelEl, onShow, onHide, manageChrome = true }) {
  if (getComputedStyle(panelEl).position === 'static') {
    panelEl.style.position = 'fixed';
  }

  if (manageChrome) {
    panelEl.style.pointerEvents = 'auto';
    addCloseButton(id, panelEl, label);
    addResizeHandle(id, panelEl);
    makeDraggable(id, panelEl);
    restoreGeom(id, panelEl);
  }

  dock.register(id, label, () => togglePanel(id));
  registered.set(id, { panelEl, onShow, onHide, visible: false });

  // Apply saved visibility (default = hidden; strict — undefined means hidden).
  const saved = loadVisibility();
  if (saved[id] === true) showPanel(id, /*skipSave=*/true);
  else hidePanel(id, /*skipSave=*/true);

  if (isMobile()) panelEl.classList.add('panel--mobile');
}

export function togglePanel(id) {
  const r = registered.get(id);
  if (!r) return;
  if (r.visible) hidePanel(id); else showPanel(id);
}

export function showPanel(id, skipSave = false) {
  const r = registered.get(id);
  if (!r) return;
  // Explicit 'block' (not '') so it overrides any CSS `display:none` baked
  // into a panel's stylesheet rule (e.g. #radarpanel starts hidden in CSS).
  r.panelEl.style.display = 'block';
  r.visible = true;
  dock.setActive(id, true);
  r.onShow?.();
  if (!skipSave) saveVisibility(id, true);
}

export function hidePanel(id, skipSave = false) {
  const r = registered.get(id);
  if (!r) return;
  r.panelEl.style.display = 'none';
  r.visible = false;
  dock.setActive(id, false);
  r.onHide?.();
  if (!skipSave) saveVisibility(id, false);
}

// --- chrome bits ----------------------------------------------------------

function addCloseButton(id, el, label) {
  const btn = document.createElement('button');
  btn.className = 'panel-close-btn';
  btn.textContent = '✕';
  btn.setAttribute('aria-label', `close ${label}`);
  btn.addEventListener('click', (e) => { e.stopPropagation(); hidePanel(id); });
  // Stop pointerdown so it can't start a drag instead of a click.
  btn.addEventListener('pointerdown', (e) => e.stopPropagation());
  el.appendChild(btn);
}

function addResizeHandle(id, el) {
  const handle = document.createElement('div');
  handle.className = 'panel-resize-handle';
  el.appendChild(handle);
  let resizing = false, sx = 0, sy = 0, sw = 0, sh = 0;
  handle.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    resizing = true;
    try { handle.setPointerCapture(e.pointerId); } catch {}
    const rect = el.getBoundingClientRect();
    sx = e.clientX; sy = e.clientY; sw = rect.width; sh = rect.height;
  });
  handle.addEventListener('pointermove', (e) => {
    if (!resizing) return;
    el.style.width = Math.max(140, sw + (e.clientX - sx)) + 'px';
    el.style.height = Math.max(100, sh + (e.clientY - sy)) + 'px';
  });
  const end = (e) => {
    if (!resizing) return;
    resizing = false;
    try { handle.releasePointerCapture(e.pointerId); } catch {}
    persistGeom(id, el);
  };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
}

function makeDraggable(id, el) {
  let dragging = false, ox = 0, oy = 0;
  el.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.panel-close-btn, .panel-resize-handle')) return;
    dragging = true;
    el.classList.add('dragging');
    try { el.setPointerCapture(e.pointerId); } catch {}
    const rect = el.getBoundingClientRect();
    ox = e.clientX - rect.left;
    oy = e.clientY - rect.top;
    // Pin via left/top so dragging works regardless of original anchoring
    // (some panels are anchored bottom/right in CSS).
    el.style.left = rect.left + 'px';
    el.style.top = rect.top + 'px';
    el.style.right = 'auto';
    el.style.bottom = 'auto';
  });
  el.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const rect = el.getBoundingClientRect();
    let x = Math.max(0, Math.min(window.innerWidth - rect.width, e.clientX - ox));
    let y = Math.max(0, Math.min(window.innerHeight - rect.height, e.clientY - oy));
    el.style.left = x + 'px';
    el.style.top = y + 'px';
  });
  const end = (e) => {
    if (!dragging) return;
    dragging = false;
    el.classList.remove('dragging');
    try { el.releasePointerCapture(e.pointerId); } catch {}
    persistGeom(id, el);
  };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
}

function persistGeom(id, el) {
  savePanelGeom(id, {
    left: el.style.left, top: el.style.top,
    width: el.style.width, height: el.style.height,
  });
}

function restoreGeom(id, el) {
  const g = loadPanelGeom(id);
  if (!g) return;
  if (g.left) { el.style.left = g.left; el.style.right = 'auto'; }
  if (g.top) { el.style.top = g.top; el.style.bottom = 'auto'; }
  if (g.width) el.style.width = g.width;
  if (g.height) el.style.height = g.height;
}

onViewportChange(() => {
  const mobile = isMobile();
  for (const [, r] of registered) {
    r.panelEl.classList.toggle('panel--mobile', mobile);
  }
});
