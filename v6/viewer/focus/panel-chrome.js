// viewer/focus/panel-chrome.js
import { saveVisibility, loadVisibility } from './state.js';
import * as dock from './dock.js';
import { isMobile, onViewportChange } from './responsive.js';

const registered = new Map(); // id -> { panelEl, onShow, onHide }

export function register({ id, glyph, label, panelEl, onShow, onHide }) {
  // Ensure the panel is positioned relatively for the absolute close button.
  if (getComputedStyle(panelEl).position === 'static') {
    panelEl.style.position = 'fixed';
  }
  // Insert the close button.
  const closeBtn = document.createElement('button');
  closeBtn.className = 'panel-close-btn';
  closeBtn.textContent = '✕'; // ✕
  closeBtn.setAttribute('aria-label', `close ${label}`);
  closeBtn.addEventListener('click', () => hidePanel(id));
  panelEl.appendChild(closeBtn);

  // Register the dock button to reopen.
  dock.register(id, glyph, label, () => showPanel(id));

  registered.set(id, { panelEl, onShow, onHide });

  // Apply initial mobile class if needed.
  if (isMobile()) panelEl.classList.add('panel--mobile');

  // Apply saved visibility (default = hidden).
  const saved = loadVisibility();
  const isVisible = saved[id] === true; // strict — undefined means hidden
  if (isVisible) showPanel(id, /*skipSave=*/true);
  else hidePanel(id, /*skipSave=*/true);
}

export function showPanel(id, skipSave = false) {
  const r = registered.get(id);
  if (!r) return;
  r.panelEl.style.display = '';
  dock.hide(id);
  r.onShow?.();
  if (!skipSave) saveVisibility(id, true);
}

export function hidePanel(id, skipSave = false) {
  const r = registered.get(id);
  if (!r) return;
  r.panelEl.style.display = 'none';
  dock.show(id);
  r.onHide?.();
  if (!skipSave) saveVisibility(id, false);
}

onViewportChange(() => {
  const mobile = isMobile();
  for (const [, r] of registered) {
    r.panelEl.classList.toggle('panel--mobile', mobile);
  }
});
