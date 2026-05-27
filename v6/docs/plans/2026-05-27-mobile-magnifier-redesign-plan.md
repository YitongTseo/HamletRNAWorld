# Focus-page redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `viewer/focus.js` into modules, default both viewports to a clean view with a corner-dock reopen pattern for all overlays, add a draggable circular magnifier lens showing the neural x-ray, scale the camera on mobile, and fix the DPR bug that makes the drifting Hamlet text fuzzy.

**Architecture:** Decompose the 1932-line `focus.js` into a `viewer/focus/` directory of small ES modules that communicate through an `index.js` entrypoint. Add three new infrastructure modules (`responsive.js`, `state.js`, `panel-chrome.js` + `dock.js`) that other panels register with. Add one new feature module (`magnifier.js`) that reuses an extracted `xray-render.js`. The simulation, server routes, and other viewer pages are not touched.

**Tech Stack:** Vanilla ES modules, Three.js 0.160 (already loaded via importmap), Canvas 2D, Pointer Events API, `localStorage`. No build step. No new dependencies. Python sim/server unchanged.

**Spec:** `/home/web/HamletRNAWorld/v6/docs/specs/2026-05-27-mobile-magnifier-redesign.md`

**Tests during refactor:**
- `python tests/test_determinism.py` — must pass throughout (no sim changes).
- `python tests/test_smoke_multi.py` — **SKIPPED for the duration of this redesign.** It fails on the baseline (`main` at commit `0f875a61`) with `AssertionError: Alice ate nothing in 5400 ticks`. This is a pre-existing regression unrelated to the viewer-only work in this plan. To be investigated after the redesign ships — see memory `project-smoke-test-regression`. Implementers: do NOT run this test as part of per-task verification; do NOT attempt to fix it as part of this plan.
- After each module extraction: `node --check viewer/focus/<file>.js` for syntax, plus boot the dev server and `curl -fsS http://127.0.0.1:8001/focus/Alice` to verify HTML serves.
- Visual verification is manual — see end of plan for the checklist.

**Dev server convention:** Run the dev server on port **8001** so it doesn't collide with the systemd-managed production app on 8000:
```bash
cd /home/web/HamletRNAWorld/v6
WORMLET_DEBUG_SECRET=devtest /home/web/.venv/bin/python main.py --host 127.0.0.1 --port 8001
```
Stop it with Ctrl-C when done. Do NOT touch the systemd `wormlet-app.service` while developing.

---

## Phase 1 — Behavior-preserving refactor of focus.js

The goal of this phase is zero visual change. Every commit must boot the server and pass the manual smoke "open `/focus/Alice`, see worm + panels exactly as before."

### Task 1: Create the module shell and entrypoint switch

**Files:**
- Create: `viewer/focus/index.js`
- Modify: `viewer/focus.html` (script tag)

- [ ] **Step 1: Create the directory and a stub index.js**

```bash
mkdir -p /home/web/HamletRNAWorld/v6/viewer/focus
```

Create `viewer/focus/index.js` with the entire current contents of `viewer/focus.js`, prepended with a single comment line `// entrypoint — modules will be extracted from this file one at a time`. This is literally `cp viewer/focus.js viewer/focus/index.js` followed by adding the comment at line 1.

- [ ] **Step 2: Update focus.html to load the new entrypoint**

In `viewer/focus.html` change:
```html
<script type="module" src="/static/focus.js?v=2"></script>
```
to:
```html
<script type="module" src="/static/focus/index.js?v=3"></script>
```

- [ ] **Step 3: Verify syntax and boot**

```bash
node --check /home/web/HamletRNAWorld/v6/viewer/focus/index.js
```
Expected: no output (success).

Then boot the dev server (see "Dev server convention" above), `curl -fsS http://127.0.0.1:8001/focus/Alice | grep 'focus/index.js'`. Expected: the script tag line printed.

Open `http://127.0.0.1:8001/focus/Alice` in a browser. Expected: identical to current production — worm renders, all panels appear when their keys ('c', 'n', etc.) are pressed, no console errors.

- [ ] **Step 4: Run Python tests**

```bash
cd /home/web/HamletRNAWorld/v6 && /home/web/.venv/bin/python tests/test_determinism.py && /home/web/.venv/bin/python tests/test_smoke_multi.py
```
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/index.js v6/viewer/focus.html
git rm v6/viewer/focus.js   # the old file is now superseded
git commit -m "v6/viewer: move focus.js to focus/index.js (no behavior change)"
```

### Task 2: Extract `three-scene.js`

**Files:**
- Create: `viewer/focus/three-scene.js`
- Modify: `viewer/focus/index.js`

- [ ] **Step 1: Identify and copy the Three.js setup block**

In `viewer/focus/index.js`, find the block that imports `THREE`, sets up the `WebGLRenderer`, `Scene`, `EffectComposer`, `OrthographicCamera`, lights, and the resize handler (approximately lines 1–80 of the original file, including line 14 `renderer.setPixelRatio(...)` and line 41 `new THREE.Vector2(...)`).

Copy that block into a new file `viewer/focus/three-scene.js`. At the top of the new file, keep the `import * as THREE from 'three';` etc. exactly as written. At the bottom of the new file, add an `export` clause for everything other code needs:

```js
export { renderer, scene, camera, composer, resize };
```

(Use whatever names the original code defines. Read the original lines first; do not assume.)

- [ ] **Step 2: Replace the block in index.js with an import**

In `viewer/focus/index.js`, delete the Three.js setup block you just copied and replace it with:

```js
import { renderer, scene, camera, composer, resize } from './three-scene.js';
```

Place this import near the top of the file, after the importmap-resolved `import * as THREE from 'three'` line (which can stay in `index.js` if other code uses `THREE.*` directly).

- [ ] **Step 3: Syntax check and boot test**

```bash
node --check /home/web/HamletRNAWorld/v6/viewer/focus/three-scene.js
node --check /home/web/HamletRNAWorld/v6/viewer/focus/index.js
```
Expected: no output.

Restart the dev server and reload `/focus/Alice`. Expected: identical to before, no console errors.

- [ ] **Step 4: Run Python tests**

```bash
cd /home/web/HamletRNAWorld/v6 && /home/web/.venv/bin/python tests/test_determinism.py
```

- [ ] **Step 5: Commit**

```bash
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/three-scene.js v6/viewer/focus/index.js
git commit -m "v6/viewer: extract three-scene.js from focus/index.js"
```

### Task 3: Extract `worm-render.js`

**Files:**
- Create: `viewer/focus/worm-render.js`
- Modify: `viewer/focus/index.js`

- [ ] **Step 1: Identify worm-rendering code**

Find functions in `index.js` that build the worm body geometry, update it each tick, and color neurons by firing state. These are the functions that talk to `THREE.Mesh`/`THREE.BufferGeometry` for the worm shape (NOT the 2D x-ray or panel rendering). Read `index.js` carefully — these functions typically have names like `buildWormGeometry`, `updateWormPositions`, `colorNeurons`, or are nested in a class.

- [ ] **Step 2: Move them to `worm-render.js`**

Create `viewer/focus/worm-render.js`. Import what they need from `./three-scene.js`:
```js
import { scene, camera } from './three-scene.js';
import * as THREE from 'three';
```
Move the worm-rendering functions in. Add named exports for what `index.js` calls externally.

- [ ] **Step 3: Replace in index.js with imports**

Remove the moved code from `index.js`; add `import { … } from './worm-render.js';` for the names you exported.

- [ ] **Step 4: Syntax check both files**

```bash
node --check /home/web/HamletRNAWorld/v6/viewer/focus/worm-render.js
node --check /home/web/HamletRNAWorld/v6/viewer/focus/index.js
```
Expected: no output.

- [ ] **Step 5: Boot test**

Restart the dev server and reload `/focus/Alice`. Expected: worm renders and animates identically to before; no console errors.

- [ ] **Step 6: Run determinism test**

```bash
cd /home/web/HamletRNAWorld/v6 && /home/web/.venv/bin/python tests/test_determinism.py
```

- [ ] **Step 7: Commit**

```bash
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/worm-render.js v6/viewer/focus/index.js
git commit -m "v6/viewer: extract worm-render.js from focus/index.js"
```

### Task 4: Extract `text-canvas.js`

**Files:**
- Create: `viewer/focus/text-canvas.js`
- Modify: `viewer/focus/index.js`

- [ ] **Step 1: Move text/smells code**

Move into `viewer/focus/text-canvas.js`:
- The `textcanvas` and `tctx` declarations (original lines ~1231-1235).
- The `resizeTextCanvas` function (original lines ~1234).
- `drawTextCanvas`, `drawSmells`, `drawPcaPopup` (we'll split PCA out later in Task 8 — for now they live together).
- `worldToScreen` and its helper (original lines ~884-888).

Export everything `index.js` needs:
```js
export { textcanvas, tctx, resizeTextCanvas, drawTextCanvas, worldToScreen };
```

- [ ] **Step 2: Update imports in index.js**

```js
import { resizeTextCanvas, drawTextCanvas } from './text-canvas.js';
```

Other modules will import from `text-canvas.js` as needed (for `worldToScreen` etc.) — don't pre-emptively wire those; we'll do it when extracting consumers.

- [ ] **Step 3: Syntax check, boot, manual verify, commit**

NOTE: this task does NOT yet fix the DPR bug. We're doing pure extraction. The text will still be fuzzy until Phase 2.

Commit message:
```
v6/viewer: extract text-canvas.js from focus/index.js
```

### Task 5: Extract `xray-render.js` then `network-panel.js`

This is two extractions in one task because `network-panel.js` uses the x-ray code, and we want the shared `xray-render.js` to exist before being used in two places.

**Files:**
- Create: `viewer/focus/xray-render.js`
- Create: `viewer/focus/network-panel.js`
- Modify: `viewer/focus/index.js`

- [ ] **Step 1: Extract pure x-ray drawing into `xray-render.js`**

Find `drawXRayCanvas` (original line ~1535) and its helpers (`_xrayBuf` at line ~1437, the NEURON_CLASS_PALETTE at ~1495). Move them into `viewer/focus/xray-render.js`. Export:
```js
export { drawXRay, NEURON_CLASS_PALETTE };
```

Rename `drawXRayCanvas` to `drawXRay` and have it take an explicit `(ctx, screenRect, opts)` signature instead of reaching for the module-global `netcanvas`. This is the reusable form the magnifier will need later.

`screenRect` is `{ x, y, w, h }` in CSS pixels — for now, callers pass `{ x: 0, y: 0, w: NET_W, h: NET_H }` to preserve behavior.

`opts` is `{ xrayLabelsVisible: boolean }`.

- [ ] **Step 2: Extract the rest of the network panel into `network-panel.js`**

Move `drawNetworkCanvas` / `drawGraphCanvas` (the legacy graph view), `netcanvas` and its `ctx`, the DPR setup at lines ~1225-1228, and the toggle handlers. Import the x-ray draw fn:
```js
import { drawXRay, NEURON_CLASS_PALETTE } from './xray-render.js';
```

Export everything `index.js` calls:
```js
export { netcanvas, ctx, drawNetworkPanel, toggleXrayMode };
```

`drawNetworkPanel` is the dispatcher that calls either `drawXRay(ctx, {x:0,y:0,w:NET_W,h:NET_H}, {xrayLabelsVisible})` or the legacy graph draw based on `xrayMode`.

- [ ] **Step 3: Update index.js imports and main loop**

In `index.js`, replace the moved code with:
```js
import { drawNetworkPanel } from './network-panel.js';
```
The main render loop's call to `drawXRayCanvas()` / `drawGraphCanvas()` becomes `drawNetworkPanel()`.

- [ ] **Step 4: Syntax check both new files**

```bash
node --check /home/web/HamletRNAWorld/v6/viewer/focus/xray-render.js
node --check /home/web/HamletRNAWorld/v6/viewer/focus/network-panel.js
```

- [ ] **Step 5: Boot test, especially x-ray view**

Boot dev server. Open `/focus/Alice`. Press `n` to toggle the network panel, press `x` to flip to x-ray, press `l` for labels. All three must work identically to before.

- [ ] **Step 6: Run Python tests, commit**

```bash
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/xray-render.js v6/viewer/focus/network-panel.js v6/viewer/focus/index.js
git commit -m "v6/viewer: extract xray-render.js and network-panel.js"
```

### Task 6: Extract `chemo-panel.js`

**Files:**
- Create: `viewer/focus/chemo-panel.js`
- Modify: `viewer/focus/index.js`

- [ ] **Step 1: Move chemosensory rendering**

Identify all code that touches `#chemosensoryPanel` — the `chemosensoryPanel` DOM element, the function that populates it with neuron labels and color swatches, the toggle handler for the `c` key. Move into `viewer/focus/chemo-panel.js`. Export:
```js
export { chemosensoryPanel, drawChemoPanel, toggleChemo };
```

- [ ] **Step 2: Update index.js import**

```js
import { drawChemoPanel } from './chemo-panel.js';
```
Replace the moved call in the main loop with `drawChemoPanel()`.

- [ ] **Step 3: Boot test**

Open `/focus/Alice`, press `c` to toggle the chemo panel. Verify it appears and disappears identically.

- [ ] **Step 4: Commit**

```
v6/viewer: extract chemo-panel.js from focus/index.js
```

### Task 7: Extract `radar-panel.js`

**Files:**
- Create: `viewer/focus/radar-panel.js`
- Modify: `viewer/focus/index.js`

- [ ] **Step 1: Move radar code**

Move the `drawRadar` / emotion radar code (around lines 686-823 of original `focus.js`), the `radarcanvas` setup and DPR scaling (lines 1290-1293), and the `e` key handler. Export:
```js
export { radarcanvas, drawRadar, toggleRadar };
```

- [ ] **Step 2: Update index.js**

```js
import { drawRadar } from './radar-panel.js';
```

- [ ] **Step 3: Boot test**

Press `e` — radar panel toggles correctly. Verify the radar renders the emotion data (the original "rebuild corpus_pca.json with emotions" warning text is OK to see if your local PCA cache doesn't include emotions).

- [ ] **Step 4: Commit**

```
v6/viewer: extract radar-panel.js from focus/index.js
```

### Task 8: Extract `pca-popup.js`

**Files:**
- Create: `viewer/focus/pca-popup.js`
- Modify: `viewer/focus/text-canvas.js`
- Modify: `viewer/focus/index.js`

- [ ] **Step 1: Move PCA popup out of text-canvas.js**

Move `drawPcaPopup` (originally part of `drawTextCanvas` in original `focus.js` around lines 1014-1140) into `viewer/focus/pca-popup.js`. Import what it needs:
```js
import { worldToScreen } from './text-canvas.js';
```

Export:
```js
export { drawPcaPopup };
```

- [ ] **Step 2: Update text-canvas.js**

In `text-canvas.js`, replace the inline `drawPcaPopup` call inside `drawTextCanvas` with an imported one:
```js
import { drawPcaPopup } from './pca-popup.js';
```
Keep the hover-detection logic (finding nearest word, distance threshold of 80) inside `drawTextCanvas`. Only the actual popup drawing moves out.

- [ ] **Step 3: Boot test**

Open `/focus/Alice`. Move mouse near a word — the small PCA popup should appear next to it identically to before.

- [ ] **Step 4: Commit**

```
v6/viewer: extract pca-popup.js from text-canvas.js
```

### Task 9: Extract `focus.css` and slim down `focus.html`

**Files:**
- Create: `viewer/focus.css`
- Modify: `viewer/focus.html`

- [ ] **Step 1: Move the `<style>` block to focus.css**

Copy everything inside `<style>...</style>` (lines 7-66 of current `focus.html`) into a new file `viewer/focus.css`. Do not modify CSS rules — pure extraction.

- [ ] **Step 2: Replace `<style>` with a `<link>` in focus.html**

In `viewer/focus.html`, delete the `<style>...</style>` block and add inside `<head>`:
```html
<link rel="stylesheet" href="/static/focus.css?v=1">
```

- [ ] **Step 3: Boot test**

Open `/focus/Alice`. All overlays must look pixel-identical to before. Use browser DevTools to confirm the CSS is loading from `/static/focus.css`.

- [ ] **Step 4: Commit**

```
v6/viewer: extract focus.css from focus.html
```

**End of Phase 1.** At this point `focus.js` is gone, replaced by `viewer/focus/index.js` + 7 other modules + a separate `focus.css`. Behavior is identical. The refactor is now reviewable as 9 small commits.

---

## Phase 2 — Fuzzy text fix

### Task 10: DPR-correct `text-canvas.js`

**Files:**
- Modify: `viewer/focus/text-canvas.js`

- [ ] **Step 1: Update `resizeTextCanvas`**

In `viewer/focus/text-canvas.js`, replace the existing resize logic with:

```js
function resizeTextCanvas() {
  const dpr = window.devicePixelRatio || 1;
  textcanvas.width  = window.innerWidth  * dpr;
  textcanvas.height = window.innerHeight * dpr;
  textcanvas.style.width  = window.innerWidth  + 'px';
  textcanvas.style.height = window.innerHeight + 'px';
  tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
```

- [ ] **Step 2: Replace `textcanvas.width`/`textcanvas.height` references with `window.innerWidth/Height`**

Grep within the module:
```bash
grep -n 'textcanvas\.\(width\|height\)' /home/web/HamletRNAWorld/v6/viewer/focus/text-canvas.js
```

For each match, change `textcanvas.width` → `window.innerWidth` and `textcanvas.height` → `window.innerHeight`. The same applies to `pca-popup.js` if it reads those properties.

The `worldToScreen` function (originally lines 884-888 of focus.js, now in `text-canvas.js`) must also be updated:
```js
function worldToScreen(wx, wy) {
  const nx = (wx - camera.left)   / (camera.right - camera.left);
  const ny = (wy - camera.bottom) / (camera.top   - camera.bottom);
  return [nx * window.innerWidth, (1 - ny) * window.innerHeight];
}
```
(Use the project's existing sign convention — read what's there before substituting.)

- [ ] **Step 3: Boot test on highest-DPR display available**

Open `/focus/Alice` in a browser on a Retina display (or Chrome DevTools → device toolbar → set device pixel ratio to 2). The drifting Hamlet words must render crisply, not blurry. The PCA popup must position correctly next to the nearest word (test by hovering near several words).

- [ ] **Step 4: Run determinism test**

```bash
cd /home/web/HamletRNAWorld/v6 && /home/web/.venv/bin/python tests/test_determinism.py
```
Expected: passes (no sim changes).

- [ ] **Step 5: Commit**

```bash
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/text-canvas.js v6/viewer/focus/pca-popup.js
git commit -m "v6/viewer: DPR-correct text-canvas for sharp Hamlet text on Retina"
```

---

## Phase 3 — Responsive infrastructure

### Task 11: Create `responsive.js`

**Files:**
- Create: `viewer/focus/responsive.js`

- [ ] **Step 1: Write the module**

```js
// viewer/focus/responsive.js
export const MOBILE_BREAKPOINT = 768;
export function isMobile() { return window.innerWidth < MOBILE_BREAKPOINT; }
export function cameraFrustumScale() { return isMobile() ? 0.55 : 1.0; }

const listeners = new Set();
export function onViewportChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }

let pending = null;
window.addEventListener('resize', () => {
  if (pending) clearTimeout(pending);
  pending = setTimeout(() => {
    pending = null;
    for (const fn of listeners) fn();
  }, 100);
});
```

- [ ] **Step 2: Syntax check**

```bash
node --check /home/web/HamletRNAWorld/v6/viewer/focus/responsive.js
```

- [ ] **Step 3: Commit**

```bash
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/responsive.js
git commit -m "v6/viewer: add responsive.js (isMobile, camera scale, resize events)"
```

### Task 12: Wire camera frustum scaling into `three-scene.js`

**Files:**
- Modify: `viewer/focus/three-scene.js`

- [ ] **Step 1: Import and use `cameraFrustumScale`**

In `three-scene.js`, find where the orthographic camera frustum is set (`camera.left`, `camera.right`, `camera.top`, `camera.bottom`). Multiply each by `cameraFrustumScale()`:

```js
import { cameraFrustumScale, onViewportChange } from './responsive.js';

function updateCameraFrustum() {
  const s = cameraFrustumScale();
  const aspect = window.innerWidth / window.innerHeight;
  // Whatever the existing base half-width is. Read existing code first.
  const halfW = BASE_HALF_WIDTH * s;
  const halfH = halfW / aspect;
  camera.left   = -halfW;
  camera.right  =  halfW;
  camera.top    =  halfH;
  camera.bottom = -halfH;
  camera.updateProjectionMatrix();
}

updateCameraFrustum();
onViewportChange(updateCameraFrustum);
```

If the existing camera setup uses something other than a symmetric frustum derived from a base half-width, adapt this pattern to match — the principle is: multiply the existing frustum extents by `cameraFrustumScale()` and call `updateProjectionMatrix()`.

The existing `resize()` function (which calls `renderer.setSize`) should also call `updateCameraFrustum()`.

- [ ] **Step 2: Boot test**

Open `/focus/Alice` on desktop. Should look identical to before (scale = 1.0).

Open Chrome DevTools, device toolbar, switch to "iPhone 12 Pro" (390×844). Reload. The worm should appear ~1.8× larger and the Hamlet text should span most of the viewport width.

Resize the desktop browser narrow (below 768px) — camera should zoom in. Widen it again — should zoom out.

- [ ] **Step 3: Run determinism test, commit**

```bash
cd /home/web/HamletRNAWorld/v6 && /home/web/.venv/bin/python tests/test_determinism.py
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/three-scene.js
git commit -m "v6/viewer: scale camera frustum 0.55x on mobile for readable text"
```

---

## Phase 4 — State, panel chrome, and dock

### Task 13: Create `state.js`

**Files:**
- Create: `viewer/focus/state.js`

- [ ] **Step 1: Write the module**

```js
// viewer/focus/state.js
const wormName = decodeURIComponent(location.pathname.replace(/^\/focus\//, '').replace(/\/$/, ''));
const visKey = `wormlet:focus:${wormName}:visible`;
const magKey = `wormlet:focus:${wormName}:magnifier-pos`;

export function loadVisibility() {
  try { return JSON.parse(localStorage.getItem(visKey)) || {}; }
  catch { return {}; }
}
export function saveVisibility(id, isVisible) {
  const cur = loadVisibility();
  cur[id] = isVisible;
  try { localStorage.setItem(visKey, JSON.stringify(cur)); } catch {}
}
export function loadMagnifierPos() {
  try { return JSON.parse(localStorage.getItem(magKey)); }
  catch { return null; }
}
export function saveMagnifierPos(pos) {
  try { localStorage.setItem(magKey, JSON.stringify(pos)); } catch {}
}
```

- [ ] **Step 2: Syntax check, commit**

```bash
node --check /home/web/HamletRNAWorld/v6/viewer/focus/state.js
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/state.js
git commit -m "v6/viewer: add state.js (per-worm localStorage for panel visibility)"
```

### Task 14: Create `dock.js` and `dock` styles in focus.css

**Files:**
- Create: `viewer/focus/dock.js`
- Modify: `viewer/focus.css`

- [ ] **Step 1: Add dock styles to focus.css**

Append to `viewer/focus.css`:

```css
#dock {
  position: fixed;
  bottom: 8px; right: 10px;
  display: flex;
  gap: 6px;
  z-index: 30;
}
#dock button {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(102, 255, 153, 0.4);
  color: #6f9;
  font: 14px ui-monospace, SFMono-Regular, Menlo, monospace;
  text-shadow: 0 0 4px #0f0a;
  width: 48px; height: 28px;
  padding: 0;
  cursor: pointer;
  opacity: 0.75;
  transition: opacity 0.15s, color 0.15s;
}
#dock button:hover { opacity: 1.0; color: #afa; }
#dock button:active { color: #fff; }
```

- [ ] **Step 2: Write `dock.js`**

```js
// viewer/focus/dock.js
const dockEl = document.createElement('div');
dockEl.id = 'dock';
document.body.appendChild(dockEl);

const buttons = new Map(); // id -> { btn, onOpen }

export function register(id, glyph, label, onOpen) {
  const btn = document.createElement('button');
  btn.textContent = glyph;
  btn.title = label;
  btn.dataset.dockId = id;
  btn.addEventListener('click', () => {
    hide(id);
    onOpen();
  });
  buttons.set(id, { btn, onOpen });
  // Initially hidden — panel-chrome decides whether to show based on saved state.
  btn.style.display = 'none';
  dockEl.appendChild(btn);
}

export function show(id) {
  const entry = buttons.get(id);
  if (entry) entry.btn.style.display = '';
}
export function hide(id) {
  const entry = buttons.get(id);
  if (entry) entry.btn.style.display = 'none';
}
```

- [ ] **Step 3: Syntax check, commit**

```bash
node --check /home/web/HamletRNAWorld/v6/viewer/focus/dock.js
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/dock.js v6/viewer/focus.css
git commit -m "v6/viewer: add dock.js (bottom-right glyph strip for reopening panels)"
```

### Task 15: Create `panel-chrome.js`

**Files:**
- Create: `viewer/focus/panel-chrome.js`
- Modify: `viewer/focus.css` (add `.panel-close-btn` and `.panel--mobile` styles)

- [ ] **Step 1: Add chrome styles to focus.css**

Append:

```css
.panel-close-btn {
  position: absolute; top: 2px; right: 4px;
  width: 16px; height: 16px;
  background: transparent;
  border: none;
  color: inherit;
  font: 14px ui-monospace, monospace;
  cursor: pointer;
  opacity: 0.6;
  padding: 0;
  line-height: 14px;
  z-index: 1;
}
.panel-close-btn:hover { opacity: 1.0; }

.panel--mobile {
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  top: auto !important;
  width: 100vw !important;
  max-height: 75vh !important;
  border-radius: 12px 12px 0 0 !important;
}
```

- [ ] **Step 2: Write `panel-chrome.js`**

```js
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
```

- [ ] **Step 3: Syntax check, commit**

```bash
node --check /home/web/HamletRNAWorld/v6/viewer/focus/panel-chrome.js
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/panel-chrome.js v6/viewer/focus.css
git commit -m "v6/viewer: add panel-chrome.js (X buttons, mobile bottom-sheet, state)"
```

---

## Phase 5 — Wire existing panels through chrome, default closed

### Task 16: Register all existing panels with chrome, remove keyboard-only toggles

**Files:**
- Modify: `viewer/focus/network-panel.js`
- Modify: `viewer/focus/chemo-panel.js`
- Modify: `viewer/focus/radar-panel.js`
- Modify: `viewer/focus/pca-popup.js`
- Modify: `viewer/focus/index.js`

The keyboard shortcuts (`n`, `c`, `e`) should still work — but they now call the same `showPanel`/`hidePanel` from `panel-chrome.js`, so state stays consistent.

- [ ] **Step 1: Register network panel**

In `viewer/focus/network-panel.js`, after the `netcanvas` element is obtained, add:

```js
import * as chrome from './panel-chrome.js';

// At module init, after netcanvas is created:
chrome.register({
  id: 'network',
  glyph: 'o-+-o',
  label: 'neural network / x-ray',
  panelEl: netcanvas,
  onShow: () => { /* nothing — drawing happens unconditionally in render loop */ },
  onHide: () => {},
});
```

Update the existing `n` key handler to:
```js
import { showPanel, hidePanel } from './panel-chrome.js';
// in the keydown handler for 'n':
if (getComputedStyle(netcanvas).display === 'none') showPanel('network');
else hidePanel('network');
```

In the main render loop (in `index.js`), guard the draw call:
```js
if (getComputedStyle(netcanvas).display !== 'none') drawNetworkPanel();
```

- [ ] **Step 2: Register chemo panel**

In `viewer/focus/chemo-panel.js`:

```js
chrome.register({
  id: 'chemo',
  glyph: '<~~~>',
  label: 'chemosensory neurons',
  panelEl: chemosensoryPanel,
});
```

Update `c` keydown handler to use `showPanel('chemo')` / `hidePanel('chemo')`.

In `index.js`, guard the `drawChemoPanel()` call similarly.

- [ ] **Step 3: Register radar panel**

In `viewer/focus/radar-panel.js`:

```js
chrome.register({
  id: 'radar',
  glyph: '<*+*>',
  label: 'emotion radar',
  panelEl: radarcanvas,
});
```

Update `e` keydown handler. Guard `drawRadar()` in main loop.

- [ ] **Step 4: Register PCA popup**

The PCA popup is drawn into the same `#textcanvas` as the words, so it doesn't have its own DOM element. To make it dock-controllable, add a module-level flag in `pca-popup.js`:

```js
let pcaVisible = false;
export function setPcaVisible(v) { pcaVisible = v; }
export function isPcaVisible() { return pcaVisible; }
```

Add to `pca-popup.js`:
```js
import * as chrome from './panel-chrome.js';
import { loadVisibility, saveVisibility } from './state.js';

// Create a 1x1 hidden div to satisfy panel-chrome's requirement of a panelEl.
const sentinel = document.createElement('div');
sentinel.style.cssText = 'position:fixed; width:1px; height:1px; pointer-events:none; opacity:0;';
document.body.appendChild(sentinel);

chrome.register({
  id: 'pca',
  glyph: '[x,y]',
  label: 'word PCA popup',
  panelEl: sentinel,
  onShow: () => { pcaVisible = true; },
  onHide: () => { pcaVisible = false; },
});

// Initialize flag from saved state on load.
pcaVisible = loadVisibility().pca === true;
```

In `text-canvas.js`, change the `drawPcaPopup` call to be conditional:
```js
import { isPcaVisible } from './pca-popup.js';
// in drawTextCanvas, around the existing PCA call:
if (isPcaVisible() && nearestWord && minDist < 80 && pcaData) {
  drawPcaPopup(nearestWord.word);
}
```

- [ ] **Step 5: Register hud/help/wormtitle/nav**

These are static HTML elements in `focus.html`. They don't need their own modules. Add to `index.js`:

```js
import * as chrome from './panel-chrome.js';

chrome.register({
  id: 'titleNav',
  glyph: '>><>><',
  label: 'worm title + nav',
  panelEl: document.getElementById('wormtitle'),
});
chrome.register({
  id: 'help',
  glyph: '[ ? ]',
  label: 'keyboard legend',
  panelEl: document.getElementById('help'),
});
```

(The `#nav` element shares the title's life — we group them under one `titleNav` dock entry. Wrap them in a parent `<div>` in `focus.html` so a single `panelEl` shows/hides both. See sub-step.)

**Sub-step:** in `viewer/focus.html`, replace:
```html
<div id="wormtitle"></div>
<div id="nav">…</div>
```
with:
```html
<div id="title-nav-wrap">
  <div id="wormtitle"></div>
  <div id="nav">…</div>
</div>
```
And in `viewer/focus.css`, add `#title-nav-wrap { position: fixed; top: 0; left: 0; right: 0; pointer-events: none; }` (so its children still position relative to viewport). Adjust `#nav` and `#wormtitle` to use `pointer-events: auto` so links remain clickable.

Then in `index.js`, use:
```js
panelEl: document.getElementById('title-nav-wrap'),
```

- [ ] **Step 6: Boot test the whole flow**

Open `/focus/Alice` in a fresh incognito window (no localStorage). Expected:
- Page loads with **only** the worm and drifting text visible.
- Dock at bottom-right shows 5 glyph buttons: `o-+-o`, `<~~~>`, `<*+*>`, `[x,y]`, `[ ? ]`, `>><>><` (6 actually).
- Click `o-+-o` → network panel appears with x-ray view; its glyph disappears from dock.
- Click the panel's ✕ → panel hides; glyph reappears in dock.
- Press `n` key → toggles network panel (state stays consistent with dock).
- Reload page → previously-open panels stay open.
- Open DevTools → Application → Local Storage → see `wormlet:focus:Alice:visible` key with your config.

- [ ] **Step 7: Run Python tests, commit**

```bash
cd /home/web/HamletRNAWorld/v6 && /home/web/.venv/bin/python tests/test_determinism.py && /home/web/.venv/bin/python tests/test_smoke_multi.py
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/
git add v6/viewer/focus.html v6/viewer/focus.css
git commit -m "v6/viewer: default panels hidden, dock+chrome for close/reopen, keyboard still works"
```

---

## Phase 6 — Magnifier

### Task 17: Create `magnifier.js`

**Files:**
- Create: `viewer/focus/magnifier.js`
- Modify: `viewer/focus.css`
- Modify: `viewer/focus/index.js`

- [ ] **Step 1: Add magnifier styles to focus.css**

Append:

```css
.magnifier {
  position: fixed;
  border: 2px solid rgba(100, 200, 255, 0.6);
  border-radius: 50%;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
  cursor: grab;
  z-index: 35;
  touch-action: none; /* prevent browser pan/zoom intercepting drag */
}
.magnifier.dragging { cursor: grabbing; }
.magnifier canvas {
  display: block;
  width: 100%; height: 100%;
  border-radius: 50%;
}
.magnifier-close {
  position: absolute; top: -8px; right: -8px;
  width: 24px; height: 24px;
  border-radius: 50%;
  background: #000;
  color: #fff;
  border: 1px solid rgba(100, 200, 255, 0.6);
  font-size: 14px;
  line-height: 22px;
  padding: 0;
  cursor: pointer;
  z-index: 36;
  font-family: ui-monospace, monospace;
}
```

- [ ] **Step 2: Write `magnifier.js`**

```js
// viewer/focus/magnifier.js
import { drawXRay } from './xray-render.js';
import { worldToScreen } from './text-canvas.js';
import * as chrome from './panel-chrome.js';
import { loadMagnifierPos, saveMagnifierPos } from './state.js';
import { isMobile, onViewportChange } from './responsive.js';

let lensEl, lensCanvas, lensCtx, closeBtn;
let xrayLabelsVisible = false;

function diameter() {
  const d = Math.round(Math.min(window.innerWidth, window.innerHeight) * 0.30);
  return Math.max(120, Math.min(360, d));
}

function build() {
  lensEl = document.createElement('div');
  lensEl.className = 'magnifier';
  lensCanvas = document.createElement('canvas');
  closeBtn = document.createElement('button');
  closeBtn.className = 'magnifier-close';
  closeBtn.textContent = '✕';
  closeBtn.setAttribute('aria-label', 'close magnifier');
  closeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    chrome.hidePanel('magnifier');
  });
  lensEl.appendChild(lensCanvas);
  lensEl.appendChild(closeBtn);
  document.body.appendChild(lensEl);

  applySize();
  applyInitialPosition();
  wireDrag();
}

function applySize() {
  const d = diameter();
  lensEl.style.width = d + 'px';
  lensEl.style.height = d + 'px';
  const dpr = window.devicePixelRatio || 1;
  lensCanvas.width  = d * dpr;
  lensCanvas.height = d * dpr;
  lensCtx = lensCanvas.getContext('2d');
  lensCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

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
  // Clamp to viewport.
  x = Math.max(0, Math.min(window.innerWidth  - d, x));
  y = Math.max(0, Math.min(window.innerHeight - d, y));
  lensEl.style.left = x + 'px';
  lensEl.style.top  = y + 'px';
}

function wireDrag() {
  let dragging = false;
  let offsetX = 0, offsetY = 0;
  lensEl.addEventListener('pointerdown', (e) => {
    if (e.target === closeBtn) return;
    dragging = true;
    lensEl.classList.add('dragging');
    lensEl.setPointerCapture(e.pointerId);
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

export function render() {
  if (!lensEl || lensEl.style.display === 'none') return;
  const rect = lensEl.getBoundingClientRect();
  // Clear, then draw x-ray scoped to the lens rect.
  lensCtx.clearRect(0, 0, rect.width, rect.height);
  drawXRay(lensCtx, { x: rect.left, y: rect.top, w: rect.width, h: rect.height }, {
    xrayLabelsVisible,
    worldToScreen,
  });
}

export function setXrayLabelsVisible(v) { xrayLabelsVisible = v; }

// Init: build DOM, register with panel-chrome (default hidden), respond to viewport changes.
build();
chrome.register({
  id: 'magnifier',
  glyph: '(°o°)',
  label: 'neural x-ray magnifier',
  panelEl: lensEl,
});
onViewportChange(() => { applySize(); applyInitialPosition(); });
```

- [ ] **Step 3: Update `xray-render.js` to accept the lens-rect-relative call**

The existing `drawXRay(ctx, screenRect, opts)` signature needs to handle the case where the caller is the magnifier (rendering into a small canvas positioned at screenRect on the page) vs the network panel (rendering into a fixed-size canvas).

The key change: the drawing must project neuron body coords from world space, through `worldToScreen` (which gives page-relative CSS pixel coords), then translate by `-screenRect.x, -screenRect.y` so they land inside the lens's local canvas coordinate system.

Update `drawXRay` signature in `viewer/focus/xray-render.js` to accept `opts.worldToScreen`:

```js
export function drawXRay(ctx, screenRect, opts) {
  const { xrayLabelsVisible, worldToScreen } = opts;
  // For each neuron body coord (nx, ny) projected into world:
  //   const [px, py] = worldToScreen(nx, ny);
  //   const lx = px - screenRect.x;
  //   const ly = py - screenRect.y;
  //   ctx.fillRect(lx - r, ly - r, r*2, r*2);
  // ...existing neuron-color/firing logic.
}
```

The network-panel caller passes `worldToScreen` too (or a synthetic one for its fixed-size canvas — see sub-step):

In `network-panel.js`, when calling `drawXRay`, pass a `worldToScreen` that maps the worm's full body extent into the fixed 500×360 panel rather than the page. This preserves the existing standalone x-ray panel behavior.

```js
function panelLocalWorldToScreen(wx, wy) {
  // Map the worm body's bounding box into the panel canvas.
  // Use the same mapping the existing drawXRayCanvas did before extraction.
  // …existing math, but returning panel-local (0..NET_W, 0..NET_H) coords.
}
drawXRay(ctx, { x: 0, y: 0, w: NET_W, h: NET_H }, {
  xrayLabelsVisible,
  worldToScreen: panelLocalWorldToScreen,
});
```

- [ ] **Step 4: Wire magnifier into main render loop**

In `viewer/focus/index.js`, in the main render loop, after the network panel draw, add:

```js
import * as magnifier from './magnifier.js';
// …
magnifier.render();
```

Propagate the existing `l` key (xray labels toggle) to **both** the network panel and the magnifier so they stay in sync. The `xrayLabelsVisible` flag now lives in two places (one per consumer), and the key handler updates both:

```js
import * as networkPanel from './network-panel.js'; // already imported earlier
import * as magnifier from './magnifier.js';

let xrayLabelsVisible = false;
// In the existing keydown handler, replace the 'l' branch:
if (e.key === 'l') {
  xrayLabelsVisible = !xrayLabelsVisible;
  networkPanel.setXrayLabelsVisible(xrayLabelsVisible);
  magnifier.setXrayLabelsVisible(xrayLabelsVisible);
}
```

This requires `network-panel.js` to also export a `setXrayLabelsVisible(v)` setter that updates its module-local copy of the flag (which was extracted from the original `let xrayLabelsVisible = false;` at line 1246 of original `focus.js`). Add that export if Task 5 didn't already.

- [ ] **Step 5: Boot test**

Open `/focus/Alice` in a fresh incognito window. Click the `(°o°)` dock glyph → magnifier appears centered. Drag it over the worm. Neurons inside the lens should align with the worm body underneath. Press `l` → neuron labels appear in the lens. Click ✕ → lens disappears, glyph returns to dock. Reload → if magnifier was open, it reopens at its last position.

Repeat on Chrome DevTools mobile emulation (390×844): lens is smaller (~120px), drags with finger.

- [ ] **Step 6: Run Python tests, commit**

```bash
cd /home/web/HamletRNAWorld/v6 && /home/web/.venv/bin/python tests/test_determinism.py
cd /home/web/HamletRNAWorld && git add v6/viewer/focus/magnifier.js v6/viewer/focus/xray-render.js v6/viewer/focus/network-panel.js v6/viewer/focus/index.js v6/viewer/focus.css
git commit -m "v6/viewer: add draggable magnifier lens with neural x-ray overlay"
```

---

## Phase 7 — Final manual verification and version bump

### Task 18: Bump the script version for cache busting and run the full manual checklist

**Files:**
- Modify: `viewer/focus.html`

- [ ] **Step 1: Bump cache-busting query strings**

In `viewer/focus.html`, ensure:
```html
<link rel="stylesheet" href="/static/focus.css?v=2">
<script type="module" src="/static/focus/index.js?v=4"></script>
```

(Increment whichever version was last used. The point is: returning users get fresh files.)

- [ ] **Step 2: Run the full manual checklist**

Boot the dev server. Open in a real browser (not just DevTools emulation if possible) at desktop width, then narrow to mobile width:

1. Default state on first load: only worm + drifting Hamlet text + dock with 6 glyphs visible.
2. Each dock glyph opens its corresponding panel: `o-+-o` → network, `<~~~>` → chemo, `<*+*>` → radar, `[x,y]` → enables PCA hover popup, `[ ? ]` → help legend, `>><>><` → worm title + nav, `(°o°)` → magnifier.
3. Each panel's ✕ closes it and returns its glyph to the dock.
4. Reload page → previously-open panels are still open, previously-closed still closed.
5. Resize browser below 768px → opened panels switch to bottom-sheet layout (cover bottom ~75%); camera zooms in; dock stays bottom-right.
6. Resize back above 768px → panels return to fixed positions; camera zooms back out.
7. Drifting Hamlet text is visibly **sharp** (no fuzz) on both 1× and 2× DPR.
8. Magnifier drags smoothly; neurons inside the lens align with the worm body underneath. Press `l` → labels appear in the lens.
9. Keyboard shortcuts still work: `n` toggles network, `c` chemo, `e` radar, `x` x-ray mode within network panel, `l` labels in both panels.
10. Open on an actual phone via wordswordsworms.org (after deploy) → all interactions work via touch.

If any item fails, fix before proceeding. Re-run `python tests/test_determinism.py` and `test_smoke_multi.py` one final time.

- [ ] **Step 3: Final commit**

```bash
cd /home/web/HamletRNAWorld && git add v6/viewer/focus.html
git commit -m "v6/viewer: bump cache-busting versions for redesign rollout"
```

- [ ] **Step 4: Deploy**

The production app is managed by `wormlet-app.service`. To pick up the new files:

```bash
sudo systemctl restart wormlet-app.service
```

Wait 5 seconds, then `curl -fsS https://wordswordsworms.org/focus/Alice | grep 'focus/index.js'` to confirm the live site is serving the new entrypoint. Spot-check on a phone via the public URL.

---

## Out of scope (do NOT add to this plan)

- Dock auto-hide on inactivity (deferred per spec).
- Pinch-to-zoom magnifier diameter (deferred).
- Two-worm compare magnifier (deferred).
- Redesigning `/`, `/poems`, `/generations`, `/about` (out of spec scope).
- Playwright/headless browser tests (manual verification chosen instead).
- Adding any build step or framework (the project is intentionally zero-build).
