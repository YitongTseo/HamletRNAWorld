# Focus-page redesign: mobile defaults, panel chrome, magnifier lens, sharp text

**Status:** Approved design, ready for implementation planning
**Author:** Yitong + Claude (brainstorm session 2026-05-27)
**Scope:** `viewer/focus.html` + `viewer/focus.js` only. Simulation code (`sim/`, `server/`) untouched.

## Problem

The `/focus/<name>` page works well on desktop but has three issues:

1. **Mobile is unusable.** All overlays (network panel, chemosensory panel, emotion radar, hover-PCA popup, hud, nav, help, title) crowd a small screen. Toggling them requires keyboard shortcuts (`c`, `n`, `x`, `l`, `o`, `e`) that don't exist on phones, so once a panel is up there's no way to dismiss it. The worm is barely visible.
2. **The drifting Hamlet text is fuzzy** on both mobile and desktop. Root cause confirmed: `focus.js:1234-1235` sizes `#textcanvas` in CSS pixels without multiplying by `devicePixelRatio`. The other two canvases (`#netcanvas` and `#radarcanvas`) scale by DPR correctly; only the text overlay is missed. On a Retina screen the browser upscales the 1x canvas to fit the CSS size, blurring everything.
3. **No way to inspect neurons in context.** The x-ray view exists (toggled with `x`) but lives in a separate fixed panel away from the worm body. There's no "look here" interaction.

## Goals

- Both viewports default to a clean view: just the worm + the drifting Hamlet text. Everything else opens via small icons in a corner dock.
- Mobile zooms the camera so the worm and text fill the screen (the simulation is unchanged; only the camera frustum scales).
- All overlays close with an X and reopen from the dock. State persists per-worm in `localStorage`.
- A new draggable circular magnifier lens reveals the neural x-ray overlaid on the worm body wherever the user drags it.
- The fuzzy Hamlet text becomes sharp on all displays.
- Determinism (`tests/test_determinism.py`) and smoke (`tests/test_smoke_multi.py`) tests still pass.

## Non-goals

- Changing the simulation in any way. World coordinates, word placement, sim tick rate, neuron firing — all untouched.
- Mobile-specific text scaling. Text size (18px) stays the same; only the camera frustum changes on mobile.
- Adding a build step or framework dependency. The project is intentionally zero-build vanilla JS + Three.js.
- Restyling the other pages (`/`, `/poems`, `/generations`, `/about`). They're out of scope for this work.

## Design decisions (from brainstorm)

- **Default visibility:** all panels collapsed by default on first visit, both mobile and desktop. Simpler than viewport-dependent defaults.
- **Mobile zoom:** camera frustum only (~0.55× scale), not text scaling. Text is part of the simulation's spatial output.
- **Magnifier:** large drag-only lens with X to close and 🔍 dock icon to reopen. Inside shows x-ray overlay; outside is unchanged (no dimming). On mobile, sized ~30% of viewport min-dimension.
- **Refactor first:** `focus.js` (1932 lines) gets decomposed into modules before features are added, so the refactor is reviewable as a separate behavior-preserving change.

## File layout

```
viewer/focus.html            ~50 lines  (was 94; inline styles moved out)
viewer/focus.css             ~150 lines (extracted from focus.html <style>)
viewer/focus/
  index.js                   ~120 lines entrypoint; wires modules; main render loop
  three-scene.js             ~250 lines Three.js renderer, composer, camera, lights
  worm-render.js             ~400 lines worm body geometry, midline, neuron firing colors
  text-canvas.js             ~250 lines drifting Hamlet words + smells (DPR-correct)
  network-panel.js           ~350 lines neural graph view (#netcanvas)
  xray-render.js             ~200 lines shared x-ray draw fn — used by network AND magnifier
  chemo-panel.js             ~150 lines chemosensory panel (#chemosensoryPanel)
  radar-panel.js             ~150 lines emotion radar (#radarcanvas)
  pca-popup.js               ~80 lines  word-hover PCA popup (was inline in focus.js)
  magnifier.js               ~200 lines NEW — draggable circular lens
  panel-chrome.js            ~120 lines NEW — wraps panels with X + dock integration
  dock.js                    ~80 lines  NEW — bottom-right strip of reopen icons
  responsive.js              ~60 lines  NEW — isMobile(), camera scale, viewport events
  state.js                   ~50 lines  NEW — localStorage save/load per worm
```

Module boundaries are strict: modules talk through `index.js`, not directly to siblings. Exception: `network-panel.js` and `magnifier.js` both import the shared `xray-render.js` (which is the whole point of extracting it).

`focus.html`'s `<script type="module" src="/static/focus.js?v=2">` becomes `<script type="module" src="/static/focus/index.js?v=3">`. Canvas IDs (`#c`, `#textcanvas`, `#netcanvas`, etc.) stay the same.

## Component details

### `responsive.js`

```js
export const MOBILE_BREAKPOINT = 768;
export function isMobile() { return window.innerWidth < MOBILE_BREAKPOINT; }
export function cameraFrustumScale() { return isMobile() ? 0.55 : 1.0; }
```

Emits `viewport-changed` on `window.resize` (debounced ~100ms). `three-scene.js` listens to update camera frustum; `panel-chrome.js` listens to re-clamp panel positions and switch between fixed-position (desktop) and bottom-sheet (mobile) layout.

### `text-canvas.js` — the fuzzy text fix

Old code (the bug):
```js
textcanvas.width  = window.innerWidth;
textcanvas.height = window.innerHeight;
```

New code:
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

`worldToScreen` and related helpers must read `window.innerWidth/Height` (CSS pixels) rather than `textcanvas.width/height` (now physical pixels) so they stay in sync with the new `setTransform`. The change is mechanical: any `textcanvas.width` reference in `focus.js` becomes `window.innerWidth`, same for height.

### `panel-chrome.js`

One exported function:

```js
register({
  id,              // 'network' | 'chemo' | 'radar' | 'magnifier' | ...
  label,           // visible label inside the dock-button tooltip
  icon,            // emoji or SVG used in the dock
  dockCorner,      // 'bottom-right' default; other corners reserved for future
  panelEl,         // DOM element to show/hide
  onShow, onHide,  // optional lifecycle hooks (e.g., start/stop the panel's draw loop)
});
```

Behavior:
- Inserts a 16×16 `✕` button absolutely-positioned in the panel's top-right (color matches the panel's existing accent border).
- On close: sets `panelEl.style.display = 'none'`, calls `dock.show(id)`, calls `state.save(id, false)`, fires `onHide?.()`.
- On dock icon click: reverse.
- On viewport change to mobile: if the panel is visible, switch its CSS class to `panel--mobile` (covers bottom ~75% as a sheet). Switch back on desktop.

### `dock.js`

A single `<div id="dock">` fixed at the bottom-right, holding a horizontal strip of `<button>` reopen icons. Each button is 32×32, semi-transparent (`background: rgba(0,0,0,0.5)`, border to match the panel it represents). Only buttons for *closed* panels are visible — clicking one shows its panel and hides the button.

The dock starts populated with all known panels' icons (since the default state is all-closed). As panels open, their icons disappear from the dock.

Initial icon set:
- 🔍 magnifier
- 🧠 network / x-ray
- 📊 chemosensory
- ⚡ emotion radar
- 📈 hover-PCA popup
- ℹ️ help / keyboard legend
- 🏷️ worm name + nav

Z-index ordering: dock at `z-index: 30` (above panels, below the magnifier and any tooltips).

### `state.js`

Persistence keyed per worm: `wormlet:focus:<wormName>:visible` → JSON object `{ [panelId]: boolean }`. First-time visitors get the all-closed default. On change, `state.save(id, bool)` updates one key and writes the whole object back. `state.load()` returns the object (or `{}` on first visit).

Magnifier position is also persisted: `wormlet:focus:<wormName>:magnifier-pos` → `{ x, y, diameter }`. Re-clamped on load if the viewport shrank since last visit.

### `magnifier.js`

DOM structure (created on first show):
```html
<div id="magnifier" class="magnifier">
  <canvas id="magnifier-canvas"></canvas>
  <button class="magnifier-close" aria-label="close magnifier">✕</button>
</div>
```

CSS:
```css
.magnifier {
  position: fixed;
  border: 2px solid rgba(100, 200, 255, 0.6);
  border-radius: 50%;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
  cursor: grab;
  z-index: 35;
}
.magnifier.dragging { cursor: grabbing; }
.magnifier canvas { display: block; width: 100%; height: 100%; border-radius: 50%; }
.magnifier-close {
  position: absolute; top: -8px; right: -8px;
  width: 24px; height: 24px; border-radius: 50%;
  background: #000; color: #fff; border: 1px solid rgba(100, 200, 255, 0.6);
  font-size: 14px; line-height: 22px; padding: 0; cursor: pointer;
  z-index: 36;
}
```

Sizing: `diameter = Math.round(Math.min(innerWidth, innerHeight) * 0.30)`. Clamped to `[120, 360]` px.

Drag implementation uses Pointer Events (handles mouse and touch uniformly):
- `pointerdown` on the lens body (not the close button) → start drag, call `setPointerCapture`, store offset.
- `pointermove` while captured → update `style.left/top`, clamp to viewport.
- `pointerup` / `pointercancel` → release capture, save position to state.

Rendering: every frame (driven by `index.js`'s main loop, not its own RAF), `magnifier.render()` calls `xray-render.drawXRay(magnifierCtx, lensScreenRect, worldToScreen)` where:
- `magnifierCtx` is the lens canvas's 2D context (DPR-scaled like `text-canvas.js`).
- `lensScreenRect` is `{ x, y, w, h }` in CSS pixels — the lens's position and size on the page.
- `worldToScreen` is the same function the main worm renderer uses, so neurons inside the lens line up with the worm body underneath.

The shared `xray-render.js` takes a screen rect and projects neuron body coords into it using the existing world-to-screen transform. When `xrayLabelsVisible` is true (the existing `l` key toggle), labels render inside the lens too.

Mobile: identical behavior, just smaller diameter and finger-drag via Pointer Events.

## Camera & viewport behavior

`three-scene.js` builds the orthographic camera with frustum extents derived from `responsive.cameraFrustumScale()`. On `viewport-changed`, it recomputes frustum and calls `camera.updateProjectionMatrix()`. The main canvas's `renderer.setSize(innerWidth, innerHeight)` already runs on resize and is preserved.

On mobile (`isMobile() === true`), the frustum is ~55% the desktop size, which means the worm and surrounding text region appear ~1.8× larger — enough that Hamlet text spans most of the viewport width.

## Testing

**Automated (must pass):**
- `python tests/test_determinism.py` — same-seed-same-trajectory invariant. Should be untouched since no sim code changes.
- `python tests/test_smoke_multi.py` — server boots, basic routes return 200.

**Manual checklist (in PR description):**
1. Boot server with the new files; visit `/focus/Alice` on desktop (Chrome + Safari).
2. Default state: only worm + drifting Hamlet text + dock visible.
3. Click each dock icon → corresponding panel opens at its expected position.
4. Click each panel's X → panel disappears, dock icon reappears.
5. Reload page → previously-open panels are still open, previously-closed are still closed.
6. Resize browser to <768px wide → panels switch to bottom-sheet layout; camera zooms in; dock stays bottom-right.
7. Drifting Hamlet text is visually sharp (no fuzziness) on both 1× and 2× DPR displays.
8. Open magnifier → drag it over worm → neurons inside lens align with worm body underneath; labels appear when `l` is pressed.
9. On a phone (or Chrome mobile emulator) at 390×844: clean default, all interactions work via touch, magnifier drags with finger.

**Optional:** one Playwright test for the open/close/persist round trip. Will add only if it doesn't introduce a CI dependency requirement.

## Risks & open questions

- **Risk:** `worldToScreen` is called from multiple places (smells rendering, PCA popup, word draw). All call sites need updating after the DPR fix. Mitigation: grep for every `textcanvas.width` / `textcanvas.height` reference and convert.
- **Risk:** Camera zoom on mobile may put text outside the visible area for words whose world coords land near frustum edges. Mitigation: verify visually during the manual checklist; if it's a problem, the camera scale can be tuned (this is the parameter — it's 0.55 in the spec but is a tweak knob).
- **Open question (deferred to implementation):** should the dock auto-hide after N seconds of no interaction on mobile to maximize worm visibility? Not in scope for v1; revisit if the dock feels visually heavy.

## Out of scope (future)

- A "tour" or first-visit overlay explaining the dock icons.
- Pinch-to-zoom the magnifier diameter on mobile.
- A "compare two worms" magnifier showing the same body region across two worms side by side.
- Equivalent close/reopen treatment on `/`, `/poems`, `/generations`, `/about`.
