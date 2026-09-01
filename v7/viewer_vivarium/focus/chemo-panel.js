// ---------------------------------------------------------------------------
// Chemosensory panel — top-left HTML overlay (#chemosensoryPanel).
// ---------------------------------------------------------------------------
// Renders one row per PC / chemosensory-neuron pair with stacked horizontal
// bars showing the per-word contributions on the L and R neurons of the pair.
// Bars are HTML, not canvas — innerHTML is rewritten each WebSocket frame.
//
// Toggle:
//   'c' — show/hide the panel
//
// Live app state (corpusPca + computed contributions) is owned by index.js
// and passed in each call via the `state` parameter; the panel only owns the
// DOM ref and the visibility flag.

import * as chrome from './panel-chrome.js';

const chemosensoryPanel = document.getElementById('chemosensoryPanel');

// ---------------------------------------------------------------------------
// View-mode flag (module-local)
// ---------------------------------------------------------------------------
// chemosensoryVisible is now synced via panel-chrome's onShow/onHide
// callbacks below. Default = hidden until the user opens it via dock or 'c'.
let chemosensoryVisible = false;

export function isChemoVisible() { return chemosensoryVisible; }
// toggleChemo delegates to panel-chrome so the dock button + saved state
// stay in sync with the keyboard shortcut.
export function toggleChemo() {
  if (chemosensoryVisible) chrome.hidePanel('chemo');
  else chrome.showPanel('chemo');
}

// ---------------------------------------------------------------------------
// Local helpers (only used by the chemo panel's stacked bars).
// ---------------------------------------------------------------------------

// Each PC gets a stable color hue spaced around the wheel so the eye can
// learn "blue = PC3" type associations over time.
function pcHue(i) { return Math.round((i * 360) / 12); }

// Hash-based color per word so the same word draws the same color in every
// bar segment it appears in, every frame. Stable across reloads.
function wordHue(word) {
  let h = 0;
  const s = word.toLowerCase();
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return ((h % 360) + 360) % 360;
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// Render a single horizontal bar composed of stacked colored segments,
// one per contributing word. Segments are sorted ALPHABETICALLY (stable
// positions across frames), colored by per-word hash hue, and labeled
// inline if the segment is wide enough. Eaten words get a dashed border
// so you can see at a glance which contributions are residual.
//
// `entries` is a list of {word, value, eaten}. `barMax` is the bar's
// pixel width. `totalScale` is the value that should fill the full bar
// (we use 1.0 as "saturated", same as the sim's per-neuron cap).
function stackedBar(entries, barMax, totalScale = 1.0) {
  const sorted = entries.slice().sort((a, b) =>
    a.word.toLowerCase().localeCompare(b.word.toLowerCase())
  );
  const total = sorted.reduce((s, e) => s + e.value, 0);
  const fillFrac = Math.max(0, Math.min(1, total / totalScale));
  const fillPx = fillFrac * barMax;

  let html = `<span style="display:inline-block; position:relative; width:${barMax}px; height:14px; background:rgba(255,255,255,0.07); border-radius:2px; vertical-align:middle; overflow:hidden;">`;

  if (total > 0) {
    let x = 0;
    for (const e of sorted) {
      const segPx = (e.value / total) * fillPx;
      const hue = wordHue(e.word);
      const bg = `hsl(${hue},75%,55%)`;
      const borderStyle = e.eaten ? `outline:1px dashed hsl(${hue},85%,80%); outline-offset:-1px;` : '';
      const labelFits = segPx > 28;  // ~3 chars of monospace at 9px
      const label = labelFits
        ? `<span style="position:absolute; left:3px; top:1px; color:#000a; font-size:9px; mix-blend-mode:luminosity; pointer-events:none; max-width:${segPx - 4}px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${escapeHTML(e.word)}</span>`
        : '';
      html += `<span title="${escapeHTML(e.word)}${e.eaten ? ' · eaten' : ''}: ${(e.value*100|0)}%" style="position:absolute; left:${x}px; top:0; width:${segPx}px; height:100%; background:${bg}; box-shadow:0 0 4px hsl(${hue},90%,60%); ${borderStyle}">${label}</span>`;
      x += segPx;
    }
  }
  html += `</span>`;
  return html;
}

// ---------------------------------------------------------------------------
// Renderer — called from the main render loop / WebSocket frame handler.
// `state` carries the live data the panel doesn't own:
//   { corpusPca, contributions }   (contributions = computeWordImpact().contributions)
// ---------------------------------------------------------------------------
export function drawChemoPanel(state) {
  if (!chemosensoryVisible) return;
  const { corpusPca, contributions } = state;
  if (!corpusPca) {
    chemosensoryPanel.innerHTML = `<div style="opacity:0.4; padding:6px 0; font-size:10px;">loading corpus PCA…</div>`;
    return;
  }

  const pairs = corpusPca.pc_neuron_pairs;

  // Header
  const anyFiring = pairs.some(([L, R]) =>
    (contributions[L] && contributions[L].length) || (contributions[R] && contributions[R].length)
  );
  let html = `<div style="font-weight:bold; margin-bottom:8px; color:${anyFiring ? '#e0c48f' : '#b3a789'}; opacity:${anyFiring ? 1 : 0.35};">● CHEMOSENSORY STATE (${corpusPca.embeddingName || 'PCA'})</div>`;

  // Column headers
  const labelW = 60, barW = 110;
  html += `<div style="display:grid; grid-template-columns:${labelW}px ${barW}px ${barW}px; gap:6px; align-items:center; font-size:9px; opacity:0.45; margin-bottom:3px;">
    <span>pair</span><span style="text-align:center;">L</span><span style="text-align:center;">R</span>
  </div>`;

  // 12 rows — one per neuron pair.
  for (let i = 0; i < pairs.length; i++) {
    const [L, R] = pairs[i];
    const Lcontribs = contributions[L] || [];
    const Rcontribs = contributions[R] || [];
    const pairBase = L.slice(0, -1);
    const pcHueValue = pcHue(i);
    html += `<div style="display:grid; grid-template-columns:${labelW}px ${barW}px ${barW}px; gap:6px; align-items:center; margin:2px 0;">
      <span style="font-size:9px; color:hsl(${pcHueValue},55%,72%); opacity:0.85;">PC${i} · ${pairBase}</span>
      ${stackedBar(Lcontribs, barW)}
      ${stackedBar(Rcontribs, barW)}
    </div>`;
  }

  // Footer legend
  html += `<div style="margin-top:10px; font-size:9px; opacity:0.5; line-height:1.5;">
    bars sum per-word contributions · colors are stable per word ·
    <span style="display:inline-block; width:8px; height:8px; background:#cfa348; vertical-align:middle;"></span> sensed ·
    <span style="display:inline-block; width:8px; height:8px; outline:1px dashed #ccc; outline-offset:-1px; vertical-align:middle;"></span> eaten (residual)
  </div>`;

  chemosensoryPanel.innerHTML = html;
}

// Register with panel-chrome AFTER the toggle/flag declarations so the
// onShow/onHide callbacks can flip chemosensoryVisible. chrome.register
// applies saved visibility immediately (default = hidden on fresh visit).
chrome.register({
  id: 'chemo',
  label: 'chemosensory',
  panelEl: chemosensoryPanel,
  onShow: () => { chemosensoryVisible = true; },
  onHide: () => { chemosensoryVisible = false; },
});

export { chemosensoryPanel };
