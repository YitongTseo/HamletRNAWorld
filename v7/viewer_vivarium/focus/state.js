// viewer/focus/state.js
// Mirror index.js's URL parsing so the two-segment route works:
//   /focus/<flask>/<worm>   ← multi-flask mode
//   /focus/<worm>           ← legacy single-group mode, flask='default'
// Without this, /focus/flaskA/Alice would key on "flaskA/Alice" instead of
// "Alice", losing the per-worm UI state on every navigation.
const _pathParts = location.pathname.replace(/^\/focus\//, '').replace(/\/$/, '').split('/');
const flaskName = _pathParts.length >= 2 ? decodeURIComponent(_pathParts[0]) : 'default';
const wormName = decodeURIComponent(_pathParts[_pathParts.length - 1]);
// Scope keys by (flask, worm) so the same worm name in different flasks keeps
// independent visibility/magnifier state — same distinction index.js makes for
// its WebSocket route.
const visKey = `wormlet:focus:${flaskName}:${wormName}:visible`;
const magKey = `wormlet:focus:${flaskName}:${wormName}:magnifier-pos`;

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

// Per-panel drag/resize geometry ({left, top, width, height} as CSS strings),
// keyed by panel id, scoped per (flask, worm) like visibility above.
const geomKey = `wormlet:focus:${flaskName}:${wormName}:geom`;
export function loadPanelGeom(id) {
  try { return (JSON.parse(localStorage.getItem(geomKey)) || {})[id] || null; }
  catch { return null; }
}
export function savePanelGeom(id, geom) {
  let cur = {};
  try { cur = JSON.parse(localStorage.getItem(geomKey)) || {}; } catch {}
  cur[id] = geom;
  try { localStorage.setItem(geomKey, JSON.stringify(cur)); } catch {}
}
