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
