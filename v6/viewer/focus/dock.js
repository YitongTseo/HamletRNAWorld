// viewer/focus/dock.js
// Persistent labelled toggle bar (bottom-right). Replaces the old ASCII
// reopen-strip: every registered panel gets one word-labelled button that
// toggles its panel and highlights ('active') while that panel is open.
const barEl = document.createElement('div');
barEl.id = 'toggle-bar';
document.body.appendChild(barEl);

const buttons = new Map(); // id -> button element

// Left-to-right order of the toggle buttons. ids not listed sort to the end.
const ORDER = ['magnifier', 'network', 'chemo', 'radar'];
function orderIdx(id) {
  const i = ORDER.indexOf(id);
  return i < 0 ? ORDER.length : i;
}
function reorder() {
  const sorted = [...buttons.entries()].sort((a, b) => orderIdx(a[0]) - orderIdx(b[0]));
  for (const [, btn] of sorted) barEl.appendChild(btn); // appendChild re-orders in DOM
}

export function register(id, label, onToggle) {
  const btn = document.createElement('button');
  btn.className = 'toggle-btn';
  btn.textContent = label;
  btn.dataset.toggleId = id;
  btn.addEventListener('click', onToggle);
  buttons.set(id, btn);
  barEl.appendChild(btn);
  reorder();
}

export function setActive(id, active) {
  const btn = buttons.get(id);
  if (btn) btn.classList.toggle('active', !!active);
}
