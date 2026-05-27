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
