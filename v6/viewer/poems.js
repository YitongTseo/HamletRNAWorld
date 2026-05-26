// Live poems: one column per (flask, worm) — so 36 columns when 6 flasks ×
// 6 worms. Bootstrap from /api/poems (nested by flask), then stream
// updates via /ws/poems. Each `eaten` event now carries flask + worm; we
// key columns by the composite "flask/worm" so different flasks' Alices
// stay separate.

const columnsEl = document.getElementById('columns');
const statusEl = document.getElementById('status');

// "flask/worm" -> { col, body, count }. Stable insertion order keeps the
// grid sorted by flask, then by worm within each flask.
const columns = new Map();
// flask_name -> { sectionWrap, sectionGrid } so flasks render as labelled
// bands. Empty wrap until the first column for that flask is created.
const flaskSections = new Map();

function ensureFlaskSection(flask) {
  let entry = flaskSections.get(flask);
  if (entry) return entry;
  const wrap = document.createElement('div');
  wrap.className = 'flask-wrap';
  wrap.innerHTML = `<h3 class="flask-head">${flask}</h3><div class="flask-grid"></div>`;
  columnsEl.appendChild(wrap);
  entry = { wrap, grid: wrap.querySelector('.flask-grid') };
  flaskSections.set(flask, entry);
  return entry;
}

function makeColumn(flask, worm) {
  const section = ensureFlaskSection(flask);
  const col = document.createElement('div');
  col.className = 'col';
  col.innerHTML = `
    <div class="col-head"><span class="who"></span><span class="count">0</span></div>
    <div class="col-body"></div>
  `;
  col.querySelector('.who').textContent = worm;
  section.grid.appendChild(col);
  return {
    col,
    body: col.querySelector('.col-body'),
    countEl: col.querySelector('.count'),
    count: 0,
  };
}

function appendWord(flask, worm, word) {
  const key = `${flask}/${worm}`;
  let c = columns.get(key);
  if (!c) {
    c = makeColumn(flask, worm);
    columns.set(key, c);
  }
  for (const el of c.body.querySelectorAll('.word.newest')) el.classList.remove('newest');
  const div = document.createElement('div');
  div.className = 'word newest';
  div.textContent = word;
  c.body.appendChild(div);
  c.count += 1;
  c.countEl.textContent = c.count;
  c.body.scrollTop = c.body.scrollHeight;
}

async function bootstrap() {
  const resp = await fetch('/api/poems');
  const data = await resp.json();
  // /api/poems returns {flask_name: {worm_name: [words...]}} in multi-flask
  // mode (legacy shape was {worm_name: [...]} — we detect and handle both).
  const flaskNames = Object.keys(data).sort();
  for (const flask of flaskNames) {
    const inner = data[flask];
    if (!inner || typeof inner !== 'object') continue;
    // Legacy fallback: inner might be a flat array (single-group mode).
    if (Array.isArray(inner)) {
      const c = makeColumn('default', flask);  // here 'flask' is actually a worm name
      columns.set(`default/${flask}`, c);
      const frag = document.createDocumentFragment();
      for (const w of inner) {
        const div = document.createElement('div');
        div.className = 'word';
        div.textContent = w;
        frag.appendChild(div);
      }
      c.body.appendChild(frag);
      c.count = inner.length;
      c.countEl.textContent = inner.length;
      c.body.scrollTop = c.body.scrollHeight;
      continue;
    }
    const wormNames = Object.keys(inner).sort();
    for (const worm of wormNames) {
      const words = inner[worm];
      const c = makeColumn(flask, worm);
      columns.set(`${flask}/${worm}`, c);
      const frag = document.createDocumentFragment();
      for (const w of words) {
        const div = document.createElement('div');
        div.className = 'word';
        div.textContent = w;
        frag.appendChild(div);
      }
      c.body.appendChild(frag);
      c.count = words.length;
      c.countEl.textContent = words.length;
      c.body.scrollTop = c.body.scrollHeight;
    }
  }
}

let ws = null;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/poems`);
  ws.onopen = () => { statusEl.textContent = 'live'; };
  ws.onclose = () => { statusEl.textContent = 'disconnected · retrying…'; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (_e) { return; }
    if (msg.type === 'eaten') {
      // Defensive: legacy `eaten` events may lack flask; default to 'default'.
      const flask = msg.flask || 'default';
      appendWord(flask, msg.worm, msg.word);
    }
  };
}

(async () => {
  await bootstrap();
  connect();
})();
