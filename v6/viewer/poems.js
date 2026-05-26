// Live poems: one column per worm, oldest at top, newest at bottom.
// Bootstrap from /api/poems, then stream updates via /ws/poems.

const columnsEl = document.getElementById('columns');
const statusEl = document.getElementById('status');

const columns = new Map(); // name -> { col, body, count }

function makeColumn(name) {
  const col = document.createElement('div');
  col.className = 'col';
  col.innerHTML = `
    <div class="col-head"><span class="who"></span><span class="count">0</span></div>
    <div class="col-body"></div>
  `;
  col.querySelector('.who').textContent = name;
  columnsEl.appendChild(col);
  return {
    col,
    body: col.querySelector('.col-body'),
    countEl: col.querySelector('.count'),
    count: 0,
  };
}

function appendWord(name, word) {
  const c = columns.get(name);
  if (!c) return;
  // Demote any previously-newest entries.
  for (const el of c.body.querySelectorAll('.word.newest')) el.classList.remove('newest');
  const div = document.createElement('div');
  div.className = 'word newest';
  div.textContent = word;
  c.body.appendChild(div);
  c.count += 1;
  c.countEl.textContent = c.count;
  // Auto-scroll to the bottom.
  c.body.scrollTop = c.body.scrollHeight;
}

async function bootstrap() {
  const resp = await fetch('/api/poems');
  const data = await resp.json();
  for (const [name, words] of Object.entries(data)) {
    if (!columns.has(name)) columns.set(name, makeColumn(name));
    const c = columns.get(name);
    // Render in bulk (avoid one append per word).
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
      appendWord(msg.worm, msg.word);
    }
  };
}

(async () => {
  await bootstrap();
  connect();
})();
