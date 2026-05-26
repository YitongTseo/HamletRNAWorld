// Grid of live worm thumbnails. Subscribes to /ws/overview (10 Hz), draws
// each worm's compact midline + a sparse smattering of nearby food words.
// Click a card to navigate to /focus/<name>.

const WORLD_W = 1600;
const WORLD_H = 1000;

const grid = document.getElementById('grid');
const status = document.getElementById('status');

// name -> { card, canvas, ctx, name, count, recent }
const cards = new Map();

function ensureCard(name) {
  if (cards.has(name)) return cards.get(name);
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML =
    `<div class="name"><span class="who"></span><span class="count">0 words</span></div>` +
    `<canvas></canvas>` +
    `<div class="recent">…</div>`;
  card.querySelector('.who').textContent = name;
  card.addEventListener('click', () => {
    location.href = `/focus/${encodeURIComponent(name)}`;
  });
  grid.appendChild(card);
  const canvas = card.querySelector('canvas');
  // Use the card's measured width but cap the DPR cost on retina.
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const entry = { card, canvas, ctx, dpr };
  cards.set(name, entry);
  return entry;
}

function drawWorm(entry, worm) {
  const { canvas, ctx } = entry;
  const cw = canvas.width / entry.dpr;
  const ch = canvas.height / entry.dpr;
  ctx.clearRect(0, 0, cw, ch);
  // Background.
  ctx.fillStyle = '#001';
  ctx.fillRect(0, 0, cw, ch);

  // Map world coords to canvas (letterbox to keep aspect).
  const sx = cw / WORLD_W;
  const sy = ch / WORLD_H;
  const s = Math.min(sx, sy);
  const ox = (cw - WORLD_W * s) / 2;
  const oy = (ch - WORLD_H * s) / 2;
  const X = (x) => ox + x * s;
  const Y = (y) => oy + y * s;

  // World rect (subtle border).
  ctx.strokeStyle = 'rgba(100,200,255,0.15)';
  ctx.lineWidth = 1;
  ctx.strokeRect(ox + 0.5, oy + 0.5, WORLD_W * s - 1, WORLD_H * s - 1);

  // Scrolling words (food).
  if (worm.food && worm.food.length) {
    ctx.fillStyle = 'rgba(180, 200, 255, 0.45)';
    ctx.font = `${Math.max(8, Math.round(10))}px ui-monospace, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const f of worm.food) {
      if (!f.word) continue;
      ctx.fillText(f.word, X(f.x), Y(f.y));
    }
  }

  // Worm midline.
  if (worm.midline && worm.midline.length > 1) {
    ctx.strokeStyle = '#6f9';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < worm.midline.length; i++) {
      const [x, y] = worm.midline[i];
      if (i === 0) ctx.moveTo(X(x), Y(y));
      else ctx.lineTo(X(x), Y(y));
    }
    ctx.stroke();
    // Head dot.
    const [hx, hy] = worm.head;
    ctx.fillStyle = '#cfc';
    ctx.beginPath();
    ctx.arc(X(hx), Y(hy), 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // Paused indicator.
  if (worm.paused) {
    ctx.fillStyle = '#fc6';
    ctx.font = '11px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('PAUSED', 6, 4);
  }
}

function updateCardChrome(entry, worm) {
  entry.card.querySelector('.count').textContent = `${worm.word_count} word${worm.word_count === 1 ? '' : 's'}`;
  const rec = entry.card.querySelector('.recent');
  rec.textContent = (worm.recent_words && worm.recent_words.length)
    ? worm.recent_words.join(' · ')
    : '…';
}

let ws = null;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/overview`);
  ws.onopen = () => { status.textContent = 'live'; };
  ws.onclose = () => { status.textContent = 'disconnected · retrying…'; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (_e) { return; }
    if (msg.type !== 'overview') return;
    for (const worm of msg.worms) {
      const entry = ensureCard(worm.name);
      drawWorm(entry, worm);
      updateCardChrome(entry, worm);
    }
  };
}
connect();
