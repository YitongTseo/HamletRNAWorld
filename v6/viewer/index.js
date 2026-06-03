// Grid of live worm thumbnails. Subscribes to /ws/overview (10 Hz), draws
// each worm's compact midline + a sparse smattering of nearby food words.
// Click a card to navigate to /focus/<flask>/<name>.

const WORLD_W = 1600;
const WORLD_H = 1000;

const grid = document.getElementById('grid');
const status = document.getElementById('status');

// Palette (set by palette.js on <html> before this loads). The worm reshades
// to the experiment's accent color; the stage stays dark in every theme so
// the light-on-dark worms remain legible (even under the white themes).
const _cssVar = (name, fb) => {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name);
  return (v && v.trim()) || fb;
};
const STAGE_BG = _cssVar('--stage', '#001');
const WORM_COLOR = _cssVar('--accent', '#6f9');

// "flask/name" -> { card, canvas, ctx, dpr }
const cards = new Map();
// flask_name -> { section, sectionGrid, headerGen }
const flaskSections = new Map();

// Inject a small bit of CSS for flask sections + denser card layout for 40+ worms.
(function injectFlaskStyle() {
  const s = document.createElement('style');
  s.textContent = `
    #grid { display: flex; flex-direction: column; gap: 22px; }
    .flask-section { padding: 0 18px; }
    .flask-section > h2 {
      margin: 12px 0 6px 0; font-size: 13px; font-weight: 600;
      color: var(--accent, #6f9); letter-spacing: 0.5px;
      display: flex; gap: 10px; align-items: baseline;
    }
    .flask-section > h2 .gen { color: var(--dim, #5a5); font-weight: 400; font-size: 11px; }
    .flask-section > .flask-grid {
      display: grid; gap: 10px;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    }
    .flask-section .card .name { padding: 4px 8px; font-size: 11px; }
    .flask-section .card canvas { height: 110px; }
    .flask-section .card .recent { padding: 4px 8px; font-size: 10px; min-height: 14px; }
  `;
  document.head.appendChild(s);
})();

function ensureFlaskSection(flaskName, display, generation) {
  let entry = flaskSections.get(flaskName);
  if (!entry) {
    const section = document.createElement('div');
    section.className = 'flask-section';
    section.innerHTML = `<h2><span class="who">${display}</span><span class="gen"></span></h2><div class="flask-grid"></div>`;
    grid.appendChild(section);
    entry = {
      section,
      sectionGrid: section.querySelector('.flask-grid'),
      headerGen: section.querySelector('.gen'),
    };
    flaskSections.set(flaskName, entry);
  }
  entry.headerGen.textContent = (generation !== undefined && generation > 0) ? `gen ${generation}` : '';
  return entry;
}

function ensureCard(flaskName, wormName) {
  const key = `${flaskName}/${wormName}`;
  if (cards.has(key)) return cards.get(key);
  const flaskEntry = flaskSections.get(flaskName);
  if (!flaskEntry) return null; // shouldn't happen — section ensured before card
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML =
    `<div class="name"><span class="who"></span><span class="count">0 words</span></div>` +
    `<canvas></canvas>` +
    `<div class="recent">…</div>`;
  card.querySelector('.who').textContent = wormName;
  card.addEventListener('click', () => {
    location.href = `/focus/${encodeURIComponent(flaskName)}/${encodeURIComponent(wormName)}`;
  });
  flaskEntry.sectionGrid.appendChild(card);
  const canvas = card.querySelector('canvas');
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const entry = { card, canvas, ctx, dpr };
  cards.set(key, entry);
  return entry;
}

function drawWorm(entry, worm) {
  const { canvas, ctx } = entry;
  const cw = canvas.width / entry.dpr;
  const ch = canvas.height / entry.dpr;
  ctx.clearRect(0, 0, cw, ch);
  // Background.
  ctx.fillStyle = STAGE_BG;
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
    ctx.strokeStyle = WORM_COLOR;
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
    // New protocol always sends a flasks[] array (single 'default' flask in
    // legacy single-group mode). Defensive fallback: if a server somehow
    // still sends the old `worms` shape, wrap it as one default flask.
    const flasks = msg.flasks || [{ name: 'default', display: 'Worms', generation: 0, worms: msg.worms || [] }];
    for (const flask of flasks) {
      ensureFlaskSection(flask.name, flask.display || flask.name, flask.generation || 0);
      for (const worm of flask.worms) {
        const entry = ensureCard(flask.name, worm.name);
        if (!entry) continue;
        drawWorm(entry, worm);
        updateCardChrome(entry, worm);
      }
    }
  };
}
connect();

// --- Generation rollover overlay ---
// Polls /api/generation_status every 500ms. Invisible during normal sim;
// shows a centered banner with progress + LLM scoring updates while the
// worms are frozen for end-of-generation processing.
(function setupGenOverlay() {
  const style = document.createElement('style');
  style.textContent = `
    #gen-overlay {
      position: fixed; inset: 0; display: none;
      background: rgba(0, 0, 0, 0.78); z-index: 9999;
      align-items: center; justify-content: center;
      font: 13px ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--accent, #6f9);
    }
    #gen-overlay.visible { display: flex; }
    #gen-overlay .panel {
      min-width: 360px; max-width: 560px;
      padding: 22px 28px;
      background: rgba(0, 20, 10, 0.92);
      border: 1px solid rgba(100, 255, 200, 0.4);
      border-radius: 6px;
      box-shadow: 0 8px 40px rgba(0, 0, 0, 0.6);
    }
    #gen-overlay h2 { margin: 0 0 14px 0; font-size: 15px; color: var(--accent, #6f9); }
    #gen-overlay .phase { margin-bottom: 12px; color: #cfd; font-size: 13px; }
    #gen-overlay .bar { height: 6px; background: rgba(100, 255, 200, 0.12); border-radius: 3px; overflow: hidden; margin-bottom: 8px; }
    #gen-overlay .bar > div { height: 100%; background: var(--accent, #6f9); transition: width 250ms ease; }
    #gen-overlay .meta { color: var(--dim, #5a5); font-size: 11px; margin-top: 8px; }
    #gen-overlay .err { color: #f88; margin-top: 10px; font-size: 12px; }
  `;
  document.head.appendChild(style);

  const overlay = document.createElement('div');
  overlay.id = 'gen-overlay';
  overlay.innerHTML =
    `<div class="panel">` +
    `<h2>generation rollover</h2>` +
    `<div class="phase">…</div>` +
    `<div class="bar"><div style="width:0%"></div></div>` +
    `<div class="meta"></div>` +
    `<div class="err"></div>` +
    `</div>`;
  document.body.appendChild(overlay);

  const $phase = overlay.querySelector('.phase');
  const $bar = overlay.querySelector('.bar > div');
  const $meta = overlay.querySelector('.meta');
  const $err = overlay.querySelector('.err');

  const PHASE_LABELS = {
    running: 'simulation running',
    corpus_draining: 'last words drifting off-screen…',
    judging: 'LLM is reading the poems',
    evolving: 'computing NES gradient',
    committing: 'committing generation to git',
    respawning: 'spawning next generation',
  };

  async function tick() {
    try {
      const res = await fetch('/api/generation_status');
      if (!res.ok) return;
      const s = await res.json();
      const active = s.enabled && s.phase && s.phase !== 'running';
      overlay.classList.toggle('visible', active);
      if (!active) return;
      $phase.textContent = PHASE_LABELS[s.phase] || s.phase;
      const pct = s.worms_total > 0 ? Math.round(100 * s.worms_done / s.worms_total) : 0;
      $bar.style.width = (s.phase === 'judging' ? pct : 100) + '%';
      $meta.textContent =
        `generation ${s.generation} · flask ${s.group || '—'} · ` +
        `${s.worms_done}/${s.worms_total} worms scored · ${s.elapsed_s}s elapsed`;
      $err.textContent = s.error || '';
    } catch (_e) { /* server briefly unavailable mid-rollover; ignore */ }
  }
  setInterval(tick, 500);
  tick();
})();
