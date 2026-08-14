// Observation-log overview: a lab tray of petri dishes, one per worm.
// Subscribes to /ws/overview (10 Hz). Each dish is a CAMERA-FOLLOW crop of
// that worm's private world, centred on its head, so the specimen renders
// large — the worm is the subject, the scrolling play is the medium.
// Click a dish to open /focus/<flask>/<name>.
//
// Rendering is dark-field microscopy: ivory translucent nematode (tapered
// both ends, faint gut line, dark pharynx tip), agar vignette, glass rim.
// Body opacity tracks satiety when the server sends it (lifelike mode);
// dead specimens draw grey and their label turns to a red DECEASED.

const WORLD_W = 1600;
const WORLD_H = 1000;
// How much of the world a dish shows, edge to edge. Smaller = bigger worm.
const DISH_SPAN = 520;

const grid = document.getElementById('grid');
const status = document.getElementById('status');

const cards = new Map();          // "flask/name" -> entry
const flaskSections = new Map();  // flask_name -> { section, sectionGrid, headerGen }

(function injectTrayStyle() {
  const s = document.createElement('style');
  s.textContent = `
    #grid { display: flex; flex-direction: column; gap: 26px; }
    .flask-section > h2 {
      margin: 18px 0 18px; font-size: 11px; font-weight: 400;
      letter-spacing: 0.3em; text-transform: uppercase; color: var(--dim);
      display: flex; gap: 14px; align-items: baseline;
    }
    .flask-section > h2 .gen { color: var(--cyan); font-size: 10.5px; letter-spacing: 0.2em; }
    .flask-section > .flask-grid {
      display: grid; gap: 26px 20px;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    }
    .cell { cursor: pointer; text-align: center; }
    .cell canvas { display: block; width: 100%; aspect-ratio: 1; }
    .cell:hover canvas { filter: brightness(1.18); }
    .cell .lbl { margin-top: 7px; font-size: 12px; letter-spacing: 0.06em; }
    .cell .lbl small { display: block; color: var(--dim); font-size: 10.5px;
                       margin-top: 3px; letter-spacing: 0.12em; }
    .cell .lbl small.warn { color: var(--warn); }
    .cell .recent { margin-top: 4px; color: var(--dim); font-size: 10px;
                    font-style: italic; min-height: 13px; }
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
  entry.headerGen.textContent = (generation !== undefined && generation > 0) ? `generation ${generation}` : '';
  return entry;
}

function ensureCard(flaskName, wormName) {
  const key = `${flaskName}/${wormName}`;
  if (cards.has(key)) return cards.get(key);
  const flaskEntry = flaskSections.get(flaskName);
  if (!flaskEntry) return null;
  const cell = document.createElement('div');
  cell.className = 'cell';
  cell.innerHTML =
    `<canvas></canvas>` +
    `<div class="lbl"><span class="who"></span><small></small></div>` +
    `<div class="recent">…</div>`;
  cell.querySelector('.who').textContent = wormName;
  cell.addEventListener('click', () => {
    location.href = `/focus/${encodeURIComponent(flaskName)}/${encodeURIComponent(wormName)}`;
  });
  flaskEntry.sectionGrid.appendChild(cell);
  const canvas = cell.querySelector('canvas');
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const px = Math.max(1, Math.round(rect.width * dpr));
  canvas.width = px; canvas.height = px;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const entry = { cell, canvas, ctx, dpr, size: rect.width };
  cards.set(key, entry);
  return entry;
}

// --- dish rendering ----------------------------------------------------------
function drawDish(entry, worm) {
  const { ctx } = entry;
  const S = entry.size;             // CSS px, square
  const c = S / 2, R = S / 2 - 4;   // dish radius
  ctx.clearRect(0, 0, S, S);

  // glass + agar
  ctx.fillStyle = '#0d1015';
  ctx.beginPath(); ctx.arc(c, c, R + 3, 0, 7); ctx.fill();
  ctx.fillStyle = '#10141a';
  ctx.beginPath(); ctx.arc(c, c, R, 0, 7); ctx.fill();
  const vg = ctx.createRadialGradient(c, c, R * 0.3, c, c, R);
  vg.addColorStop(0, 'rgba(120,140,160,0.05)');
  vg.addColorStop(1, 'rgba(0,0,0,0.38)');
  ctx.fillStyle = vg;
  ctx.beginPath(); ctx.arc(c, c, R, 0, 7); ctx.fill();
  // rim + glass highlight
  ctx.strokeStyle = 'rgba(200,220,235,0.14)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(c, c, R, 0, 7); ctx.stroke();
  ctx.strokeStyle = 'rgba(230,240,250,0.28)'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(c, c, R + 2, Math.PI * 1.12, Math.PI * 1.45); ctx.stroke();

  ctx.save();
  ctx.beginPath(); ctx.arc(c, c, R - 2, 0, 7); ctx.clip();

  // camera follows the head; clamp so the crop stays inside the world
  const scale = S / DISH_SPAN;
  const half = DISH_SPAN / 2;
  let cx = worm.head ? worm.head[0] : WORLD_W / 2;
  let cy = worm.head ? worm.head[1] : WORLD_H / 2;
  cx = Math.max(half, Math.min(WORLD_W - half, cx));
  cy = Math.max(half, Math.min(WORLD_H - half, cy));
  const X = (x) => c + (x - cx) * scale;
  const Y = (y) => c + (y - cy) * scale;

  const dead = worm.dead === true;
  const sat = (typeof worm.satiety === 'number') ? worm.satiety : 1;

  // words on the agar — quiet ivory, brightening slightly near the head
  if (worm.food && worm.food.length) {
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const f of worm.food) {
      if (!f.word) continue;
      const d = Math.hypot(f.x - cx, f.y - cy);
      if (d > half * 1.15) continue;
      const near = dead ? 0 : Math.max(0, 1 - d / half);
      ctx.font = `300 ${10 + 3 * near}px 'IBM Plex Mono', monospace`;
      ctx.fillStyle = `rgba(210,225,235,${0.28 + 0.5 * near})`;
      ctx.fillText(f.word, X(f.x), Y(f.y));
    }
  }

  // the nematode: layered ivory strokes, tapered both ends
  const M = worm.midline;
  if (M && M.length > 3) {
    const n = M.length;
    const body = dead ? 0.22 : 0.30 + 0.55 * sat;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    const taper = (u) => Math.sin(Math.PI * Math.min(1, u * 0.92 + 0.05)) ** 0.55;
    // translucent sheath
    for (let k = 0; k < n - 1; k++) {
      const w = taper(k / n);
      ctx.strokeStyle = `rgba(232,228,218,${0.10 * body + 0.10 * body * w})`;
      ctx.lineWidth = 34 * w * scale + 1;   // ~34 world-units of sheath
      ctx.beginPath();
      ctx.moveTo(X(M[k][0]), Y(M[k][1])); ctx.lineTo(X(M[k + 1][0]), Y(M[k + 1][1]));
      ctx.stroke();
    }
    // cuticle
    for (let k = 0; k < n - 1; k++) {
      const w = taper(k / n);
      ctx.strokeStyle = dead
        ? `rgba(150,150,148,${0.30 + 0.25 * w})`
        : `rgba(232,228,218,${0.45 * body + 0.55 * body * w})`;
      ctx.lineWidth = 20 * w * scale + 0.7; // ~20 world-units of cuticle
      ctx.beginPath();
      ctx.moveTo(X(M[k][0]), Y(M[k][1])); ctx.lineTo(X(M[k + 1][0]), Y(M[k + 1][1]));
      ctx.stroke();
    }
    // gut line
    ctx.strokeStyle = `rgba(90,88,80,${dead ? 0.2 : 0.45 * body + 0.15})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X(M[0][0]), Y(M[0][1]));
    for (let k = 1; k < n; k++) ctx.lineTo(X(M[k][0]), Y(M[k][1]));
    ctx.stroke();
    // pharynx at the head
    if (!dead && worm.head) {
      ctx.fillStyle = `rgba(120,116,105,${0.65 * body + 0.2})`;
      ctx.beginPath(); ctx.arc(X(worm.head[0]), Y(worm.head[1]), 2, 0, 7); ctx.fill();
    }
  }

  if (worm.paused) {
    ctx.fillStyle = 'rgba(224,80,63,0.9)';
    ctx.font = `500 10px 'IBM Plex Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.fillText('PAUSED', c, c - R + 18);
  }
  ctx.restore();
}

function updateCardChrome(entry, worm) {
  const small = entry.cell.querySelector('.lbl small');
  if (worm.dead === true) {
    small.textContent = 'deceased';
    small.className = 'warn';
  } else if (typeof worm.satiety === 'number') {
    const starving = worm.satiety < 0.3;
    small.textContent = starving
      ? `starving · ${worm.word_count} words`
      : `sat ${worm.satiety.toFixed(2)} · ${worm.word_count} words`;
    small.className = starving ? 'warn' : '';
  } else {
    small.textContent = `${worm.word_count} word${worm.word_count === 1 ? '' : 's'}`;
    small.className = '';
  }
  const rec = entry.cell.querySelector('.recent');
  rec.textContent = (worm.recent_words && worm.recent_words.length)
    ? worm.recent_words.join(' · ')
    : '…';
}

let ws = null;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/overview`);
  ws.onopen = () => { status.textContent = 'LIVE'; };
  ws.onclose = () => { status.textContent = 'DISCONNECTED · RETRYING…'; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (_e) { return; }
    if (msg.type !== 'overview') return;
    const flasks = msg.flasks || [{ name: 'default', display: 'Worms', generation: 0, worms: msg.worms || [] }];
    for (const flask of flasks) {
      ensureFlaskSection(flask.name, flask.display || flask.name, flask.generation || 0);
      for (const worm of flask.worms) {
        const entry = ensureCard(flask.name, worm.name);
        if (!entry) continue;
        drawDish(entry, worm);
        updateCardChrome(entry, worm);
      }
    }
  };
}
connect();

// --- Generation rollover overlay --------------------------------------------
// Polls /api/generation_status every 500ms; restyled to the observation-log
// language but functionally identical to the old overlay.
(function setupGenOverlay() {
  const style = document.createElement('style');
  style.textContent = `
    #gen-overlay {
      position: fixed; inset: 0; display: none;
      background: rgba(4, 5, 7, 0.82); z-index: 9999;
      align-items: center; justify-content: center;
      font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 13px;
      color: var(--ivory);
    }
    #gen-overlay.visible { display: flex; }
    #gen-overlay .panel {
      min-width: 380px; max-width: 560px; padding: 24px 30px;
      background: #0d1015; border: 1px solid #232a33;
    }
    #gen-overlay h2 { margin: 0 0 14px; font-size: 11px; font-weight: 400;
      letter-spacing: 0.3em; text-transform: uppercase; color: var(--cyan); }
    #gen-overlay .phase { margin-bottom: 12px; font-size: 13px; }
    #gen-overlay .bar { height: 3px; background: #1a1f26; overflow: hidden; margin-bottom: 8px; }
    #gen-overlay .bar > div { height: 100%; background: var(--cyan); transition: width 250ms ease; }
    #gen-overlay .meta { color: var(--dim); font-size: 11px; margin-top: 8px; }
    #gen-overlay .err { color: var(--warn); margin-top: 10px; font-size: 12px; }
  `;
  document.head.appendChild(style);

  const overlay = document.createElement('div');
  overlay.id = 'gen-overlay';
  overlay.innerHTML =
    `<div class="panel">` +
    `<h2>brood turnover in progress</h2>` +
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
    corpus_draining: 'last words drifting off the agar…',
    judging: 'the judge is reading the poems',
    evolving: 'computing NES gradient',
    committing: 'archiving the generation',
    respawning: 'plating the next brood',
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
        `${s.worms_done}/${s.worms_total} specimens judged · ${s.elapsed_s}s elapsed`;
      $err.textContent = s.error || '';
    } catch (_e) { /* server briefly unavailable mid-rollover; ignore */ }
  }
  setInterval(tick, 500);
  tick();
})();
