// Generations viewer.
// Bootstrap: GET /api/generations to populate the flask dropdown,
// then for the selected flask:
//   GET /api/generations/<flask>       — per-gen summaries (newest first)
//   GET /api/generations/<flask>/<gen> — full detail when a row is clicked
//   GET /api/generations/<flask>/weights/trajectory — NN-evolution heatmap
//   GET /api/generations/meta/index    — meta-gardener log per epoch
//
// Charts: Chart.js (loaded from CDN). One chart each for best, average,
// and sigma per generation. Heatmap is hand-drawn to a canvas because
// Chart.js doesn't ship one.

const statusEl = document.getElementById('status');
const flaskSelect = document.getElementById('flask-select');
const flaskMeta = document.getElementById('flask-meta');
const genListEl = document.getElementById('gen-list');
const detailCard = document.getElementById('detail-card');
const detailTitle = document.getElementById('detail-title');
const detailLog = document.getElementById('detail-log');
const detailPoem = document.getElementById('detail-poem');
const detailWindows = document.getElementById('detail-windows');
const detailHighlights = document.getElementById('detail-highlights');
const heatmapEl = document.getElementById('heatmap');
const heatStatsEl = document.getElementById('heat-stats');
const metaLogListEl = document.getElementById('meta-log-list');

const charts = { best: null, avg: null, sigma: null, lineage: null, pos: null };
let currentFlask = null;
let currentGenerations = []; // newest-first
let currentExperiment = null; // set by /api/experiment via experiment_dropdown.js

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

function setStatus(text, isError) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? '#cd5d4a' : '';
}

function chartOpts(title) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: title, color: '#cfa348', font: { size: 11, weight: 'normal' } },
    },
    scales: {
      x: { ticks: { color: '#8f8266', font: { size: 10 } }, grid: { color: 'rgba(205,127,93,0.06)' } },
      y: { ticks: { color: '#8f8266', font: { size: 10 } }, grid: { color: 'rgba(205,127,93,0.06)' } },
    },
    elements: { point: { radius: 0, hoverRadius: 4 } },
  };
}

function lineDataset(label, color, data) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color,
    borderWidth: 1.5,
    tension: 0.15,
    pointRadius: 0,
  };
}

// Mirror the bounds defined in server/evolution.py so the σ chart can show
// the band σ is allowed to bounce in. If the user retunes those constants,
// update them here too — there's no /api/config to fetch from yet.
const SIGMA_MIN = 0.05;
const SIGMA_INIT = 0.5;
const SIGMA_MAX = 3.0;

function constDataset(label, color, value, n, dash) {
  return {
    label,
    data: Array(n).fill(value),
    borderColor: color,
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderDash: dash || [4, 4],
    tension: 0,
    pointRadius: 0,
  };
}

function rollingStd(arr, win) {
  // Sample std over a trailing window. Output aligns with input length;
  // the leading (win-1) entries are nulls so Chart.js draws a gap.
  const out = new Array(arr.length).fill(null);
  for (let i = win - 1; i < arr.length; i++) {
    const slice = arr.slice(i - win + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const v = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length;
    out[i] = Math.sqrt(v);
  }
  return out;
}

function chartOptsWithLegend(title) {
  const opts = chartOpts(title);
  opts.plugins.legend = {
    display: true,
    position: 'top',
    align: 'end',
    labels: { color: '#b3a789', font: { size: 9 }, boxWidth: 8, boxHeight: 2 },
  };
  return opts;
}

function renderCharts(gens) {
  // gens is newest-first; reverse for chronological plotting.
  const chrono = [...gens].reverse();
  const labels = chrono.map(g => g.generation);
  const best = chrono.map(g => g.best_score);
  const avg = chrono.map(g => {
    const fs = (g.worms || []).map(w => w.fitness);
    return fs.length ? fs.reduce((a, b) => a + b, 0) / fs.length : 0;
  });
  const sigmas = chrono.map(g => g.sigma_used);
  const sigmaStd10 = rollingStd(sigmas, Math.min(10, sigmas.length));

  for (const k of ['best', 'avg', 'sigma']) {
    if (charts[k]) { charts[k].destroy(); charts[k] = null; }
  }
  charts.best = new Chart(document.getElementById('chart-best'), {
    type: 'line',
    data: { labels, datasets: [lineDataset('best fitness', '#cfa348', best)] },
    options: chartOpts('best fitness per generation'),
  });
  charts.avg = new Chart(document.getElementById('chart-avg'), {
    type: 'line',
    data: { labels, datasets: [lineDataset('avg fitness', '#cfa348', avg)] },
    options: chartOpts('average fitness per generation'),
  });
  // σ chart shows the actual σ used each gen, the rolling 10-gen sample
  // std (a direct "how jumpy is it" signal — flat means stable, spiky
  // means runaway adaptation), and dashed reference lines at the bounds
  // declared in evolution.py so the eye can immediately spot when σ is
  // pinned at the cap (= NES is yelling 'we're stuck, mutate harder').
  charts.sigma = new Chart(document.getElementById('chart-sigma'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        lineDataset('σ used', '#cd5d4a', sigmas),
        lineDataset('σ stability (rolling std, 10 gens)', '#cd7f5d', sigmaStd10),
        constDataset(`max (${SIGMA_MAX})`, 'rgba(255,102,102,0.45)', SIGMA_MAX, sigmas.length, [2, 4]),
        constDataset(`init (${SIGMA_INIT})`, 'rgba(207,163,72,0.45)', SIGMA_INIT, sigmas.length, [2, 4]),
        constDataset(`min (${SIGMA_MIN})`, 'rgba(150,150,150,0.4)', SIGMA_MIN, sigmas.length, [2, 4]),
      ],
    },
    options: chartOptsWithLegend('σ (learning rate) — is it steady?'),
  });
  // New experiment-mode-aware charts (no-op when the flask has no per-worm
  // / pos-totals data — i.e. prod-format generations).
  try { renderLineageChart(gens); } catch (e) { console.warn('lineage chart failed:', e); }
  try { renderPOSChart(gens); } catch (e) { console.warn('POS chart failed:', e); }
}

function renderGenList(gens) {
  genListEl.innerHTML = '';
  const header = document.createElement('div');
  header.className = 'gen-row';
  header.style.cssText = 'background:rgba(205,127,93,0.07); font-weight:600; cursor:default;';
  header.innerHTML = `
    <span class="num">gen</span>
    <span class="score">best</span>
    <span class="sigma">σ used</span>
    <span class="winner">winner · rank order</span>
    <span class="nscored">n scored</span>
  `;
  genListEl.appendChild(header);

  for (const g of gens) {
    const row = document.createElement('div');
    row.className = 'gen-row';
    row.dataset.gen = g.generation;
    const winner = (g.ranks && g.ranks[0]) || (g.worms[0] && g.worms[0].name) || '—';
    const nscored = (g.worms || []).reduce((a, w) => a + (w.windows_scored || 0), 0);
    const rest = (g.ranks || []).slice(1, 6).join(' · ');
    row.innerHTML = `
      <span class="num">${g.generation}</span>
      <span class="score">${g.best_score.toFixed(3)}</span>
      <span class="sigma">${g.sigma_used.toFixed(2)}</span>
      <span class="winner"><strong>${winner}</strong>${rest ? ' › ' + rest : ''}</span>
      <span class="nscored">${nscored}</span>
    `;
    row.addEventListener('click', () => selectGeneration(g.generation, row));
    genListEl.appendChild(row);
  }
}

async function selectGeneration(genNum, rowEl) {
  document.querySelectorAll('.gen-row.selected').forEach(r => r.classList.remove('selected'));
  if (rowEl) rowEl.classList.add('selected');
  detailCard.style.display = 'block';
  detailTitle.textContent = `generation ${genNum} detail`;
  detailLog.innerHTML = '<div class="subtle">loading…</div>';
  detailPoem.textContent = '';
  detailWindows.innerHTML = '';
  detailHighlights.innerHTML = '';
  let data;
  try {
    data = await fetchJSON(`/api/generations/${currentFlask}/${genNum}`);
  } catch (e) {
    detailLog.innerHTML = `<div class="subtle" style="color:#cd5d4a;">error: ${e.message}</div>`;
    return;
  }

  // Log
  if (data.log) {
    detailLog.innerHTML = `<div class="log-box">${escapeHtml(data.log)}</div>`;
  } else if (data.log_skipped) {
    detailLog.innerHTML = '<div class="log-box skipped">the gardener rested this generation.</div>';
  } else {
    detailLog.innerHTML = '<div class="subtle">(no per-flask log; meta-gardener writes once per epoch — see bottom of page)</div>';
  }

  // Top worm's poem
  const winner = (data.summary && data.summary.ranks && data.summary.ranks[0]) || null;
  const winnerData = winner && data.worms.find(w => w.name === winner);
  if (winnerData && winnerData.poem_clean) {
    detailPoem.innerHTML =
      `<div class="meta">${winner} — winner</div>` + escapeHtml(winnerData.poem_clean);
  } else if (winner) {
    detailPoem.innerHTML = `<div class="meta">${winner} — winner</div><div class="subtle">poem text not preserved on disk (purged after commit; recover from git history if needed).</div>`;
  } else {
    detailPoem.textContent = '(no winner recorded)';
  }

  // Top windows across all worms
  const allWindows = [];
  for (const w of data.worms) {
    for (const sw of (w.scored_windows || [])) {
      allWindows.push({ worm: w.name, ...sw });
    }
  }
  allWindows.sort((a, b) => (b.quality || 0) - (a.quality || 0));

  // Standout windows on each axis. "artsy" = emotional, "comprehensible" =
  // coherence (the two axes the judge grades). Combined = emotional+coherence.
  if (allWindows.length) {
    const bestBy = (key) => allWindows.reduce((a, b) => ((b[key] || 0) > (a[key] || 0) ? b : a));
    detailHighlights.innerHTML = [
      hlCard('best combined', 'var(--accent)', bestBy('quality')),
      hlCard('most artsy', 'var(--warm)', bestBy('emotional')),
      hlCard('most comprehensible', '#cd7f5d', bestBy('coherence')),
    ].join('');
  } else {
    detailHighlights.innerHTML = '<div class="subtle">no scored windows for this generation yet.</div>';
  }

  const top = allWindows.slice(0, 12);
  detailWindows.innerHTML = top.length
    ? top.map(w => `
        <div class="window">
          <span class="scores">E=${w.emotional} C=${w.coherence}</span>
          <span style="color:#b3a789;">${w.worm}</span>
          “${escapeHtml((w.tokens || []).join(' '))}”
        </div>`).join('')
    : '<div class="subtle">no scored windows for this generation.</div>';
}

function hlCard(label, color, w) {
  return `
    <div class="highlight" style="border-color:${color};">
      <div class="hl-label" style="color:${color};">${label}</div>
      <div class="hl-scores">
        <span style="color:var(--warm);">artsy ${w.emotional}</span> ·
        <span style="color:#cd7f5d;">comprehensible ${w.coherence}</span>
        <span class="hl-worm">· ${escapeHtml(w.worm)}</span>
      </div>
      <div class="hl-text">“${escapeHtml((w.tokens || []).join(' '))}”</div>
    </div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);
}

// --- Heatmap of weight trajectory --------------------------------------
function colorForWeight(w, maxAbs) {
  if (!maxAbs) return '#1d1710';
  const t = Math.max(-1, Math.min(1, w / maxAbs));
  if (t === 0) return '#1d1710';
  if (t > 0) {
    // warm: black -> #cfa348
    const r = Math.round(255 * t);
    const g = Math.round(170 * t);
    const b = Math.round(51 * t);
    return `rgb(${r},${g},${b})`;
  }
  const r = Math.round(51 * -t);
  const g = Math.round(102 * -t);
  const b = Math.round(204 * -t);
  return `rgb(${r},${g},${b})`;
}

function renderHeatmap(trajectory) {
  const ctx = heatmapEl.getContext('2d');
  const traj = trajectory.trajectory || [];
  const keys = trajectory.keys || [];
  if (!traj.length || !keys.length) {
    ctx.fillStyle = '#211a12'; ctx.fillRect(0, 0, heatmapEl.width, heatmapEl.height);
    ctx.fillStyle = '#8f8266'; ctx.font = '12px monospace';
    ctx.fillText('(no winner weights on disk yet — heatmap will populate as new generations finish)', 12, 24);
    heatStatsEl.textContent = '';
    return;
  }
  const nCols = traj.length;
  const nRows = keys.length;
  const W = heatmapEl.clientWidth || 800;
  const cellW = Math.max(2, Math.floor(W / nCols));
  const cellH = 5;
  heatmapEl.width = cellW * nCols;
  heatmapEl.height = cellH * nRows;

  let maxAbs = 0;
  for (const row of traj) {
    for (const w of row.weights) {
      if (Math.abs(w) > maxAbs) maxAbs = Math.abs(w);
    }
  }
  for (let x = 0; x < nCols; x++) {
    const row = traj[x].weights;
    for (let y = 0; y < nRows; y++) {
      ctx.fillStyle = colorForWeight(row[y], maxAbs);
      ctx.fillRect(x * cellW, y * cellH, cellW, cellH);
    }
  }
  heatStatsEl.textContent =
    `${nRows} edges × ${nCols} gens · |max|=${maxAbs.toFixed(0)}`;
}

// --- Meta-gardener log list --------------------------------------------
function renderMetaLogs(epochs) {
  metaLogListEl.innerHTML = '';
  if (!epochs.length) {
    metaLogListEl.innerHTML = '<div class="subtle">no meta-gardener logs yet.</div>';
    return;
  }
  for (const ep of epochs.slice(0, 30)) {
    const div = document.createElement('div');
    div.style.cssText = 'margin:8px 0; padding-left:8px; border-left:2px solid rgba(205,127,93,0.15);';
    const winner = ep.winner
      ? `<span style="color:var(--warm);">winner: ${ep.winner.flask}/${ep.winner.worm} @ ${(ep.winner.score || 0).toFixed(3)}</span>`
      : '';
    let body;
    if (ep.log) body = `<div class="log-box" style="margin-top:4px;">${escapeHtml(ep.log)}</div>`;
    else if (ep.log_skipped) body = `<div class="log-box skipped" style="margin-top:4px;">(rested)</div>`;
    else body = `<div class="subtle">(no log on disk)</div>`;
    div.innerHTML = `<div style="color:var(--accent); font-weight:600; font-size:12px;">epoch ${ep.epoch} ${winner}</div>${body}`;
    metaLogListEl.appendChild(div);
  }
}

// --- Bootstrap ---------------------------------------------------------
async function loadFlask(flaskName) {
  currentFlask = flaskName;
  setStatus(`loading ${flaskName}…`);
  detailCard.style.display = 'none';
  try {
    const [gensResp, traj] = await Promise.all([
      fetchJSON(`/api/generations/${flaskName}?limit=500`),
      fetchJSON(`/api/generations/${flaskName}/weights/trajectory?top_n=64`).catch(() => ({ keys: [], trajectory: [] })),
    ]);
    currentGenerations = gensResp.generations || [];
    flaskMeta.textContent =
      `${gensResp.total} generations · showing ${currentGenerations.length}`;
    renderCharts(currentGenerations);
    renderGenList(currentGenerations);
    renderHeatmap(traj);
    setStatus(`${flaskName} ready`);
  } catch (e) {
    setStatus(`error: ${e.message}`, true);
  }
}

// --- Experiment banner -------------------------------------------------
// Populated by the experiment_dropdown.js fetch via window.__experimentLoaded
// or by reading the global it stashes (in case load order is reversed).
function paintExperimentBanner(payload) {
  if (!payload) return;
  currentExperiment = payload;
  const banner = document.getElementById('experiment-banner');
  const lab = document.getElementById('experiment-banner-label');
  const blurb = document.getElementById('experiment-banner-blurb');
  const mode = document.getElementById('experiment-banner-mode');
  if (!banner) return;
  lab.textContent = payload.current_label || payload.current || '';
  blurb.textContent = payload.current_blurb || '';
  mode.textContent = `mode = ${payload.current}`;
  banner.style.display = 'block';
}

window.__experimentLoaded = paintExperimentBanner;
if (window.__experiment) paintExperimentBanner(window.__experiment);

// --- Per-worm lineage chart (Chart.js scatter with manual connector lines)
function renderLineageChart(gens) {
  const card = document.getElementById('lineage-card');
  const canvas = document.getElementById('chart-lineage');
  if (!card || !canvas) return;
  if (charts.lineage) { charts.lineage.destroy(); charts.lineage = null; }
  // Need per_worm data; bail if this is a prod flask (no per_worm field).
  const hasPerWorm = gens.some(g => Array.isArray(g.per_worm) && g.per_worm.length);
  if (!hasPerWorm) { card.style.display = 'none'; return; }
  card.style.display = 'block';

  const chrono = [...gens].reverse();
  // Build a fitness lookup: byGen.get(genNum) = {wormName: fitness}.
  const byGen = new Map();
  for (const g of chrono) {
    const m = new Map();
    for (const r of g.per_worm || []) m.set(r.name, r.fitness);
    byGen.set(g.generation, m);
  }

  const dotsElite = [];
  const dotsFresh = [];
  const connectors = []; // array of [{x,y},{x,y}] pairs for lines

  for (let i = 0; i < chrono.length; i++) {
    const g = chrono[i];
    for (const r of g.per_worm || []) {
      const pt = { x: g.generation, y: r.fitness, name: r.name };
      // is_elite info lives in g.next_gen_lineage from the PREVIOUS gen,
      // but here we just want to plot points — use rank=0..N_ELITES-1
      // as a proxy for "this is what fed forward as elite next round".
      // For visual emphasis: rank-0 worm = brightest dot.
      if (r.rank === 0) dotsElite.push(pt);
      else dotsFresh.push(pt);
    }
    // Lineage lines: gen N+1 (which is this gen, if i>0) has lineage info
    // about each slot's parent in the PREVIOUS gen.
    if (i > 0) {
      const prev = chrono[i - 1];
      const prevFit = byGen.get(prev.generation) || new Map();
      const curFit = byGen.get(g.generation) || new Map();
      for (const link of g.next_gen_lineage || []) {
        // next_gen_lineage was written at the END of gen N describing slots
        // FOR gen N+1, with name (the slot's worm name) and
        // parent_name_in_this_gen (the prev-gen worm whose genome flowed in).
        // The lineage we're rendering now connects gen N+1's worm to its
        // parent in gen N — so when reading gen g's per_worm we want
        // gen (g-1)'s next_gen_lineage. But here `g.next_gen_lineage` is
        // the lineage written for the FOLLOWING gen, not the current one.
        // So skip: lineage data is consumed on the NEXT iteration below.
      }
      // Read the prev gen's next_gen_lineage instead, which tells us where
      // each of THIS gen's worms came from.
      for (const link of prev.next_gen_lineage || []) {
        const childY = curFit.get(link.name);
        const parentY = prevFit.get(link.parent_name_in_this_gen);
        if (childY === undefined || parentY === undefined) continue;
        connectors.push({
          from: { x: prev.generation, y: parentY },
          to: { x: g.generation, y: childY },
          elite: link.is_elite,
        });
      }
    }
  }

  // Custom plugin: after Chart.js draws, draw connectors on top of dots.
  const connectorPlugin = {
    id: 'connectors',
    afterDatasetsDraw(chart) {
      const { ctx, scales } = chart;
      ctx.save();
      for (const c of connectors) {
        ctx.beginPath();
        ctx.moveTo(scales.x.getPixelForValue(c.from.x), scales.y.getPixelForValue(c.from.y));
        ctx.lineTo(scales.x.getPixelForValue(c.to.x), scales.y.getPixelForValue(c.to.y));
        ctx.strokeStyle = c.elite ? 'rgba(207,163,72,0.6)' : 'rgba(150,150,150,0.22)';
        ctx.lineWidth = c.elite ? 1.2 : 0.6;
        ctx.stroke();
      }
      ctx.restore();
    },
  };

  charts.lineage = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'gen winner (rank 0)', data: dotsElite, backgroundColor: '#cfa348', borderColor: '#cfa348', pointRadius: 4, pointHoverRadius: 6 },
        { label: 'other worms', data: dotsFresh, backgroundColor: 'rgba(150,255,200,0.7)', borderColor: 'rgba(120,210,170,0.9)', pointRadius: 3, pointHoverRadius: 5 },
      ],
    },
    options: {
      ...chartOptsWithLegend('per-worm fitness with lineage lines (orange = elite carry, grey = fresh NES child)'),
      parsing: false,
      plugins: {
        legend: { display: true, position: 'top', align: 'end', labels: { color: '#b3a789', font: { size: 9 }, boxWidth: 8, boxHeight: 8 } },
        title: { display: true, text: 'per-worm fitness with lineage lines', color: '#cfa348', font: { size: 11, weight: 'normal' } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const p = ctx.raw;
              return `${p.name || ''} · gen ${p.x} · fitness ${p.y.toFixed(2)}`;
            },
          },
        },
      },
    },
    plugins: [connectorPlugin],
  });
}

// --- POS-breakdown stacked area ----------------------------------------
// Warm categorical set (Tobacco & Ochre restyle): content words get the
// strong hues (ochre/terracotta/olive/rose), function words recede into
// translucent earth tones. All distinguishable on the tobacco ground.
const POS_COLORS = {
  NOUN: '#cfa348', VERB: '#cd7f5d', ADJ: '#a8a468', ADV: '#b08272',
  DET: 'rgba(143,130,102,0.7)', ADP: 'rgba(125,157,127,0.6)',
  PRON: 'rgba(176,130,150,0.6)', PRT: 'rgba(150,140,120,0.55)',
  CONJ: 'rgba(180,166,130,0.55)', other: 'rgba(90,82,66,0.5)',
};
const POS_KEYS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'ADP', 'PRON', 'PRT', 'CONJ', 'other'];

function renderPOSChart(gens) {
  const card = document.getElementById('pos-card');
  const canvas = document.getElementById('chart-pos');
  if (!card || !canvas) return;
  if (charts.pos) { charts.pos.destroy(); charts.pos = null; }
  const hasPOS = gens.some(g => g.pos_totals && Object.keys(g.pos_totals).length);
  if (!hasPOS) { card.style.display = 'none'; return; }
  card.style.display = 'block';

  const chrono = [...gens].reverse();
  const labels = chrono.map(g => g.generation);
  const datasets = POS_KEYS.map(tag => ({
    label: tag,
    data: chrono.map(g => {
      const pt = g.pos_totals || {};
      if (tag === 'other') {
        const known = new Set(POS_KEYS.filter(k => k !== 'other'));
        let s = 0;
        for (const [k, v] of Object.entries(pt)) if (!known.has(k)) s += v;
        return s;
      }
      return pt[tag] || 0;
    }),
    backgroundColor: POS_COLORS[tag],
    borderColor: POS_COLORS[tag],
    fill: true,
    pointRadius: 0,
    tension: 0.2,
  }));

  charts.pos = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'top', align: 'end', labels: { color: '#b3a789', font: { size: 9 }, boxWidth: 8, boxHeight: 8 } },
        title: { display: true, text: 'POS breakdown of eaten words (stacked, all worms in flask)', color: '#cfa348', font: { size: 11, weight: 'normal' } },
      },
      scales: {
        x: { ticks: { color: '#8f8266', font: { size: 10 } }, grid: { color: 'rgba(205,127,93,0.06)' } },
        y: { stacked: true, ticks: { color: '#8f8266', font: { size: 10 } }, grid: { color: 'rgba(205,127,93,0.06)' } },
      },
    },
  });
}

async function init() {
  let idx;
  try {
    idx = await fetchJSON('/api/generations');
  } catch (e) {
    setStatus(`error: ${e.message}`, true);
    return;
  }
  const flasks = idx.flasks || [];
  if (!flasks.length) {
    setStatus('no generation data on disk yet — run for at least one rollover.');
    return;
  }
  // A poetry process serves ALL 8 flasks (it reads its sibling processes'
  // data dirs read-only), so this dropdown pages through the whole run — not
  // just the 2 flasks belonging to whichever port you happen to be on.
  // Each option shows σ and the σ-control scheme, so you can see at a glance
  // which arm is frozen and which is ratcheting.
  flaskSelect.innerHTML = '';
  for (const f of flasks) {
    const opt = document.createElement('option');
    opt.value = f.name;
    const gen = f.current_generation || f.n_generations;
    const sigma = (typeof f.sigma === 'number') ? `, σ=${f.sigma.toFixed(3)}` : '';
    const scheme = f.sigma_scheme ? ` ${f.sigma_scheme}` : '';
    // Corpus title in the dropdown so three-text deployments read as
    // "flask_2 - Tao Teh King" (textContent: safe sink).
    const corpus = f.corpus ? ` \u00b7 ${f.corpus}` : '';
    opt.textContent = `${f.label || f.name}${corpus} (gen ${gen}${sigma}${scheme})`;
    flaskSelect.appendChild(opt);
  }
  flaskSelect.addEventListener('change', e => loadFlask(e.target.value));
  await loadFlask(flasks[0].name);

  try {
    const meta = await fetchJSON('/api/generations/meta/index');
    renderMetaLogs(meta.epochs || []);
  } catch (e) {
    metaLogListEl.innerHTML = `<div class="subtle" style="color:#cd5d4a;">meta logs unavailable: ${e.message}</div>`;
  }
}

init();
