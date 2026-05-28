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
const heatmapEl = document.getElementById('heatmap');
const heatStatsEl = document.getElementById('heat-stats');
const metaLogListEl = document.getElementById('meta-log-list');

const charts = { best: null, avg: null, sigma: null };
let currentFlask = null;
let currentGenerations = []; // newest-first

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

function setStatus(text, isError) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? '#f99' : '';
}

function chartOpts(title) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: title, color: '#6f9', font: { size: 11, weight: 'normal' } },
    },
    scales: {
      x: { ticks: { color: '#5a5', font: { size: 10 } }, grid: { color: 'rgba(100,200,255,0.06)' } },
      y: { ticks: { color: '#5a5', font: { size: 10 } }, grid: { color: 'rgba(100,200,255,0.06)' } },
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
const SIGMA_INIT = 0.3;
const SIGMA_MAX = 1.0;

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
    labels: { color: '#9c9', font: { size: 9 }, boxWidth: 8, boxHeight: 2 },
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
    data: { labels, datasets: [lineDataset('best fitness', '#6f9', best)] },
    options: chartOpts('best fitness per generation'),
  });
  charts.avg = new Chart(document.getElementById('chart-avg'), {
    type: 'line',
    data: { labels, datasets: [lineDataset('avg fitness', '#fc6', avg)] },
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
        lineDataset('σ used', '#f66', sigmas),
        lineDataset('σ stability (rolling std, 10 gens)', '#9cf', sigmaStd10),
        constDataset(`max (${SIGMA_MAX})`, 'rgba(255,102,102,0.45)', SIGMA_MAX, sigmas.length, [2, 4]),
        constDataset(`init (${SIGMA_INIT})`, 'rgba(102,255,153,0.45)', SIGMA_INIT, sigmas.length, [2, 4]),
        constDataset(`min (${SIGMA_MIN})`, 'rgba(150,150,150,0.4)', SIGMA_MIN, sigmas.length, [2, 4]),
      ],
    },
    options: chartOptsWithLegend('σ (learning rate) — is it steady?'),
  });
}

function renderGenList(gens) {
  genListEl.innerHTML = '';
  const header = document.createElement('div');
  header.className = 'gen-row';
  header.style.cssText = 'background:rgba(100,200,255,0.07); font-weight:600; cursor:default;';
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
  let data;
  try {
    data = await fetchJSON(`/api/generations/${currentFlask}/${genNum}`);
  } catch (e) {
    detailLog.innerHTML = `<div class="subtle" style="color:#f99;">error: ${e.message}</div>`;
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
  const top = allWindows.slice(0, 12);
  detailWindows.innerHTML = top.length
    ? top.map(w => `
        <div class="window">
          <span class="scores">E=${w.emotional} C=${w.coherence}</span>
          <span style="color:#9c9;">${w.worm}</span>
          “${escapeHtml((w.tokens || []).join(' '))}”
        </div>`).join('')
    : '<div class="subtle">no scored windows for this generation.</div>';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);
}

// --- Heatmap of weight trajectory --------------------------------------
function colorForWeight(w, maxAbs) {
  if (!maxAbs) return '#111';
  const t = Math.max(-1, Math.min(1, w / maxAbs));
  if (t === 0) return '#111';
  if (t > 0) {
    // warm: black -> #ffaa33
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
    ctx.fillStyle = '#001'; ctx.fillRect(0, 0, heatmapEl.width, heatmapEl.height);
    ctx.fillStyle = '#5a5'; ctx.font = '12px monospace';
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
    div.style.cssText = 'margin:8px 0; padding-left:8px; border-left:2px solid rgba(100,200,255,0.15);';
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
  flaskSelect.innerHTML = '';
  for (const f of flasks) {
    const opt = document.createElement('option');
    opt.value = f.name;
    opt.textContent = `${f.name} (gen ${f.current_generation || f.n_generations})`;
    flaskSelect.appendChild(opt);
  }
  flaskSelect.addEventListener('change', e => loadFlask(e.target.value));
  await loadFlask(flasks[0].name);

  try {
    const meta = await fetchJSON('/api/generations/meta/index');
    renderMetaLogs(meta.epochs || []);
  } catch (e) {
    metaLogListEl.innerHTML = `<div class="subtle" style="color:#f99;">meta logs unavailable: ${e.message}</div>`;
  }
}

init();
