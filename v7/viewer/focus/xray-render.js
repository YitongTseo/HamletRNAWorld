// ---------------------------------------------------------------------------
// X-ray worm view
// ---------------------------------------------------------------------------
// Overlay the 301 connectome neurons onto the actual moving worm body
// instead of a static anatomical layout. The worm midline streams in
// live from the WS; each neuron has a precomputed (axial∈[0,1],
// lateral∈[-1,+1]) anchor from cache/neuron_body_coords.json. At render
// time we sample the midline at the neuron's axial position, get the
// local tangent + perpendicular, and place the neuron at
// midline_point + perpendicular × lateral × body_half_width.
//
// Net effect: when the worm twists, the neurons twist with it — like
// peering at the wiggling animal through a soft X-ray.
//
// This module is the reusable form: `drawXRay(ctx, screenRect, opts)`
// takes its target context + region explicitly, so the same code paints
// into the bottom-right network panel (Task 5) and the touch-magnifier
// (Task 17). The panel passes the worm midline through `getMidline()`
// from worm-render; the magnifier will pass the same.

import { getMidline } from './worm-render.js';

const XRAY_BODY_HALF_WIDTH = 18;   // panel-px to one side at midbody
const XRAY_HEAD_THIN_FRAC = 0.92;  // fraction along axis where body still wider
const XRAY_TAIL_THIN_FRAC = 0.85;

// Cache: contour points so we don't keep allocating arrays
const _xrayBuf = { contourTop: [], contourBot: [], neuronPos: new Map() };

// Class-color palette shared between the static graph view and the x-ray
// view so visitors learn one color → class association.
const NEURON_CLASS_PALETTE = [
  { key: 'chemo',  color: 'rgba(224,196,143,0.85)', label: 'chemosensory' },
  { key: 'sensory',color: 'rgba(100,160,255,0.7)', label: 'sensory' },
  { key: 'motor',  color: 'rgba(255,150,40,0.85)', label: 'motor' },
  { key: 'inter',  color: 'rgba(207,163,72,0.85)', label: 'interneuron' },
  { key: 'muscle', color: 'rgba(255,220,40,0.85)', label: 'muscle' },
  { key: 'firing', color: '#cfa348',               label: 'firing' },
];

function _fitMidlineToPanel(midline, panelLeft, panelTop, panelW, panelH, headerH) {
  // Returns { scale, points } — points = midline mapped into the rect
  // coords. Preserves aspect, centers, leaves margin for neurons sitting
  // outside the body line (head sensilla, tail sensory).
  if (!midline.length) return null;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of midline) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const w = Math.max(1, maxX - minX);
  const h = Math.max(1, maxY - minY);
  const MARGIN = 28;  // leave room around the worm for protruding sensilla
  const bodyW = panelW;
  const bodyH = panelH - headerH;
  const usableW = bodyW - 2 * MARGIN;
  const usableH = bodyH - 2 * MARGIN;
  const scale = Math.min(usableW / w, usableH / h);
  const offX = panelLeft + MARGIN + (usableW - w * scale) / 2 - minX * scale;
  const offY = panelTop + headerH + MARGIN + (usableH - h * scale) / 2 - minY * scale;
  const points = midline.map(([x, y]) => [x * scale + offX, y * scale + offY]);
  return { scale, points };
}

function _midlineSampleAt(points, axial) {
  // Map axial∈[0,1] to a position + tangent on the midline polyline.
  if (points.length < 2) return null;
  const t = Math.max(0, Math.min(0.999999, axial)) * (points.length - 1);
  const i = t | 0;
  const f = t - i;
  const [ax, ay] = points[i];
  const [bx, by] = points[i + 1] || points[i];
  const px = ax + (bx - ax) * f;
  const py = ay + (by - ay) * f;
  // Tangent: smooth a tiny bit by using neighbors when available.
  const i0 = Math.max(0, i - 1);
  const i1 = Math.min(points.length - 1, i + 2);
  const [tx0, ty0] = points[i0];
  const [tx1, ty1] = points[i1];
  let tx = tx1 - tx0, ty = ty1 - ty0;
  const tlen = Math.hypot(tx, ty) || 1;
  tx /= tlen; ty /= tlen;
  // Perpendicular (rotate 90° CCW): (-ty, tx). Convention: lateral > 0
  // means the worm's left side. The worm advances along its +tangent so
  // (-ty, tx) is its left in screen coords (y goes down).
  return { x: px, y: py, tx, ty, nx: -ty, ny: tx };
}

function _bodyHalfWidthAt(axial) {
  // Smooth taper at head and tail so the worm looks like a worm.
  const a = Math.max(0, Math.min(1, axial));
  let factor = 1.0;
  if (a < 0.08) factor = a / 0.08;
  else if (a > XRAY_TAIL_THIN_FRAC) factor = (1 - a) / (1 - XRAY_TAIL_THIN_FRAC);
  else if (a > XRAY_HEAD_THIN_FRAC) factor = 0.95;
  return XRAY_BODY_HALF_WIDTH * factor;
}

function _tracePolyline(ctx, pts, startNew) {
  // Trace pts as a smooth open curve using the quadratic-midpoint method:
  // each original vertex becomes a control point, the curve passes through
  // the midpoints between vertices. Rounds off the faceting/kinks that a
  // straight lineTo chain would show on a bending body.
  if (!pts.length) return;
  if (startNew) ctx.moveTo(pts[0][0], pts[0][1]);
  else ctx.lineTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2;
    const my = (pts[i][1] + pts[i + 1][1]) / 2;
    ctx.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last[0], last[1]);
}

function drawNeuronLegend(ctx, x0, yBase) {
  // Compact legend: dot + label, packed to fit horizontally starting at x0.
  ctx.font = '8px ui-monospace,monospace';
  const SPC = 64;
  const LX = x0 + 8;
  for (let i = 0; i < NEURON_CLASS_PALETTE.length; i++) {
    const x = LX + i * SPC;
    ctx.beginPath();
    ctx.arc(x + 4, yBase, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = NEURON_CLASS_PALETTE[i].color;
    ctx.fill();
    ctx.fillStyle = 'rgba(207,163,72,0.7)';
    ctx.fillText(NEURON_CLASS_PALETTE[i].label, x + 10, yBase + 3);
  }
}

// Paint the x-ray view into `ctx`, inside the screen-space rect
// `screenRect = {x, y, w, h}` (CSS pixels of the target canvas/region).
//
// opts:
//   xrayLabelsVisible  — show neuron name labels
//   neuronBodyCoords   — { neurons: { name: {axial, lateral, dv} }, n_neurons }
//   graph              — { neurons, fireThreshold, chemosensorySet, motorSet,
//                          muscleSet, sensorySet, _chemoNames?, ... }
//   neuronActivity     — { neuronName: charge }
//   worldToScreen      — (optional) Used by the magnifier (Task 17). When
//                        provided, the midline is projected through this
//                        function to map world coords to ctx-local coords
//                        — so the lens shows the neurons in their actual
//                        page-space position under the worm body. Header
//                        strip + legend are also suppressed since the lens
//                        is meant to be a clean "peek" surface. When NOT
//                        provided (the network-panel path), the existing
//                        `_fitMidlineToPanel` behavior is preserved: fit
//                        the whole worm into the rect with a header strip.
//
// Returns nothing; mutates `ctx`. Caller is expected to have cleared the
// region beforehand if it wants a fresh frame.
export function drawXRay(ctx, screenRect, opts) {
  const {
    xrayLabelsVisible = false,
    neuronBodyCoords,
    graph,
    neuronActivity = {},
    worldToScreen,   // optional — magnifier path; see comment above
  } = opts;
  if (!neuronBodyCoords || !graph) return;
  const latestMidline = getMidline();
  if (latestMidline.length < 2) return;

  const { x: X0, y: Y0, w: W, h: H } = screenRect;
  const lensMode = typeof worldToScreen === 'function';

  ctx.clearRect(X0, Y0, W, H);

  if (!lensMode) {
    // Header strip (panel-only — the lens skips this for a clean look).
    ctx.fillStyle = '#e0c48f';
    ctx.font = 'bold 11px ui-monospace, monospace';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText('● LIVE BODY CONNECTOME', X0 + 8, Y0 + 6);
    ctx.fillStyle = '#b3a789';
    ctx.font = '9px ui-monospace, monospace';
    ctx.fillText(`${neuronBodyCoords.n_neurons} neurons mapped to wormbody`, X0 + 8, Y0 + 19);
    ctx.fillStyle = '#b3a789';
    ctx.textAlign = 'right';
    ctx.fillText("'x' → connectome graph · 'l' labels" + (xrayLabelsVisible ? ' ✓' : ''), X0 + W - 8, Y0 + 19);
    ctx.textAlign = 'left';

    drawNeuronLegend(ctx, X0, Y0 + 36);
  }

  // Project the midline to ctx-local coords. Two paths:
  //   - Panel path: fit world-space worm bbox into the rect (existing math).
  //     `bodyScale = 1` keeps the panel-tuned XRAY_BODY_HALF_WIDTH = 18 px.
  //   - Lens path: project each world point through the caller's
  //     `worldToScreen` so the worm appears at its actual on-screen position.
  //     `bodyScale` is computed from the world-vs-screen midline arc length
  //     so the drawn silhouette half-width matches what the underlying worm
  //     mesh occupies on screen (otherwise the silhouette is a panel-sized
  //     18 px thick worm floating across a viewport-sized animal).
  // `points` is always a list of [x,y] in the ctx's drawing coord system.
  let points, bodyScale = 1;
  if (lensMode) {
    points = latestMidline.map(([wx, wy]) => worldToScreen(wx, wy));
    // Empirical scale: ratio of projected midline arc length to world arc
    // length. Robust to camera scale + DPR + responsive frustum changes.
    let worldLen = 0, screenLen = 0;
    for (let i = 1; i < latestMidline.length; i++) {
      const [wx0, wy0] = latestMidline[i - 1], [wx1, wy1] = latestMidline[i];
      const [sx0, sy0] = points[i - 1], [sx1, sy1] = points[i];
      worldLen  += Math.hypot(wx1 - wx0, wy1 - wy0);
      screenLen += Math.hypot(sx1 - sx0, sy1 - sy0);
    }
    if (worldLen > 0) {
      // The world worm has radius ~22 (WORM_BASE_RADIUS in worm-render.js);
      // XRAY_BODY_HALF_WIDTH is 18 panel-px tuned for that radius. So the
      // ratio (screenLen/worldLen) is the world→screen scale; multiply by
      // (22/18) ≈ 1.22 to roughly match the actual worm-body radius on screen.
      bodyScale = (screenLen / worldLen) * (22 / 18);
    }
  } else {
    const HEADER_H = 50;
    const fit = _fitMidlineToPanel(latestMidline, X0, Y0, W, H, HEADER_H);
    if (!fit) return;
    points = fit.points;
  }

  // ── Body outline (two parallel curves) ──
  // Sample the silhouette densely so the edge reads as a smooth curve
  // rather than a faceted polygon, and trace it with quadratic segments
  // (see _tracePolyline) so hard bends stay rounded instead of kinking.
  const samples = 200;
  const top = [], bot = [];
  for (let i = 0; i <= samples; i++) {
    const a = i / samples;
    const m = _midlineSampleAt(points, a);
    if (!m) continue;
    const r = _bodyHalfWidthAt(a) * bodyScale;
    top.push([m.x + m.nx * r, m.y + m.ny * r]);
    bot.push([m.x - m.nx * r, m.y - m.ny * r]);
  }

  // Filled body, soft outline — smooth curve through both edges.
  ctx.fillStyle = 'rgba(33, 26, 18, 0.6)';
  ctx.strokeStyle = 'rgba(120, 220, 170, 0.55)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  _tracePolyline(ctx, top, true);
  _tracePolyline(ctx, bot.slice().reverse(), false);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Midline (faint)
  ctx.strokeStyle = 'rgba(100, 200, 255, 0.18)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    if (i === 0) ctx.moveTo(points[i][0], points[i][1]);
    else ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.stroke();

  // Head dot for orientation
  const head = points[0];
  ctx.fillStyle = 'rgba(255, 230, 150, 0.6)';
  ctx.beginPath();
  ctx.arc(head[0], head[1], 4 * (lensMode ? Math.min(2.5, Math.max(0.8, bodyScale * 0.5)) : 1), 0, Math.PI * 2);
  ctx.fill();

  // ── Neurons ──
  const fireThr = graph.fireThreshold || 30;
  // graph.{chemo,motor,muscle,sensory}Set hold INTEGER indices into
  // graph.neurons. Build name-keyed sets once and stash on `graph` for
  // future frames.
  if (!graph._chemoNames) {
    const namesByIdx = graph.neurons;
    const mk = (s) => new Set([...s].map(i => namesByIdx[i]));
    graph._chemoNames = mk(graph.chemosensorySet);
    graph._motorNames = mk(graph.motorSet);
    graph._muscleNames = mk(graph.muscleSet);
    graph._sensoryNames = mk(graph.sensorySet);
  }
  const chemoSet = graph._chemoNames;
  const motorSet = graph._motorNames;
  const muscleSet = graph._muscleNames;
  const sensorySet = graph._sensoryNames;

  // Two passes: first faint-all, then bright firing on top.
  // In lens mode, the neuron dot radii are scaled with `bodyScale` so a
  // big-on-screen worm gets proportionally readable neuron dots — but capped
  // so they don't balloon past usefulness on extreme zooms.
  const dotScale = lensMode ? Math.min(3.5, Math.max(1.0, bodyScale * 0.6)) : 1;
  const nbody = neuronBodyCoords.neurons;
  for (const [name, anatomy] of Object.entries(nbody)) {
    const m = _midlineSampleAt(points, anatomy.axial);
    if (!m) continue;
    const halfW = _bodyHalfWidthAt(anatomy.axial) * bodyScale;
    // lateral∈[-1,+1] already maps the OpenWorm position range to a unit
    // half-width; 0.85 keeps even the most-lateral neurons safely inside
    // the body silhouette.
    const off = anatomy.lateral * halfW * 0.85;
    const nx = m.x + m.nx * off;
    const ny = m.y + m.ny * off;

    const charge = neuronActivity[name] || 0;
    const firing = charge > fireThr;

    // Base color by neuron class.
    let baseHue = 210, baseAlpha = 0.18, dotR = 1.6;
    if (chemoSet && chemoSet.has(name))      { baseHue = 195; baseAlpha = 0.35; dotR = 2.0; }
    else if (sensorySet && sensorySet.has(name)) { baseHue = 220; baseAlpha = 0.25; dotR = 1.8; }
    else if (motorSet && motorSet.has(name)) { baseHue = 28;  baseAlpha = 0.25; dotR = 1.8; }
    else if (muscleSet && muscleSet.has(name)) continue; // muscles handled by main worm, skip
    dotR *= dotScale;

    ctx.fillStyle = `hsla(${baseHue}, 70%, 75%, ${baseAlpha})`;
    ctx.beginPath();
    ctx.arc(nx, ny, dotR, 0, Math.PI * 2);
    ctx.fill();

    if (firing) {
      // Glow + bright dot when firing
      const glowR = dotR + Math.min(7, charge / 12);
      const grd = ctx.createRadialGradient(nx, ny, 0.5, nx, ny, glowR);
      grd.addColorStop(0, `hsla(${baseHue}, 95%, 75%, 0.95)`);
      grd.addColorStop(1, `hsla(${baseHue}, 95%, 60%, 0)`);
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(nx, ny, glowR, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `hsla(${baseHue}, 100%, 88%, 1)`;
      ctx.beginPath();
      ctx.arc(nx, ny, dotR + 0.7, 0, Math.PI * 2);
      ctx.fill();
    }

    // Optional name label (toggled by 'l'). Slight offset perpendicular
    // to the local body axis on whichever side the neuron sits, so the
    // text floats off-body and doesn't overlap the dot/silhouette.
    if (xrayLabelsVisible) {
      const labelOffset = (anatomy.lateral >= 0 ? 1 : -1) * 6;
      const lx = nx + m.nx * labelOffset;
      const ly = ny + m.ny * labelOffset;
      ctx.font = '7px ui-monospace, monospace';
      ctx.textBaseline = 'middle';
      ctx.textAlign = anatomy.lateral >= 0 ? 'left' : 'right';
      // Slightly more legible when firing.
      ctx.fillStyle = firing
        ? `hsla(${baseHue}, 100%, 92%, 0.95)`
        : `hsla(${baseHue}, 60%, 80%, 0.55)`;
      ctx.fillText(name, lx, ly);
    }
  }
  // Reset text alignment for downstream drawing.
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
}

export { NEURON_CLASS_PALETTE, drawNeuronLegend, _xrayBuf };
