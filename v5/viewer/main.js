import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// ---------------------------------------------------------------------------
// Scene / camera / renderer
// ---------------------------------------------------------------------------
const canvas = document.getElementById('c');
const hud = document.getElementById('hud');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
// ACES tone-mapping squashes the bloom highlights into a more cinematic curve
// (otherwise the rim glow blows out flat). The exposure tunes overall
// brightness; we want the worm bright on a near-black background.
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.7;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// World coords arrive in worm-sim "pixel" units (~1600×1000). The camera is
// y-flipped (top=0, bottom=WORLD_H) so we don't have to invert worm-sim's
// body math. That flip reverses screen-space winding for all triangles, which
// would normally backface-cull half of any flat geometry — we use DoubleSide
// in the affected materials to dodge this.
const WORLD_W = 1600, WORLD_H = 1000;
const camera = new THREE.OrthographicCamera(0, WORLD_W, WORLD_H, 0, -1000, 1000);
camera.position.z = 10;

// ---------------------------------------------------------------------------
// Postprocessing — UnrealBloomPass gives the GFP halo around the silhouette.
// Threshold is tuned so only the bright rim glow blooms, not the dim core.
// ---------------------------------------------------------------------------
const composer = new EffectComposer(renderer);
composer.setPixelRatio(window.devicePixelRatio);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  /* strength  */ 0.22,
  /* radius    */ 0.55,
  /* threshold */ 0.9,
);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  composer.setSize(w, h);
  bloom.setSize(w, h);

  const worldAspect = WORLD_W / WORLD_H;
  const viewAspect = w / h;
  let vw = WORLD_W, vh = WORLD_H;
  if (viewAspect > worldAspect) vw = WORLD_H * viewAspect;
  else vh = WORLD_W / viewAspect;
  const cx = WORLD_W / 2, cy = WORLD_H / 2;
  camera.left = cx - vw / 2;
  camera.right = cx + vw / 2;
  camera.top = cy - vh / 2;     // y-down (see camera comment above)
  camera.bottom = cy + vh / 2;
  camera.updateProjectionMatrix();
}
resize();
window.addEventListener('resize', resize);

// ---------------------------------------------------------------------------
// Body shader — fresnel rim glow imitating GFP at the silhouette.
//
// ShaderMaterial auto-injects the standard attributes/uniforms (position,
// normal, uv, modelViewMatrix, etc.), so we only declare our custom uniforms
// and varyings.
// ---------------------------------------------------------------------------
const bodyVert = /* glsl */`
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec2 vUv;
varying float vWorldZ;

void main() {
  vUv = uv;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vViewDir = normalize(-mv.xyz);
  vNormal = normalize(normalMatrix * normal);
  vWorldZ = position.z;
  gl_Position = projectionMatrix * mv;
}
`;

const bodyFrag = /* glsl */`
precision highp float;

varying vec3 vNormal;
varying vec3 vViewDir;
varying vec2 vUv;
varying float vWorldZ;

uniform vec3 uColor;
uniform float uTime;
uniform float uRimPower;
uniform float uCoreBoost;
uniform float uRimBoost;
uniform float uCenterAlpha;
uniform float uEdgeAlpha;

// Cheap hash → smooth 1D noise.
float hash(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
float noise(float x) {
  float i = floor(x), f = fract(x);
  float a = hash(i), b = hash(i + 1.0);
  float u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u);
}

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewDir);
  float ndotv = clamp(abs(dot(N, V)), 0.0, 1.0);
  float rim = pow(1.0 - ndotv, uRimPower);

  // Subtle long-axis fluctuation so the body wall isn't dead flat. Real
  // confocal images have visible granular texture from the cuticle and
  // hypodermal cells.
  float n = noise(vUv.x * 35.0 + uTime * 0.4) * 0.5
          + noise(vUv.x * 110.0 - uTime * 0.9) * 0.3;
  vec3 core = uColor * (uCoreBoost + 0.18 * n);

  // Rim is pushed past 1.0 in linear space so the bloom pass picks it up.
  vec3 rimGlow = uColor * uRimBoost * rim;

  // Confocal-like focal-plane intensity falloff: surfaces facing "up" out
  // of the imaging plane glow slightly more.
  float topBias = clamp(vWorldZ * 0.03 + 0.55, 0.0, 1.0);

  vec3 col = core * topBias + rimGlow;
  // Translucent body wall — opaque at silhouette so we see the worm shape,
  // mostly transparent over the midline so the inner organs show through.
  float alpha = mix(uCenterAlpha, uEdgeAlpha, rim);
  gl_FragColor = vec4(col, alpha);
}
`;

const bodyMaterial = new THREE.ShaderMaterial({
  vertexShader: bodyVert,
  fragmentShader: bodyFrag,
  uniforms: {
    uColor: { value: new THREE.Color(0x44ff77) },
    uTime: { value: 0 },
    uRimPower: { value: 2.2 },
    uCoreBoost: { value: 0.12 },   // dim body interior — let organs show
    uRimBoost: { value: 0.95 },    // softer silhouette glow
    uCenterAlpha: { value: 0.18 }, // mostly transparent at midline
    uEdgeAlpha: { value: 0.7 },    // less-than-fully-opaque silhouette
  },
  side: THREE.DoubleSide,
  transparent: true,
  depthWrite: false,
});

// ---------------------------------------------------------------------------
// Worm body geometry — a custom skinned tube along a 2D midline polyline.
// Radius varies along the body length via bodyProfile() so the worm has a
// distinct head bulge and a tapered tail (matches real C. elegans anatomy).
// ---------------------------------------------------------------------------
function bodyProfile(s) {
  // s in [0, 1]: 0 = head, 1 = tail. Real C. elegans is rounded at the head
  // (anterior pharyngeal bulb) and tapers to a fine point at the tail.
  const headLen = 0.06;
  const tailStart = 0.72;
  let r;
  if (s < headLen) {
    // Smooth-step from 0 to ~1, with a sqrt curve so the very tip is rounded
    // rather than knife-pointed.
    const t = s / headLen;
    r = Math.pow(t, 0.4);
  } else if (s > tailStart) {
    // Concave taper to a fine point.
    const t = (1 - s) / (1 - tailStart);
    r = Math.pow(t, 1.2);
  } else {
    r = 1.0;
  }
  // Subtle gravid-hermaphrodite mid-body bulge.
  const bulge = 1.0 + 0.07 * Math.sin(s * Math.PI);
  return r * bulge;
}

function buildWormGeometry(midline, baseRadius, profile = bodyProfile, radial = 14) {
  const n = midline.length;
  const positions = new Float32Array(n * radial * 3);
  const normals = new Float32Array(n * radial * 3);
  const uvs = new Float32Array(n * radial * 2);
  const indices = [];

  for (let i = 0; i < n; i++) {
    // Tangent (forward direction along the midline).
    let tx, ty;
    if (i === 0) {
      tx = midline[1][0] - midline[0][0];
      ty = midline[1][1] - midline[0][1];
    } else if (i === n - 1) {
      tx = midline[n - 1][0] - midline[n - 2][0];
      ty = midline[n - 1][1] - midline[n - 2][1];
    } else {
      tx = midline[i + 1][0] - midline[i - 1][0];
      ty = midline[i + 1][1] - midline[i - 1][1];
    }
    const tlen = Math.hypot(tx, ty) || 1;
    tx /= tlen; ty /= tlen;
    // In-plane normal (left of the tangent, in 2D).
    const nx = -ty, ny = tx;
    // Binormal is the world Z axis (we lift the tube into 3D so fresnel works).

    const s = i / (n - 1);
    const r = baseRadius * profile(s);
    const mx = midline[i][0], my = midline[i][1];

    for (let j = 0; j < radial; j++) {
      const theta = (j / radial) * Math.PI * 2;
      const c = Math.cos(theta), si = Math.sin(theta);
      // Radial direction = c * inplane-normal + si * binormal(z).
      const rx = c * nx;
      const ry = c * ny;
      const rz = si;
      const idx = (i * radial + j) * 3;
      positions[idx + 0] = mx + rx * r;
      positions[idx + 1] = my + ry * r;
      positions[idx + 2] = rz * r;
      normals[idx + 0] = rx;
      normals[idx + 1] = ry;
      normals[idx + 2] = rz;
      const uv = (i * radial + j) * 2;
      uvs[uv + 0] = s;
      uvs[uv + 1] = j / radial;
    }
  }
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < radial; j++) {
      const a = i * radial + j;
      const b = i * radial + ((j + 1) % radial);
      const c = (i + 1) * radial + ((j + 1) % radial);
      const d = (i + 1) * radial + j;
      indices.push(a, b, d, b, c, d);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geom.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(indices);
  return geom;
}

// ---------------------------------------------------------------------------
// Internal organs — a brighter inner "spine" tube that shows through the
// translucent body wall. This is what gives the GFP look its characteristic
// "see the guts glowing" appearance: in real confocal images the intestinal
// granules and gonad arms are visible through the body wall.
// ---------------------------------------------------------------------------
function organProfile(s) {
  // Inner organ profile: pinched at head and tail, bulgy through mid-body
  // (where the intestine + gonad arms run). Stops short of the body extents.
  if (s < 0.10) return Math.pow(s / 0.10, 0.7) * 0.55;
  if (s > 0.85) return Math.pow((1 - s) / 0.15, 1.4) * 0.4;
  // Mid-body: undulating thickness — mimics the irregular gonad/intestine
  // boundary visible in confocal images.
  return 0.50 + 0.10 * Math.sin(s * Math.PI * 4.0)
              + 0.06 * Math.sin(s * Math.PI * 11.0);
}

const organVert = bodyVert; // same vertex shader: world-space normal + view dir.
const organFrag = /* glsl */`
precision highp float;

varying vec3 vNormal;
varying vec3 vViewDir;
varying vec2 vUv;
varying float vWorldZ;

uniform vec3 uOrganColor;
uniform float uTime;
uniform float uBoost;

float hash(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
float noise(float x) {
  float i = floor(x), f = fract(x);
  float a = hash(i), b = hash(i + 1.0);
  float u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u);
}

void main() {
  // Granular intestinal/gonad pattern along the body length.
  float gran = noise(vUv.x * 90.0 + uTime * 0.3) * 0.6
             + noise(vUv.x * 280.0 - uTime * 0.7) * 0.3
             + noise(vUv.x * 18.0 + uTime * 0.1) * 0.4;
  // A brighter "midline focal-plane" hotspot near vUv.y == 0.25 / 0.75
  // (top/bottom of the cross-section, where the focal plane sits in the
  // confocal acquisition).
  float focal = exp(-pow((vUv.y - 0.25) * 8.0, 2.0))
              + exp(-pow((vUv.y - 0.75) * 8.0, 2.0));

  float intensity = uBoost * (0.4 + 0.6 * gran) * (0.5 + focal);
  vec3 col = uOrganColor * intensity;
  gl_FragColor = vec4(col, 1.0);
}
`;

const organMaterial = new THREE.ShaderMaterial({
  vertexShader: organVert,
  fragmentShader: organFrag,
  uniforms: {
    uOrganColor: { value: new THREE.Color(0x80ffaa) },
    uTime: { value: 0 },
    uBoost: { value: 1.1 },
  },
  side: THREE.DoubleSide,
});

// ---------------------------------------------------------------------------
// Pharynx + intestinal granules — small bright spheres positioned at fixed
// fractional body-length anchors. Additively blended and depth-test off, so
// they punch through the translucent body wall regardless of z-order.
//
// In real GFP confocal images of C. elegans, the brightest internal features
// are: anterior pharyngeal bulb (s≈0.04), terminal bulb (s≈0.10), then a
// scattered field of intestinal granules through mid-body. We approximate
// with a hand-picked anchor set.
// ---------------------------------------------------------------------------
const FOCI = [
  // [s along body, scale factor (× WORM_BASE_RADIUS), brightness factor, lateral offset]
  [0.020, 0.55, 1.5, 0.0],   // anterior pharyngeal bulb
  [0.075, 0.65, 1.7, 0.0],   // terminal pharyngeal bulb (the bright "head" blob)
  [0.18, 0.30, 0.9, 0.20],   // upper intestinal granule
  [0.24, 0.35, 1.0, -0.25],
  [0.32, 0.32, 0.9, 0.15],
  [0.40, 0.38, 1.1, -0.10],
  [0.48, 0.45, 1.3, 0.0],    // approximate vulval region (bright spot in
                             //   confocal of gravid hermaphrodites)
  [0.56, 0.36, 1.0, 0.20],
  [0.64, 0.32, 0.95, -0.15],
  [0.72, 0.28, 0.85, 0.10],
  [0.80, 0.22, 0.70, 0.0],
];
const fociGeom = new THREE.SphereGeometry(1, 16, 12);
const fociBaseColor = new THREE.Color(0xb8ffd0);
const fociGroup = new THREE.Group();
fociGroup.renderOrder = 3; // above organ tube + body wall
scene.add(fociGroup);
for (let i = 0; i < FOCI.length; i++) {
  const bright = FOCI[i][2];
  // Each focus gets its own material so we can pre-multiply brightness into
  // the color (additive blending then naturally ramps with brightness).
  const mat = new THREE.MeshBasicMaterial({
    color: fociBaseColor.clone().multiplyScalar(bright),
    toneMapped: false,
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthTest: false,
  });
  const m = new THREE.Mesh(fociGeom, mat);
  m.renderOrder = 3;
  fociGroup.add(m);
}

// ---------------------------------------------------------------------------
// Worm meshes — the body wall and the inner organ tube share a midline.
// ---------------------------------------------------------------------------
let wormMesh = null;
let organMesh = null;
const WORM_BASE_RADIUS = 22;

function setMidline(points) {
  if (points.length < 2) return;
  const bodyGeom = buildWormGeometry(points, WORM_BASE_RADIUS, bodyProfile, 16);
  const organGeom = buildWormGeometry(points, WORM_BASE_RADIUS, organProfile, 12);
  if (wormMesh) {
    wormMesh.geometry.dispose();
    organMesh.geometry.dispose();
    wormMesh.geometry = bodyGeom;
    organMesh.geometry = organGeom;
  } else {
    organMesh = new THREE.Mesh(organGeom, organMaterial);
    organMesh.renderOrder = 1; // organs first
    scene.add(organMesh);
    wormMesh = new THREE.Mesh(bodyGeom, bodyMaterial);
    wormMesh.renderOrder = 2; // translucent body wall second
    scene.add(wormMesh);
  }

  // Place each focus at its body-length anchor, with an optional lateral
  // offset perpendicular to the local tangent (so granules don't all sit on
  // the midline — gives the natural "scattered" look of confocal granules).
  const last = points.length - 1;
  for (let i = 0; i < FOCI.length; i++) {
    const [s, scale, bright, lateral] = FOCI[i];
    const idx = Math.min(last, Math.max(0, Math.round(s * last)));
    const idxNext = Math.min(last, idx + 1);
    const [px, py] = points[idx];
    const [pxn, pyn] = points[idxNext];
    let tx = pxn - px, ty = pyn - py;
    const tlen = Math.hypot(tx, ty) || 1;
    tx /= tlen; ty /= tlen;
    const nx = -ty, ny = tx;
    const offset = lateral * WORM_BASE_RADIUS * 0.45;
    const mesh = fociGroup.children[i];
    mesh.position.set(px + nx * offset, py + ny * offset, 4);
    mesh.scale.setScalar(WORM_BASE_RADIUS * scale);
  }
}

// ---------------------------------------------------------------------------
// Food — yellow-ish bacterial spots. Modest emission so they bloom slightly.
// ---------------------------------------------------------------------------
const foodMaterial = new THREE.MeshBasicMaterial({
  color: 0xffd040,
  toneMapped: false,
  side: THREE.DoubleSide,
  depthTest: false,
});
const foodGeom = new THREE.SphereGeometry(11, 24, 16);
const foodGroup = new THREE.Group();
foodGroup.renderOrder = 10;
scene.add(foodGroup);

function setFood(items) {
  wordFoodMap.clear();
  while (foodGroup.children.length) foodGroup.remove(foodGroup.children[0]);
  for (const item of items) {
    if (item.word) {
      // Hamlet word food — tracked for text canvas
      wordFoodMap.set(`${item.line_id}_${item.word_idx}`, item);
    } else {
      // Manual click food — yellow 3D sphere
      const m = new THREE.Mesh(foodGeom, foodMaterial);
      m.position.set(item.x || item[0], item.y || item[1], 1);
      m.renderOrder = 10;
      foodGroup.add(m);
    }
  }
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;
let ws = null;
function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => { hud.textContent = 'connected'; };
  ws.onclose = () => { hud.textContent = 'disconnected · retrying…'; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type !== 'state') return;
    setMidline(msg.midline);
    setFood(msg.food);
    smellsData = msg.smells || [];
    updateChemosensoryPanel();
    wormHeadPos = { x: msg.head[0], y: msg.head[1] };
    neuronActivity = msg.neurons || {};
    stimFlags = msg.stim;
    const stims = Object.entries(msg.stim).filter(([, v]) => v).map(([k]) => k).join(',');

    // Collect active chemosensory neurons from smells
    const activeChemoNeurons = {};
    for (const smell of smellsData) {
      for (const [neuron, activation] of Object.entries(smell.neurons)) {
        if (activation > 0) {
          activeChemoNeurons[neuron] = Math.max(activeChemoNeurons[neuron] || 0, activation);
        }
      }
    }
    const chemoStr = Object.keys(activeChemoNeurons).length > 0
      ? Object.entries(activeChemoNeurons)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([n, v]) => `${n}(${(v*100|0)}%)`)
          .join(' ')
      : '-';

    hud.textContent =
      `speed=${msg.speed.toFixed(2)}  ` +
      `motor L=${msg.motor.L.toFixed(1)} R=${msg.motor.R.toFixed(1)}  ` +
      `food=${msg.food.length}  stim=${stims || '-'}  ` +
      `chemo=${chemoStr}`;
  };
}
connect();

function updateChemosensoryPanel() {
  if (!chemosensoryVisible) return;

  // Collect all active neurons across all smells
  const allNeurons = {};
  for (const smell of smellsData) {
    for (const [neuron, activation] of Object.entries(smell.neurons)) {
      if (activation > 0) {
        allNeurons[neuron] = Math.max(allNeurons[neuron] || 0, activation);
      }
    }
  }

  // Sort neurons by activation
  const sortedNeurons = Object.entries(allNeurons).sort((a, b) => b[1] - a[1]);

  // Determine if there's active sensory input
  const hasActive = sortedNeurons.length > 0 || smellsData.length > 0;
  const idleOpacity = hasActive ? 1.0 : 0.35;
  const idleColor = hasActive ? '#8f8' : '#6f6';

  let html = `<div style="font-weight: bold; margin-bottom: 6px; color: ${idleColor}; opacity: ${idleOpacity};">● CHEMOSENSORY STATE</div>`;

  // Show active neurons with descriptions
  if (sortedNeurons.length > 0) {
    html += '<div style="margin-bottom: 8px; border-bottom: 1px solid rgba(100,200,255,0.2); padding-bottom: 6px;">';
    html += '<div style="font-size: 10px; opacity: 0.6; margin-bottom: 4px;">Active Neurons:</div>';
    for (const [neuron, activation] of sortedNeurons) {
      const emotion = neuronEmotionMap[neuron] || 'other';
      const percent = (activation * 100).toFixed(0);
      const hue = neuron.endsWith('L') ? 120 : neuron.endsWith('R') ? 240 : 210;
      html += `<div class="neuron" style="background: hsla(${hue}, 70%, 40%, 0.4); margin: 2px 0;">
        <strong>${neuron}</strong> (${emotion})<br>
        <span style="font-size: 10px; opacity: 0.7;">activation: ${percent}%</span>
      </div>`;
    }
    html += '</div>';
  } else {
    html += `<div style="opacity: 0.4; padding: 6px 0; font-size: 10px;">no neurons firing</div>`;
  }

  // Show detected words and their emotions
  if (smellsData.length > 0) {
    html += '<div style="font-weight: bold; margin: 8px 0 4px 0; color: #8f8;">DETECTED WORDS</div>';
    const sortedSmells = smellsData.sort((a, b) => a.distance - b.distance);
    for (const smell of sortedSmells) {
      if (Object.keys(smell.neurons).length === 0) continue;

      const maxActivation = Math.max(...Object.values(smell.neurons));
      const percent = (maxActivation * 100).toFixed(0);
      const distance = smell.distance.toFixed(0);

      // Show all emotions with values
      const emotionEntries = Object.entries(smell.weighted_emotions)
        .filter(([, v]) => v > 0.01)
        .sort((a, b) => b[1] - a[1]);

      // Find dominant emotion for color
      let dominantEmotion = emotionEntries[0]?.[0] || 'neutral';

      const emotionColors = {
        joy: 60, trust: 120, anticipation: 150, surprise: 180,
        fear: 270, disgust: 30, sadness: 240, anger: 0,
      };
      const hue = emotionColors[dominantEmotion] || 210;

      let emotionStr = emotionEntries
        .slice(0, 3)
        .map(([e, v]) => `${e}(${(v*100|0)}%)`)
        .join(' ');

      html += `<div class="word" style="background: hsla(${hue}, 60%, 30%, 0.3); border-left: 3px solid hsl(${hue}, 80%, 50%); margin: 3px 0;">
        <span style="color: #aff; font-weight: bold;">"${smell.word}"</span><br>
        <span style="font-size: 9px; opacity: 0.7;">emotions: ${emotionStr}</span><br>
        <span style="font-size: 9px; opacity: 0.6;">
          max neuron activation: ${percent}% | distance: ${distance}u
        </span>
      </div>`;
    }
  } else {
    html += `<div style="opacity: 0.4; padding: 6px 0; font-size: 10px;">no words in range</div>`;
  }

  chemosensoryPanel.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Text rendering functions
// ---------------------------------------------------------------------------

function worldToScreen(wx, wy) {
  const nx = (wx - camera.left) / (camera.right - camera.left);
  const ny = (wy - camera.top) / (camera.bottom - camera.top);
  return [nx * textcanvas.width, ny * textcanvas.height];
}

function screenScale() {
  return textcanvas.width / (camera.right - camera.left);
}

async function initEmbeddings() {
  try {
    const data = await (await fetch('/embeddings')).json();
    pcaData = data;
  } catch (e) {
    console.warn('embeddings not available', e);
  }
}
initEmbeddings();

function drawTextCanvas() {
  const w = textcanvas.width, h = textcanvas.height;
  tctx.clearRect(0, 0, w, h);

  // Draw smells first (so they appear behind words)
  drawSmells();

  // Find nearest word to mouse cursor
  let minDist = Infinity;
  nearestWord = null;
  for (const [key, item] of wordFoodMap) {
    const dx = item.x - mouseWorldPos.x, dy = item.y - mouseWorldPos.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d < minDist) {
      minDist = d;
      nearestWord = item;
    }
  }

  // Draw each word
  tctx.font = '18px ui-monospace, monospace';
  for (const [key, item] of wordFoodMap) {
    const [sx, sy] = worldToScreen(item.x, item.y);
    const isHovered = nearestWord === item && minDist < 80;
    tctx.fillStyle = isHovered ? 'rgba(255,255,255,1.0)' : 'rgba(255,255,255,0.7)';
    tctx.textAlign = 'center';
    tctx.textBaseline = 'middle';
    tctx.fillText(item.word, sx, sy);
  }

  // PCA popup for nearest hovered word
  if (nearestWord && minDist < 80 && pcaData) {
    drawPcaPopup(nearestWord.word);
  }
}

function drawPcaPopup(word) {
  const PW = 200, PH = 200, PAD_X = 20, PAD_Y = 20;
  // Position popup at mouse cursor with offset
  let px = mouseScreenPos.x + PAD_X;
  let py = mouseScreenPos.y + PAD_Y;

  // Clamp to viewport
  px = Math.min(px, textcanvas.width - PW - 4);
  py = Math.min(py, textcanvas.height - PH - 4);
  px = Math.max(px, 4);
  py = Math.max(py, 4);

  // Background
  tctx.fillStyle = 'rgba(0,0,0,0.9)';
  tctx.fillRect(px, py, PW, PH);
  tctx.strokeStyle = 'rgba(255,255,255,0.3)';
  tctx.lineWidth = 1;
  tctx.strokeRect(px, py, PW, PH);

  // Crosshairs
  tctx.strokeStyle = 'rgba(255,255,255,0.08)';
  tctx.beginPath();
  tctx.moveTo(px + PW / 2, py);
  tctx.lineTo(px + PW / 2, py + PH);
  tctx.moveTo(px, py + PH / 2);
  tctx.lineTo(px + PW, py + PH / 2);
  tctx.stroke();

  const { tokens, pca, token_to_idx } = pcaData;
  const margin = 16;

  // All tokens as grey dots
  tctx.fillStyle = 'rgba(180,180,180,0.35)';
  for (let i = 0; i < tokens.length; i++) {
    const [cx, cy] = pca[i];
    tctx.beginPath();
    tctx.arc(
      px + margin + cx * (PW - 2 * margin),
      py + margin + cy * (PH - 2 * margin),
      2,
      0,
      Math.PI * 2
    );
    tctx.fill();
  }

  // Highlighted word
  const idx =
    token_to_idx[word] ?? token_to_idx[word.toLowerCase()];
  if (idx !== undefined) {
    const [cx, cy] = pca[idx];
    const hx = px + margin + cx * (PW - 2 * margin);
    const hy = py + margin + cy * (PH - 2 * margin);
    tctx.fillStyle = 'rgba(255,255,255,0.95)';
    tctx.beginPath();
    tctx.arc(hx, hy, 4, 0, Math.PI * 2);
    tctx.fill();
    // Label
    tctx.fillStyle = 'rgba(255,255,255,0.9)';
    tctx.font = '9px ui-monospace, monospace';
    tctx.textAlign = 'left';
    tctx.fillText(word, hx + 6, hy + 3);
  }
}

function drawSmells() {
  if (!smellsVisible || smellsData.length === 0) return;

  // Neuron type colors
  const neuronColors = {
    ASE: { h: 240, s: 100, l: 50 },  // Blue - valence
    AWA: { h: 120, s: 100, l: 50 },  // Green - appetitive
    AWB: { h: 150, s: 100, l: 50 },  // Cyan - approach
    AWC: { h: 180, s: 100, l: 50 },  // Turquoise - CO2
    ASI: { h: 60, s: 100, l: 50 },   // Yellow - intensity
    ASJ: { h: 30, s: 100, l: 50 },   // Orange - feeding
    ASH: { h: 0, s: 100, l: 50 },    // Red - protective
  };

  const [headsx, headsy] = worldToScreen(wormHeadPos.x, wormHeadPos.y);
  const headRadius = 12;  // Worm head radius for offset calculation

  for (const smell of smellsData) {
    if (!smell.neurons || Object.keys(smell.neurons).length === 0) continue;

    const [wordsx, wordsy] = worldToScreen(smell.x, smell.y);

    // Draw separate lines for each active chemosensory neuron
    for (const [neuronName, activation] of Object.entries(smell.neurons)) {
      if (activation === 0) continue;

      // Get neuron type (first 3 chars: ASE, AWA, AWB, etc.)
      const neuronType = neuronName.substring(0, 3);
      const neuronSide = neuronName.endsWith('L') ? 'L' :
                         neuronName.endsWith('R') ? 'R' : 'C';

      const color = neuronColors[neuronType] || { h: 210, s: 100, l: 50 };
      const intensity = Math.min(1.0, activation);
      const lineWidth = 1 + intensity * 3;

      // Offset line origin based on neuron side (bilateral asymmetry)
      let startX = headsx;
      let startY = headsy;

      if (neuronSide === 'L') {
        startX -= headRadius * 0.6;  // Left side neurons start left
      } else if (neuronSide === 'R') {
        startX += headRadius * 0.6;  // Right side neurons start right
      }
      // Center neurons start from head center

      tctx.strokeStyle = `hsla(${color.h}, ${color.s}%, ${color.l}%, ${intensity * 0.8})`;
      tctx.lineWidth = lineWidth;
      tctx.setLineDash([5, 5]);
      tctx.lineCap = 'round';
      tctx.lineJoin = 'round';

      tctx.beginPath();
      tctx.moveTo(startX, startY);
      tctx.lineTo(wordsx, wordsy);
      tctx.stroke();

      tctx.setLineDash([]);
    }

    // Small circle at smell source
    let maxActivation = 0;
    for (const activation of Object.values(smell.neurons)) {
      maxActivation = Math.max(maxActivation, activation);
    }
    if (maxActivation > 0) {
      const radius = 2 + maxActivation * 3;
      tctx.fillStyle = `hsla(210, 100%, 50%, ${maxActivation * 0.6})`;
      tctx.beginPath();
      tctx.arc(wordsx, wordsy, radius, 0, Math.PI * 2);
      tctx.fill();
    }
  }
}

// ---------------------------------------------------------------------------
// Neural network graph visualization
// ---------------------------------------------------------------------------
// Neural network graph visualization (anatomical layout)
// ---------------------------------------------------------------------------
const netcanvas = document.getElementById('netcanvas');
const ctx = netcanvas.getContext('2d');
const NET_W = 500, NET_H = 360;
const LEGEND_H = 72;
const NEURO_TOP = LEGEND_H + 2;
const NEURO_H = NET_H - LEGEND_H;
const PAD = 12;
const dpr = window.devicePixelRatio || 1;
netcanvas.width  = NET_W * dpr;
netcanvas.height = NET_H * dpr;
ctx.scale(dpr, dpr);

// Text canvas setup
const textcanvas = document.getElementById('textcanvas');
const tctx = textcanvas.getContext('2d');
function resizeText() {
  textcanvas.width = window.innerWidth;
  textcanvas.height = window.innerHeight;
}
resizeText();
window.addEventListener('resize', resizeText);

// Track mouse position in screen coordinates
let mouseScreenPos = { x: 0, y: 0 };

let netVisible = true;
let motorLabelsVisible = false;
let graph = null;
let neuronActivity = {};
let stimFlags = { hunger: false, nose_touch: false, food_sense: false };

// Text food and word embedding data
let wordFoodMap = new Map();  // key: `${line_id}_${word_idx}` → {x, y, word}
let pcaData = null;           // fetched from /embeddings
let nearestWord = null;       // the word closest to mouse
let wormHeadPos = { x: 800, y: 500 };  // updated from snapshot
let mouseWorldPos = { x: WORLD_W / 2, y: WORLD_H / 2 };  // mouse position in world coords

// Smell visualization
let smellsData = [];          // list of sensed smells from snapshot
let smellsVisible = true;     // toggle with 'o' key

// Chemosensory panel
let chemosensoryVisible = true;  // toggle with 'c' key
const chemosensoryPanel = document.getElementById('chemosensoryPanel');

// Neuron type to emotion mapping (for display)
const neuronEmotionMap = {
  'ASEL': 'joy/trust (L)',
  'ASER': 'sadness/disgust (R)',
  'AWAL': 'approach (L)',
  'AWAR': 'caution (R)',
  'AWBL': 'attract (L)',
  'AWBR': 'repel (R)',
  'AWC': 'safety',
  'ASI': 'intensity',
  'ASJ': 'food/novelty',
  'ASH': 'protective',
};

// Track mouse movement
document.addEventListener('mousemove', (ev) => {
  const rect = textcanvas.getBoundingClientRect();
  const screenX = ev.clientX - rect.left;
  const screenY = ev.clientY - rect.top;
  mouseScreenPos.x = screenX;
  mouseScreenPos.y = screenY;
  // Transform screen coords to world coords
  const normX = screenX / textcanvas.width;
  const normY = screenY / textcanvas.height;
  mouseWorldPos.x = camera.left + normX * (camera.right - camera.left);
  mouseWorldPos.y = camera.top + normY * (camera.bottom - camera.top);
});

window.addEventListener('keydown', ev => {
  if (ev.key === 'n' || ev.key === 'N') {
    netVisible = !netVisible;
    netcanvas.style.display = netVisible ? 'block' : 'none';
  }
  if (ev.key === 'm' || ev.key === 'M') {
    motorLabelsVisible = !motorLabelsVisible;
  }
  if (ev.key === 'o' || ev.key === 'O') {
    smellsVisible = !smellsVisible;
  }
  if (ev.key === 'c' || ev.key === 'C') {
    chemosensoryVisible = !chemosensoryVisible;
    chemosensoryPanel.style.display = chemosensoryVisible ? 'block' : 'none';
  }
});

// Map OpenWorm anatomical coords to canvas coords.
const AP_MIN = -290, AP_MAX = 420;
const DV_MIN = -90,  DV_MAX = 65;

function buildPositions(neurons, rawPositions) {
  const N = neurons.length;
  const pos = new Float32Array(N * 2);
  const apRange = AP_MAX - AP_MIN;
  const dvRange = DV_MAX - DV_MIN;
  const neuroW = NET_W - PAD * 2;
  let fallbackX = PAD + neuroW * 0.6;

  for (let i = 0; i < N; i++) {
    const xyz = rawPositions[i];
    let cx, cy;
    if (xyz) {
      cx = PAD + (xyz[1] - AP_MIN) / apRange * neuroW;
      cy = NEURO_TOP + PAD + (1 - (xyz[2] - DV_MIN) / dvRange) * (NEURO_H - PAD * 2);
    } else {
      cx = fallbackX;
      cy = NEURO_TOP + NEURO_H / 2 + (Math.random() - 0.5) * 40;
      fallbackX = PAD + neuroW * 0.6 + ((fallbackX + 3 - PAD) % (neuroW * 0.4));
    }
    pos[i * 2]     = cx;
    pos[i * 2 + 1] = cy;
  }
  return pos;
}

async function initGraph() {
  const data = await (await fetch('/graph')).json();
  const N = data.neurons.length;
  const pos = buildPositions(data.neurons, data.positions);
  const adjOut = Array.from({length: N}, () => []);
  for (const [pi, qi] of data.edges) adjOut[pi].push(qi);

  graph = {
    neurons:          data.neurons,
    fireThreshold:    data.fire_threshold,
    muscleSet:        new Set(data.muscle_indices),
    sensorySet:       new Set(data.sensory_indices),
    chemosensorySet:  new Set(data.chemosensory_indices),
    motorSet:         new Set(data.motor_indices),
    foodSet:          new Set(data.food_indices),
    noseSet:          new Set(data.nose_indices),
    hungerSet:        new Set(data.hunger_indices),
    pos, adjOut, N,
  };
}
initGraph();

function drawNetCanvas() {
  if (!graph || !netVisible) return;
  const { neurons, fireThreshold, muscleSet, sensorySet, chemosensorySet,
          motorSet, foodSet, noseSet, hungerSet, pos, adjOut, N } = graph;

  ctx.clearRect(0, 0, NET_W, NET_H);

  // ── Legend ──────────────────────────────────────────────────────────────
  const LX = 8, LY = 14, SPC = 82;
  ctx.font = '8px ui-monospace,monospace';
  const legendItems = [
    { color: 'rgba(40,200,255,0.85)',  label: 'chemosensory' },
    { color: 'rgba(100,160,255,0.7)',  label: 'sensory' },
    { color: 'rgba(255,150,40,0.85)',  label: 'motor' },
    { color: 'rgba(68,255,119,0.85)',  label: 'interneuron' },
    { color: 'rgba(255,220,40,0.85)',  label: 'muscle' },
    { color: '#44ff77',                label: 'firing' },
  ];
  legendItems.forEach(({color, label}, i) => {
    const x = LX + i * SPC;
    ctx.beginPath();
    ctx.arc(x + 5, LY, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.fillStyle = 'rgba(68,255,119,0.6)';
    ctx.fillText(label, x + 12, LY + 3);
  });

  ctx.fillStyle = 'rgba(68,255,119,0.35)';
  ctx.fillText('[n] toggle panel  [m] motor labels' + (motorLabelsVisible ? '  ✓' : ''), LX, LY + 18);

  ctx.fillStyle = 'rgba(68,255,119,0.2)';
  ctx.fillText('← head', PAD, LEGEND_H - 2);
  ctx.fillText('tail →', NET_W - 40, LEGEND_H - 2);
  ctx.fillText('dorsal', PAD, NEURO_TOP + 10);
  ctx.fillText('ventral', PAD, NET_H - 4);

  // ── Build activity array ────────────────────────────────────────────────
  const activity = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const v = neuronActivity[neurons[i]];
    if (v) activity[i] = v;
  }

  // ── Edges: from firing neurons only ─────────────────────────────────────
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(68,255,119,0.15)';
  ctx.lineWidth = 0.4;
  for (let pi = 0; pi < N; pi++) {
    if (activity[pi] <= fireThreshold) continue;
    const px = pos[pi*2], py = pos[pi*2+1];
    for (const qi of adjOut[pi]) {
      ctx.moveTo(px, py);
      ctx.lineTo(pos[qi*2], pos[qi*2+1]);
    }
  }
  ctx.stroke();

  // ── Nodes ───────────────────────────────────────────────────────────────
  const hungerOn = stimFlags.hunger;
  const noseOn   = stimFlags.nose_touch;
  const foodOn   = stimFlags.food_sense;
  const pendingLabels = [];

  for (let i = 0; i < N; i++) {
    const x = pos[i*2], y = pos[i*2+1];
    const v = activity[i];
    const firing  = v > fireThreshold;
    const charged = v > 0;
    const t = firing ? 1 : (charged ? Math.min(v / fireThreshold, 1) : 0);

    const stimulated =
      (foodSet.has(i)   && foodOn) ||
      (noseSet.has(i)   && noseOn) ||
      (hungerSet.has(i) && hungerOn);

    const isChemo   = chemosensorySet.has(i);
    const isSensory = sensorySet.has(i);
    const isMotor   = motorSet.has(i);
    const isMuscle  = muscleSet.has(i);

    let baseColor, fireColor, r, haloColor, hr;
    if (isChemo) {
      baseColor  = `rgba(40,200,255,${(0.25 + t * 0.7).toFixed(2)})`;
      fireColor  = 'rgba(40,255,255,0.95)';
      haloColor  = 'rgba(40,220,255,0.22)';
      r = 1.8 + t;  hr = 7;
    } else if (isSensory) {
      baseColor  = `rgba(100,160,255,${(0.2 + t * 0.75).toFixed(2)})`;
      fireColor  = 'rgba(120,180,255,0.95)';
      haloColor  = 'rgba(100,160,255,0.2)';
      r = 1.5 + t;  hr = 6;
    } else if (isMotor) {
      baseColor  = `rgba(255,150,40,${(0.2 + t * 0.75).toFixed(2)})`;
      fireColor  = 'rgba(255,180,40,0.95)';
      haloColor  = 'rgba(255,150,40,0.2)';
      r = 1.5 + t;  hr = 6;
    } else if (isMuscle) {
      baseColor  = `rgba(255,220,40,${(0.12 + t * 0.7).toFixed(2)})`;
      fireColor  = 'rgba(255,240,80,0.9)';
      haloColor  = 'rgba(255,200,40,0.18)';
      r = 1.2 + t * 0.8;  hr = 5;
    } else {
      baseColor  = `rgba(68,180,80,${(0.15 + t * 0.8).toFixed(2)})`;
      fireColor  = '#44ff77';
      haloColor  = 'rgba(68,255,119,0.18)';
      r = 1.3 + t * 0.9;  hr = 6;
    }

    const useCyan = stimulated && !firing;
    const color = useCyan ? 'rgba(40,255,255,0.9)' : (firing ? fireColor : baseColor);
    const halo  = (firing || stimulated || t > 0.5) ? haloColor : null;
    const haloR = (firing || stimulated) ? hr : hr * t;

    if (halo && haloR > 1) {
      ctx.beginPath();
      ctx.arc(x, y, haloR, 0, Math.PI * 2);
      ctx.fillStyle = firing ? haloColor : (useCyan ? 'rgba(40,255,255,0.15)' : haloColor);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(x, y, Math.max(0.8, r), 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    const shouldLabel =
      firing || stimulated ||
      (isMotor && motorLabelsVisible);
    if (shouldLabel && (isSensory || isChemo || isMotor)) {
      pendingLabels.push({ x, y, name: neurons[i], firing, isChemo, isSensory, isMotor });
    }
  }

  // ── Labels (second pass) ────────────────────────────────────────────────
  ctx.font = '6.5px ui-monospace,monospace';
  for (const {x, y, name, firing, isChemo, isSensory, isMotor} of pendingLabels) {
    let lc;
    if (isChemo)   lc = firing ? 'rgba(40,255,255,0.95)' : 'rgba(40,200,255,0.65)';
    else if (isSensory) lc = firing ? 'rgba(140,200,255,0.95)' : 'rgba(100,160,255,0.55)';
    else           lc = firing ? 'rgba(255,200,80,0.95)' : 'rgba(255,150,40,0.5)';
    ctx.fillStyle = lc;
    ctx.fillText(name, x + 3, y - 3);
  }
}

// ---------------------------------------------------------------------------
// Click → world coords → drop food
// ---------------------------------------------------------------------------
function eventToWorld(ev) {
  const r = canvas.getBoundingClientRect();
  const nx = (ev.clientX - r.left) / r.width;
  const ny = (ev.clientY - r.top) / r.height;
  const x = camera.left + nx * (camera.right - camera.left);
  const y = camera.top + ny * (camera.bottom - camera.top);
  return [x, y];
}
canvas.addEventListener('mousedown', (ev) => {
  if (!ws || ws.readyState !== 1) return;
  if (ev.shiftKey) {
    ws.send(JSON.stringify({ type: 'clear_food' }));
  } else {
    const [x, y] = eventToWorld(ev);
    ws.send(JSON.stringify({ type: 'add_food', x, y }));
  }
});

// ---------------------------------------------------------------------------
// Debug handles + render loop
// ---------------------------------------------------------------------------
window.__sim = {
  THREE, scene, camera, renderer, composer, bloom,
  foodGroup, bodyMaterial,
  get wormMesh() { return wormMesh; },
  get graph() { return graph; },
  get neuronActivity() { return neuronActivity; },
  get motorLabelsVisible() { return motorLabelsVisible; },
};

const clock = new THREE.Clock();
function render() {
  requestAnimationFrame(render);
  const t = clock.getElapsedTime();
  bodyMaterial.uniforms.uTime.value = t;
  organMaterial.uniforms.uTime.value = t;
  composer.render();
  drawNetCanvas();
  drawTextCanvas();
}
render();
