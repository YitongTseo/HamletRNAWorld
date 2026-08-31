// Worm-body rendering: shaders, geometry, organ tube, intestinal foci,
// and per-tick midline updates. Extracted from focus/index.js — behavior
// unchanged.
import * as THREE from 'three';
import { scene } from './three-scene.js';

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
    uColor: { value: new THREE.Color(0xe8dcc0) },
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
    uOrganColor: { value: new THREE.Color(0xc9b795) },
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
const fociBaseColor = new THREE.Color(0xf2e8cf);
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

// Latest midline polyline in world coords. Used by the x-ray panel to
// place neurons on the live, twisting body.
let latestMidline = [];

function setMidline(points) {
  if (points.length < 2) return;
  latestMidline = points;
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

// Accessor for the latest midline polyline. The x-ray panel reads this
// every frame to overlay neurons on the live body.
function getMidline() { return latestMidline; }

// Accessor for the body-wall mesh (for debug handles in window.__sim).
function getWormMesh() { return wormMesh; }

export {
  bodyMaterial,
  organMaterial,
  setMidline,
  getMidline,
  getWormMesh,
};
