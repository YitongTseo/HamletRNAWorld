// Three.js infrastructure: renderer, scene, camera, composer, resize.
// Extracted from focus/index.js — behavior unchanged.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { cameraFrustumScale, onViewportChange } from './responsive.js';

// ---------------------------------------------------------------------------
// Scene / camera / renderer
// ---------------------------------------------------------------------------
const canvas = document.getElementById('c');

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
  updateCameraFrustum();
}

// Set the orthographic frustum based on current viewport aspect ratio and
// the mobile zoom factor from responsive.js. On mobile (s = 0.55) the half
// extents shrink, which makes the worm and surrounding Hamlet text appear
// ~1.8x larger so they read on a narrow viewport.
function updateCameraFrustum() {
  const w = window.innerWidth, h = window.innerHeight;
  const worldAspect = WORLD_W / WORLD_H;
  const viewAspect = w / h;
  let vw = WORLD_W, vh = WORLD_H;
  if (viewAspect > worldAspect) vw = WORLD_H * viewAspect;
  else vh = WORLD_W / viewAspect;
  const s = cameraFrustumScale();
  const cx = WORLD_W / 2, cy = WORLD_H / 2;
  camera.left = cx - (vw * s) / 2;
  camera.right = cx + (vw * s) / 2;
  camera.top = cy - (vh * s) / 2;   // y-down (see camera comment above)
  camera.bottom = cy + (vh * s) / 2;
  camera.updateProjectionMatrix();
}

resize();
window.addEventListener('resize', resize);
// responsive.js fires a debounced (100ms) resize event; the immediate
// window 'resize' listener above handles the sync renderer.setSize + frustum
// update, this listener handles the case where cameraFrustumScale() toggles
// across the MOBILE_BREAKPOINT during a debounced final event. Both calls
// are idempotent, so the redundant firing is harmless.
onViewportChange(updateCameraFrustum);

export { canvas, renderer, scene, camera, composer, bloom, resize, WORLD_W, WORLD_H };
