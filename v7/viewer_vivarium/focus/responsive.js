// viewer/focus/responsive.js
// Viewport helpers for the focus page: detects mobile width,
// exposes a camera frustum scale factor, and broadcasts debounced
// resize events to subscribed listeners.
export const MOBILE_BREAKPOINT = 768;
export function isMobile() { return window.innerWidth < MOBILE_BREAKPOINT; }
export function cameraFrustumScale() { return isMobile() ? 0.55 : 1.0; }

const listeners = new Set();
export function onViewportChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }

let pending = null;
window.addEventListener('resize', () => {
  if (pending) clearTimeout(pending);
  pending = setTimeout(() => {
    pending = null;
    for (const fn of listeners) fn();
  }, 100);
});
