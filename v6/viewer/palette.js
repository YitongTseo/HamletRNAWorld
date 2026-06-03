// Per-experiment color palette. Loaded as a RENDER-BLOCKING <script> in the
// <head> of every viewer page (before the body paints) so there's no
// flash-of-default-theme. Each experiment runs on its own subdomain/port, so
// we can pick the palette synchronously from the hostname — no network round
// trip, no dependency on /api/experiment.
//
// The whole site reshades when you switch experiments via the top-right
// dropdown (which just navigates to the chosen experiment's subdomain). The
// simulation "stage" (worm canvas background) is kept dark in every theme so
// the light-on-dark worms stay legible even under the white themes.
(function () {
  "use strict";

  // mode -> palette. `stage` is the (always-dark) worm-canvas background;
  // `accent` is bright/saturated in every theme so worms read on the stage.
  var PALETTES = {
    words:     { label: "black / green", bg: "#000000", fg: "#aaffaa", accent: "#66ff99", dim: "#55aa55", warm: "#ffcc66", hot: "#ff6666", panel: "#001108", line: "rgba(102,255,153,0.18)", stage: "#001108" },
    nouns:     { label: "black / purple", bg: "#050011", fg: "#e7d6ff", accent: "#b388ff", dim: "#8a6cb0", warm: "#ffcf6b", hot: "#ff6b9d", panel: "#0c0420", line: "rgba(179,136,255,0.20)", stage: "#0c0420" },
    adj_noun:  { label: "white / pink", bg: "#fff6fa", fg: "#3a1020", accent: "#e5447f", dim: "#b06b84", warm: "#e08a00", hot: "#dd1133", panel: "#ffe6ef", line: "rgba(229,68,127,0.22)", stage: "#1a0512" },
    pos_chain: { label: "black / gold", bg: "#0a0700", fg: "#f3e4b0", accent: "#ffcc33", dim: "#a8893f", warm: "#ffd97a", hot: "#ff7a45", panel: "#140d00", line: "rgba(255,204,51,0.20)", stage: "#140d00" },
    poetry:    { label: "black / green", bg: "#000000", fg: "#c6f6d5", accent: "#3ddc84", dim: "#5a8f6a", warm: "#ffcc66", hot: "#ff6b6b", panel: "#03140a", line: "rgba(61,220,132,0.18)", stage: "#03140a" }
  };

  function detectMode() {
    var host = (location.hostname || "").toLowerCase();
    if (host.indexOf("words.") === 0) return "words";
    if (host.indexOf("nouns.") === 0) return "nouns";
    if (host.indexOf("adj-noun.") === 0 || host.indexOf("adj_noun.") === 0) return "adj_noun";
    if (host.indexOf("pos-chain.") === 0 || host.indexOf("pos_chain.") === 0) return "pos_chain";
    // Local dev (127.0.0.1:800x) — map by the per-experiment port.
    var byPort = { "8001": "words", "8002": "nouns", "8003": "adj_noun", "8004": "pos_chain" };
    if (byPort[location.port]) return byPort[location.port];
    // www / apex / prod box.
    return "poetry";
  }

  var mode = detectMode();
  var p = PALETTES[mode] || PALETTES.poetry;
  var root = document.documentElement;
  for (var key in p) {
    if (key === "label") continue;
    root.style.setProperty("--" + key, p[key]);
  }
  root.setAttribute("data-mode", mode);

  // Expose for other scripts (e.g. the worm canvas reads --accent/--stage).
  window.__paletteMode = mode;
  window.__palettes = PALETTES;
})();
