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

  // Type. header.js (shared with viewer_vivarium/) styles its h1 from
  // var(--font-serif,...) and its nav from var(--font-mono,...) so one file
  // can serve both trees' typography. Vivarium sets these to Instrument
  // Serif / Fragment Mono and pulls the matching Google Fonts stylesheet;
  // classic has always been one plain monospace face, so both variables
  // just point at the same local stack here — no network fetch, and the
  // header renders in classic's own type instead of falling back to
  // header.js's built-in (serif) default.
  root.style.setProperty("--font-serif", "ui-monospace, SFMono-Regular, Menlo, monospace");
  root.style.setProperty("--font-mono", "ui-monospace, SFMono-Regular, Menlo, monospace");

  // Where THIS script was loaded from. server/ui_variant.py's page() rewrites
  // the bare static path in the HTML to a per-variant one as it serves the
  // page, but it can't touch a path baked into a .js file — a hardcoded
  // prefix here would keep fetching the DEFAULT variant's favicon even when
  // this copy is running under ?ui=vivarium (tests/test_ui_variant.py::
  // test_no_static_refs_outside_html fails the build on exactly that kind of
  // literal outside HTML). Derive it instead from our own <script> tag's
  // resolved src.
  function selfPrefix() {
    var el = document.currentScript;
    if (!el) {
      // Only null for type="module" scripts per spec (not used here) or a
      // script inserted after parsing — fall back to finding our own tag by
      // filename; it is always in the document by the time we run.
      var scripts = document.getElementsByTagName("script");
      for (var i = scripts.length - 1; i >= 0; i--) {
        if (/palette\.js/.test(scripts[i].src)) { el = scripts[i]; break; }
      }
    }
    var src = el.src;
    return src.slice(0, src.lastIndexOf("/") + 1);
  }

  // Favicon. Injected here for the same reason as the theme itself: this is
  // the one file every page loads in <head>, so the tab mark arrives
  // without editing six HTML files (and the next page can't forget it).
  // Browsers that ignore SVG icons fall back to /favicon.ico, which the
  // server answers with the same file.
  var icon = document.createElement("link");
  icon.rel = "icon";
  icon.type = "image/svg+xml";
  icon.href = selfPrefix() + "favicon.svg?v=1";
  document.head.appendChild(icon);

  // Expose for other scripts (e.g. the worm canvas reads --accent/--stage).
  window.__paletteMode = mode;
  window.__palettes = PALETTES;
})();
