// The site header — markup AND styling, in one place.
//
// It used to be copy-pasted into every page's <head>, and the copies drifted:
// the overview styled itself from private var names (--bench/--ivory/--ochre)
// that palette.js never sets, so it alone ignored the theme; poems, about and
// generations used --accent; and the focus view had no header at all, just a
// hand-rolled #nav with hardcoded colours, 12px non-uppercase type and a
// cyan-tinted hover left over from the pre-vivarium palette. Entering a
// specimen page visibly changed the chrome. One file now, so a sixth page
// cannot drift a sixth way.
//
// Load it as the FIRST element inside <body> (synchronously — it must exist
// before experiment_dropdown.js looks for #experiment-switcher):
//   <body data-page="worms">
//     <script src="/static/header.js?v=1"></script>
// `data-page` marks the current nav item; data-header="overlay" renders the
// bar transparent and fixed, for the full-bleed canvas pages.
(function () {
  "use strict";

  var PAGES = [
    { key: "worms",       href: "/",            label: "worms" },
    { key: "poems",       href: "/poems",       label: "poems" },
    { key: "generations", href: "/generations", label: "generations" },
    { key: "about",       href: "/about",       label: "about" },
  ];

  // The graveyard is deliberately NOT in the nav row: you find it by noticing
  // the stone in the corner. A mono glyph, currentColor so it inherits the
  // nav's dim-to-ochre hover like any other link.
  var TOMBSTONE =
    '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" focusable="false">' +
      '<path d="M4 15V6a4 4 0 0 1 8 0v9" fill="none" stroke="currentColor" stroke-width="1.3"/>' +
      '<path d="M6.4 8.2h3.2M6.4 10.4h3.2" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>' +
      '<path d="M2.2 15h11.6" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>' +
    '</svg>';

  var CSS = [
    '#site-header{display:flex;align-items:baseline;justify-content:space-between;',
      'gap:18px;padding:22px 34px 16px;border-bottom:1px solid var(--line);}',
    '#site-header h1{margin:0;font-family:var(--font-serif,"Instrument Serif",serif);',
      'font-style:italic;font-weight:400;font-size:26px;letter-spacing:0.01em;color:var(--fg);}',
    '#site-header h1 a{color:inherit;text-decoration:none;}',
    '#site-header h1 b{font-weight:400;color:var(--accent);}',
    '#site-header h1 small{font-weight:300;font-size:10px;letter-spacing:0.08em;',
      'color:var(--dim);margin-left:14px;text-transform:none;font-style:normal;}',
    '#site-header h1 small a{color:var(--dim);}',
    '#site-header .header-right{display:flex;align-items:center;gap:10px;}',
    '#site-header nav{font-size:11.5px;letter-spacing:0.12em;text-transform:uppercase;',
      'font-family:var(--font-mono,ui-monospace,monospace);}',
    '#site-header nav a{color:var(--dim);text-decoration:none;padding:4px 8px;}',
    '#site-header nav a:hover{color:var(--accent);}',
    '#site-header nav a.here{color:var(--accent);}',
    // The stone sits slightly proud of the text baseline so it reads as an
    // object in the corner rather than another word in the row.
    '#site-header .graveyard-link{display:inline-flex;align-items:center;color:var(--dim);',
      'padding:2px 6px;line-height:0;transition:color 160ms ease,transform 160ms ease;}',
    '#site-header .graveyard-link:hover{color:var(--accent);transform:translateY(-1px);}',
    '#site-header .graveyard-link.here{color:var(--accent);}',
    // Overlay variant (focus): same type, no bar. A scrim keeps the links
    // legible over a pale patch of agar without boxing the view in.
    'body[data-header="overlay"] #site-header{position:fixed;top:0;left:0;right:0;z-index:30;',
      'border-bottom:none;padding:10px 16px 22px;pointer-events:none;align-items:flex-start;',
      // Tobacco literal first as the floor for old browsers, then the themed
      // version: the scrim has to fade to transparent, and a CSS variable
      // can't carry an alpha ramp on its own.
      'background:linear-gradient(to bottom,rgba(41,33,24,0.82),rgba(41,33,24,0));',
      'background:linear-gradient(to bottom,color-mix(in srgb,var(--bg) 82%,transparent),transparent);}',
    'body[data-header="overlay"] #site-header h1{font-size:17px;pointer-events:auto;}',
    'body[data-header="overlay"] #site-header h1 small{display:none;}',
    'body[data-header="overlay"] #site-header .header-right{pointer-events:auto;}',
    '@media (max-width:768px){',
      '#site-header{padding:14px 16px 10px;gap:10px;flex-wrap:wrap;}',
      '#site-header h1{font-size:20px;}',
      '#site-header h1 small{display:none;}',
      '#site-header nav{font-size:10.5px;}',
      '#site-header nav a{padding:3px 5px;}',
    '}',
  ].join("");

  function build() {
    var page = document.body.getAttribute("data-page") || "";
    var nav = PAGES.map(function (p) {
      var here = p.key === page ? ' class="here"' : "";
      return '<a href="' + p.href + '"' + here + ">" + p.label + "</a>";
    }).join("");
    var stoneHere = page === "graveyard" ? " here" : "";
    return (
      '<header id="site-header">' +
        '<h1><a href="/">wormlet <b>vivarium</b></a>' +
          '<small>an experiment by <a href="https://www.ternalbiota.com" target="_blank" ' +
          'rel="noopener">Yitong</a></small></h1>' +
        '<div class="header-right">' +
          "<nav>" + nav + "</nav>" +
          '<a class="graveyard-link' + stoneHere + '" href="/graveyard" ' +
            'title="graveyard" aria-label="graveyard">' + TOMBSTONE + "</a>" +
          '<div id="experiment-switcher"></div>' +
        "</div>" +
      "</header>"
    );
  }

  var style = document.createElement("style");
  style.id = "wormlet-header-css";
  style.textContent = CSS;
  document.head.appendChild(style);

  // Replace a placeholder if the page left one, else prepend. Either way the
  // header exists before the rest of <body> parses, so experiment_dropdown.js
  // finds #experiment-switcher where it expects it.
  var existing = document.getElementById("site-header");
  if (existing) existing.outerHTML = build();
  else document.body.insertAdjacentHTML("afterbegin", build());
})();
