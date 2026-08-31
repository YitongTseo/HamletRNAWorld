// The graveyard: one plot per recorded death, each with the judge's
// highest-rated passage from the poem that worm was writing when it starved.
//
// The animation is the point of the page, so it is worth saying what it does
// and why it is that slow. Each stone carries a ghost: a single undulating
// line, the same tapered sine the live worms are drawn with, that swims up out
// of the stone and fades at the top of the frame. One pass takes ~11 seconds
// and the phase is seeded from the worm's name, so no two stones move
// together and the page never reads as a row of blinking widgets. It honours
// them by looking like the animal, not like a spinner.
//
// One rAF loop drives every canvas, and only the ones on screen are drawn
// (IntersectionObserver) — a lineage can accumulate hundreds of graves.
(function () {
  "use strict";

  var CYCLE_MS = 11000;          // one ghost pass, deliberately unhurried
  var FPS = 30;                  // plenty for a slow line; halves the work

  function css(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v || "").trim() || fallback;
  }

  // Deterministic per-name phase so each grave has its own rhythm.
  function seedOf(str) {
    var h = 2166136261;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = (h * 16777619) >>> 0;
    }
    return h / 4294967295;
  }

  var CJK = /[⺀-鿿豈-﫿　-〿]/;
  var NO_SPACE_BEFORE = /^[.,;:!?)\]}’”。，、：；！？-]$/;

  function joinTokens(tokens) {
    if (!tokens || !tokens.length) return "";
    // Classical Chinese is unsegmented: one character per token, set solid.
    if (tokens.every(function (t) { return CJK.test(t); })) return tokens.join("");
    var out = "";
    tokens.forEach(function (t, i) {
      if (i && !NO_SPACE_BEFORE.test(t)) out += " ";
      out += t;
    });
    return out;
  }

  function ago(iso) {
    if (!iso) return "";
    var then = Date.parse(iso);
    if (isNaN(then)) return "";
    var mins = Math.max(0, (Date.now() - then) / 60000);
    if (mins < 90) return Math.round(mins) + " min ago";
    var hrs = mins / 60;
    if (hrs < 36) return Math.round(hrs) + " h ago";
    return Math.round(hrs / 24) + " days ago";
  }

  // --- the stone ------------------------------------------------------------

  function drawStone(ctx, w, h, t, palette) {
    ctx.clearRect(0, 0, w, h);
    var stoneW = w * 0.62, stoneH = h * 0.60;
    var x = (w - stoneW) / 2, y = h - stoneH;

    // Ground line first, so the stone sits on it.
    ctx.strokeStyle = palette.dim;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w * 0.06, h - 1.5);
    ctx.lineTo(w * 0.94, h - 1.5);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // The ghost goes down FIRST so the stone occludes it: it rises from
    // behind the headstone, clears the shoulders, and fades out above. Drawn
    // over the stone it just looked like a scribble across the slab.
    var u = t;                                   // 0..1 through the cycle
    var rise = h * 0.86 * u;
    var fade = Math.sin(Math.PI * Math.min(1, Math.max(0, u))) ;
    if (fade > 0.01) {
      ctx.save();
      ctx.globalAlpha = 0.75 * fade;
      ctx.strokeStyle = palette.accent;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      var segs = 26, len = h * 0.30, baseY = h - stoneH * 0.55;
      ctx.beginPath();
      for (var i = 0; i <= segs; i++) {
        var s = i / segs;
        var yy = baseY - rise - s * len;
        var xx = w / 2 + Math.sin(s * Math.PI * 2.1 + u * 5.2) * (w * 0.16) * (1 - s * 0.35);
        if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      }
      // One even stroke: at 74 px wide a taper is a pixel of difference, and
      // the fade already carries the sense of something leaving.
      ctx.lineWidth = 2.1;
      ctx.stroke();
      ctx.restore();
    }

    // The stone: a slab with a rounded head, same silhouette as the tombstone
    // glyph in the header so the icon and the page are visibly the same
    // object. Filled a shade LIGHTER than the ground — --panel is darker than
    // --bg in this palette, which made the stone read as a doorway rather
    // than something standing in the light.
    var r = stoneW / 2;
    ctx.beginPath();
    ctx.moveTo(x, h - 2);
    ctx.lineTo(x, y + r);
    ctx.arc(x + r, y + r, r, Math.PI, 0);
    ctx.lineTo(x + stoneW, h - 2);
    ctx.closePath();
    ctx.fillStyle = palette.bg;
    ctx.fill();                      // opaque, so it occludes the ghost
    ctx.globalAlpha = 0.14;
    ctx.fillStyle = palette.dim;
    ctx.fill();
    ctx.globalAlpha = 0.55;
    ctx.strokeStyle = palette.dim;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // --- rendering ------------------------------------------------------------

  var canvases = [];

  function plotEl(g) {
    var el = document.createElement("div");
    el.className = "plot";

    var cv = document.createElement("canvas");
    cv.className = "stone";
    cv.setAttribute("aria-hidden", "true");
    el.appendChild(cv);

    var body = document.createElement("div");

    var name = document.createElement("div");
    name.className = "name";
    name.textContent = g.worm;
    body.appendChild(name);

    var vitals = document.createElement("div");
    vitals.className = "vitals";
    var bits = [
      g.flask_label || g.flask,
      g.corpus,
      "generation " + g.generation,
      g.words_eaten + " words eaten",
    ];
    if (g.starved_s) bits.push(Math.round(g.starved_s / 60) + " min without food");
    var when = ago(g.died_at);
    if (when) bits.push(when);
    vitals.innerHTML = bits.map(function (b) {
      return '<span>' + b + "</span>";
    }).join('<span class="sep">·</span>');
    body.appendChild(vitals);

    if (g.epitaph && g.epitaph.tokens && g.epitaph.tokens.length) {
      var ep = document.createElement("blockquote");
      ep.className = "epitaph";
      ep.textContent = joinTokens(g.epitaph.tokens);
      var src = document.createElement("span");
      src.className = "src";
      // Say honestly where the lines came from. Most fresh graves are worms
      // that died in the generation now running, which the judge only scores
      // at rollover — their epitaph is either an older poem of theirs or the
      // unjudged tail of the one they were writing, and the page must not
      // imply the critic praised lines it never saw.
      var e = g.epitaph, rated = " · emotion " + e.emotional + " · coherence " + e.coherence;
      if (e.source === "judged") {
        src.textContent = "its best-judged lines" + rated;
      } else if (e.source === "judged-earlier") {
        src.textContent = "its best-judged lines, generation " + e.generation + rated;
      } else {
        src.textContent = "the last lines it wrote · unjudged";
      }
      ep.appendChild(src);
      body.appendChild(ep);
    }

    if (g.last_words && g.last_words.length) {
      var lw = document.createElement("div");
      lw.className = "last-words";
      lw.innerHTML = "last words &nbsp;<b>" + joinTokens(g.last_words) + "</b>";
      body.appendChild(lw);
    }

    el.appendChild(body);
    canvases.push({ cv: cv, phase: seedOf(g.worm + g.generation), visible: false });
    return el;
  }

  function sizeCanvas(entry) {
    var cv = entry.cv;
    var dpr = window.devicePixelRatio || 1;
    var r = cv.getBoundingClientRect();
    if (!r.width) return false;
    cv.width = Math.round(r.width * dpr);
    cv.height = Math.round(r.height * dpr);
    entry.ctx = cv.getContext("2d");
    entry.ctx.scale(dpr, dpr);
    entry.w = r.width;
    entry.h = r.height;
    return true;
  }

  function startLoop() {
    var palette = {
      dim: css("--dim", "#8f8266"),
      accent: css("--accent", "#cfa348"),
      bg: css("--bg", "#292118"),
    };
    var still = window.matchMedia &&
                window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    var io = window.IntersectionObserver ? new IntersectionObserver(function (rows) {
      rows.forEach(function (row) {
        canvases.forEach(function (e) {
          if (e.cv === row.target) e.visible = row.isIntersecting;
        });
      });
    }, { rootMargin: "120px" }) : null;
    canvases.forEach(function (e) {
      if (io) io.observe(e.cv); else e.visible = true;
    });

    var last = 0;
    function frame(now) {
      if (now - last >= 1000 / FPS) {
        last = now;
        for (var i = 0; i < canvases.length; i++) {
          var e = canvases[i];
          if (!e.visible) continue;
          if (!e.ctx && !sizeCanvas(e)) continue;
          // Reduced motion: hold the ghost at mid-rise, no animation.
          var t = still ? 0.5 : ((now / CYCLE_MS) + e.phase) % 1;
          drawStone(e.ctx, e.w, e.h, t, palette);
        }
      }
      if (!still) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    window.addEventListener("resize", function () {
      canvases.forEach(function (e) { e.ctx = null; });
    });
  }

  fetch("/api/graveyard").then(function (r) { return r.json(); }).then(function (data) {
    var plots = document.getElementById("plots");
    var count = document.getElementById("count");
    var graves = (data && data.graves) || [];
    if (!graves.length) {
      count.textContent = "";
      plots.innerHTML = '<div id="empty">No deaths recorded. Every worm in ' +
                        "every flask is still eating.</div>";
      return;
    }
    count.textContent = graves.length === 1 ? "1 worm remembered"
                                            : graves.length + " worms remembered";
    var frag = document.createDocumentFragment();
    graves.forEach(function (g) { frag.appendChild(plotEl(g)); });
    plots.appendChild(frag);
    startLoop();
  }).catch(function (e) {
    document.getElementById("count").textContent = "graveyard unavailable";
    console.error(e);
  });
})();
