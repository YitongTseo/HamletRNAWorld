// Shared experiment switcher — included on every viewer page.
//
// Renders a <select> dropdown into any element matching #experiment-switcher
// (or as a fallback, appends one to the page <header>). The dropdown lists
// all 5 experiments and pre-selects the current one. Changing selection
// navigates to the chosen experiment's public_url.
//
// Listens for window.__experimentLoaded callbacks so other page scripts
// can react when the experiment metadata arrives (e.g. the generations
// page wants to title its callout banner).

(async function () {
  const TARGET_SELECTOR = "#experiment-switcher";

  function injectStyles() {
    if (document.getElementById("experiment-switcher-styles")) return;
    const style = document.createElement("style");
    style.id = "experiment-switcher-styles";
    style.textContent = `
      .experiment-switcher { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; color: var(--dim, #5a5); }
      .experiment-switcher label { color: var(--dim, #5a5); text-transform: uppercase; letter-spacing: 0.5px; font-size: 10px; }
      .experiment-switcher select {
        background: #001; color: var(--accent, #6f9);
        border: 1px solid rgba(100, 200, 255, 0.3);
        padding: 3px 6px; font: inherit; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        cursor: pointer;
      }
      .experiment-switcher select:hover { border-color: rgba(100, 200, 255, 0.6); }
      .experiment-switcher .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent, #6f9); box-shadow: 0 0 5px var(--accent, #6f9); }
    `;
    document.head.appendChild(style);
  }

  function findTarget() {
    let t = document.querySelector(TARGET_SELECTOR);
    if (t) return t;
    // Fallback: append to the page <header> so the switcher always shows up
    // even on pages that didn't add the explicit hook.
    const hdr = document.querySelector("header");
    if (!hdr) return null;
    t = document.createElement("div");
    t.id = "experiment-switcher";
    t.style.marginLeft = "auto";
    hdr.appendChild(t);
    return t;
  }

  function render(target, payload) {
    injectStyles();
    const wrap = document.createElement("div");
    wrap.className = "experiment-switcher";
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.title = payload.current_label;
    const label = document.createElement("label");
    label.textContent = "Experiment";
    label.htmlFor = "experiment-switcher-select";
    const sel = document.createElement("select");
    sel.id = "experiment-switcher-select";
    for (const opt of payload.options) {
      const o = document.createElement("option");
      o.value = opt.public_url;
      o.textContent = opt.label;
      o.dataset.mode = opt.mode;
      if (opt.mode === payload.current) o.selected = true;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => {
      const url = sel.value;
      if (url) window.location.href = url;
    });
    wrap.appendChild(dot);
    wrap.appendChild(label);
    wrap.appendChild(sel);
    target.replaceChildren(wrap);
  }

  try {
    const resp = await fetch("/api/experiment", { cache: "no-store" });
    if (!resp.ok) return;
    const payload = await resp.json();
    const target = findTarget();
    if (target) render(target, payload);
    if (typeof window.__experimentLoaded === "function") {
      try { window.__experimentLoaded(payload); } catch (_) {}
    }
    // Stash globally for other scripts that load after this one.
    window.__experiment = payload;
  } catch (e) {
    console.warn("experiment switcher failed:", e);
  }
})();
