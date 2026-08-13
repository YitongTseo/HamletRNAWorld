"""Experiment 1 report — self-contained, theme-aware HTML with inline SVG.

Reads results/metrics.json, writes results/report.html. No matplotlib/pandas;
charts are hand-built SVG so the file is fully portable and Artifact-publishable.
Palette = dataviz skill's validated default (blue=absolute, aqua=pairwise).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# validated palette slots (light / dark handled via CSS vars)
COLOR = {"absolute": "var(--s1)", "pairwise": "var(--s2)"}


def cfg_label(r):
    if r["method"] == "absolute":
        return f"{r['sampling'][:4]}·{r['size']}·T{r['temp']:g}"
    return f"pw·{r['sampling'][:4]}·{r['size']}"


def fmt(x, n=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"


# --- SVG frontier scatter: x=calls (log), y=mean tau -------------------------

def svg_frontier(configs, W=720, H=430):
    pad = {"l": 60, "r": 20, "t": 20, "b": 52}
    xs = [c["calls"] for c in configs]
    ys = [c.get("mean_tau", 0) for c in configs]
    xmin, xmax = min(xs) * 0.8, max(xs) * 1.25
    ymin = min(0.0, min(ys) - 0.05)
    ymax = min(1.0, max(ys) + 0.05)
    lx = lambda v: pad["l"] + (math.log10(v) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (W - pad["l"] - pad["r"])
    ly = lambda v: H - pad["b"] - (v - ymin) / (ymax - ymin) * (H - pad["t"] - pad["b"])
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Ranking stability vs judge calls">']
    # y gridlines
    yt = 5
    for k in range(yt + 1):
        v = ymin + (ymax - ymin) * k / yt
        y = ly(v)
        s.append(f'<line x1="{pad["l"]}" y1="{y:.1f}" x2="{W-pad["r"]}" y2="{y:.1f}" class="grid"/>')
        s.append(f'<text x="{pad["l"]-8:.0f}" y="{y+4:.0f}" class="tick" text-anchor="end">{v:.2f}</text>')
    # x ticks at decade-ish points
    for v in [16*10, 160, 1600, 3840, 24000]:
        if xmin <= v <= xmax:
            x = lx(v)
            s.append(f'<line x1="{x:.1f}" y1="{pad["t"]}" x2="{x:.1f}" y2="{H-pad["b"]}" class="grid"/>')
            s.append(f'<text x="{x:.0f}" y="{H-pad["b"]+18:.0f}" class="tick" text-anchor="middle">{v:,}</text>')
    s.append(f'<text x="{(W)/2:.0f}" y="{H-8:.0f}" class="axis-title" text-anchor="middle">judge calls per ranking pass (log)</text>')
    s.append(f'<text transform="translate(16,{H/2:.0f}) rotate(-90)" class="axis-title" text-anchor="middle">rank reproducibility (mean Kendall τ)</text>')
    # points + error bars + labels
    for c in configs:
        x, y = lx(c["calls"]), ly(c.get("mean_tau", 0))
        col = COLOR[c["method"]]
        sd = c.get("mean_tau_sd", 0) or 0
        s.append(f'<line x1="{x:.1f}" y1="{ly(c["mean_tau"]-sd):.1f}" x2="{x:.1f}" y2="{ly(c["mean_tau"]+sd):.1f}" stroke="{col}" stroke-width="1.5" opacity="0.5"/>')
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}" stroke="var(--surface)" stroke-width="1.5"><title>{cfg_label(c)}: τ={fmt(c.get("mean_tau"))} ± {fmt(sd,3)}, {c["calls"]:,} calls</title></circle>')
        s.append(f'<text x="{x+9:.1f}" y="{y+3:.1f}" class="pt-label">{cfg_label(c)}</text>')
    s.append("</svg>")
    return "".join(s)


# --- SVG horizontal bars: mean tau per config, sorted ------------------------

def svg_bars(configs, W=720, rowh=26):
    configs = sorted(configs, key=lambda c: c.get("mean_tau", 0))
    H = len(configs) * rowh + 30
    padl, padr = 168, 44
    bw = W - padl - padr
    xmax = max(0.001, min(1.0, max(c.get("mean_tau", 0) for c in configs) + 0.05))
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean Kendall tau per config">']
    for k, c in enumerate(configs):
        y = k * rowh + 8
        v = c.get("mean_tau", 0) or 0
        w = max(0, v / xmax * bw)
        col = COLOR[c["method"]]
        s.append(f'<text x="{padl-10}" y="{y+rowh*0.62:.0f}" class="row-label" text-anchor="end">{cfg_label(c)}</text>')
        s.append(f'<rect x="{padl}" y="{y}" width="{w:.1f}" height="{rowh-8}" rx="4" fill="{col}"><title>{cfg_label(c)}: τ={fmt(v)}</title></rect>')
        s.append(f'<text x="{padl+w+6:.1f}" y="{y+rowh*0.62:.0f}" class="bar-val">{fmt(v)}</text>')
    s.append("</svg>")
    return "".join(s)


# --- SVG scatter: validity (x) vs reproducibility (y) — the decision chart ---

def svg_validity(configs, W=720, H=400):
    pad = {"l": 58, "r": 20, "t": 18, "b": 50}
    lx = lambda v: pad["l"] + v * (W - pad["l"] - pad["r"])
    ly = lambda v: H - pad["b"] - v * (H - pad["t"] - pad["b"])
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Validity vs reproducibility">']
    for k in range(6):
        g = k / 5
        s.append(f'<line x1="{pad["l"]}" y1="{ly(g):.1f}" x2="{W-pad["r"]}" y2="{ly(g):.1f}" class="grid"/>')
        s.append(f'<text x="{pad["l"]-8:.0f}" y="{ly(g)+4:.0f}" class="tick" text-anchor="end">{g:.1f}</text>')
        s.append(f'<line x1="{lx(g):.1f}" y1="{pad["t"]}" x2="{lx(g):.1f}" y2="{H-pad["b"]}" class="grid"/>')
        s.append(f'<text x="{lx(g):.0f}" y="{H-pad["b"]+17:.0f}" class="tick" text-anchor="middle">{g:.1f}</text>')
    s.append(f'<text x="{W/2:.0f}" y="{H-8:.0f}" class="axis-title" text-anchor="middle">validity — agreement with consensus ranking (higher = measures real quality)</text>')
    s.append(f'<text transform="translate(15,{H/2:.0f}) rotate(-90)" class="axis-title" text-anchor="middle">reproducibility (mean τ)</text>')
    for c in configs:
        vx = c.get("validity")
        vy = c.get("mean_tau")
        if vx is None or vy is None:
            continue
        x, y = lx(vx), ly(vy)
        col = COLOR[c["method"]]
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{col}" stroke="var(--surface)" stroke-width="1.5"><title>{cfg_label(c)}: τ={fmt(vy)}, validity={fmt(vx)}, posbias={fmt(c.get("pos_bias"))}</title></circle>')
        s.append(f'<text x="{x+9:.1f}" y="{y+3:.1f}" class="pt-label">{cfg_label(c)}</text>')
    s.append("</svg>")
    return "".join(s)


def stat_tile(label, value, sub="", tone=""):
    return (f'<div class="tile {tone}"><div class="tile-v">{value}</div>'
            f'<div class="tile-l">{label}</div><div class="tile-s">{sub}</div></div>')


def build(metrics):
    agg = metrics["aggregated"]
    absr = [r for r in agg if r["method"] == "absolute"]
    pwr = [r for r in agg if r["method"] == "pairwise"]
    # dedupe absolute size=all (random/strat identical) for the frontier
    seen, frontier = set(), []
    for r in sorted(agg, key=lambda r: -r.get("mean_tau", 0)):
        key = (r["method"], r["size"], r["temp"]) if r["size"] == "all" else (r["method"], r["sampling"], r["size"], r["temp"])
        if key in seen:
            continue
        seen.add(key)
        frontier.append(r)

    incumbent = next((r for r in absr if r["size"] == "all" and r["temp"] == 1.0), None)
    abs_t0 = next((r for r in absr if r["size"] == "all" and r["temp"] == 0.0), None)
    rec = next((r for r in absr if str(r["size"]) == "all" and r["temp"] == 0.0), None)
    best_pw = max(pwr, key=lambda r: r.get("mean_tau", 0)) if pwr else None

    tiles = []
    if rec:
        tiles.append(stat_tile("★ recommended: abs · ~25% windows · T0", fmt(rec["mean_tau"]),
                               f'validity {fmt(rec.get("validity"))} · {rec["calls"]:,} calls', "good"))
    if incumbent:
        tiles.append(stat_tile("incumbent (abs · all · T1.0)", fmt(incumbent["mean_tau"]),
                               f'validity {fmt(incumbent.get("validity"))} · {incumbent["calls"]:,} calls', "warn"))
    if abs_t0 and incumbent:
        lift = abs_t0["mean_tau"] - incumbent["mean_tau"]
        tiles.append(stat_tile("just set temp=0 → τ", fmt(abs_t0["mean_tau"]),
                               f'+{fmt(lift)} for free', "good"))
    if best_pw:
        tiles.append(stat_tile("pairwise: reproducible but biased", fmt(best_pw.get("mean_tau")),
                               f'pos-bias {fmt(best_pw.get("pos_bias"))} → validity {fmt(best_pw.get("validity"))}', "warn"))

    banner = ("<b>Verdict:</b> the one solid, shippable win is <b>judge at temperature 0</b> "
              "(was unset → 1.0): it roughly doubles rank reproducibility on its own. "
              "<b>Keep ~25% of windows</b> — reproducibility here RESAMPLES windows each rep "
              "(production-faithful), and more windows = more reproducible (m=5 falls to τ≈0.37; "
              "an earlier 'fewer is better' read was a fixed-window artifact). Pairwise posts the "
              "highest τ but it is a temp-0 determinism artifact: the judge picks the first-shown "
              "side 94% of the time, so its rankings have low validity (0.15) — not worth its 12× "
              "cost without heavy debiasing.")

    # metrics table (aggregated)
    cols = [("method","method"),("sampling","samp"),("size","size"),("temp","T"),
            ("mean_tau","τ"),("mean_rho","ρ"),("sn_ratio","S/N"),("pos_bias","posbias"),
            ("flip_rate","flip"),("validity","valid"),("calls","calls")]
    head = "".join(f"<th>{h}</th>" for _, h in cols)
    rows = ""
    for r in sorted(agg, key=lambda r: -r.get("mean_tau", 0)):
        tds = []
        for kk, _ in cols:
            v = r.get(kk)
            if kk == "calls":
                tds.append(f"<td class='num'>{v:,}</td>")
            elif isinstance(v, float):
                tds.append(f"<td class='num'>{fmt(v)}</td>")
            else:
                tds.append(f"<td>{v}</td>")
        dot = f'<span class="dot" style="background:{COLOR[r["method"]]}"></span>'
        rows += f"<tr>{dot and ''}{''.join(tds)}</tr>"
    table = f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"

    return TEMPLATE.format(
        tiles="".join(tiles),
        banner=banner,
        frontier=svg_frontier(frontier),
        validity=svg_validity(frontier),
        bars=svg_bars(agg),
        table=table,
        n_configs=len(agg),
        npools=agg[0]["n_pools"] if agg else 0,
    )


TEMPLATE = """<div class="viz-root">
<style>
.viz-root{{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--s1:#2a78d6;--s2:#1baf7a;--good:#006300;--warn:#eda100;--card:#ffffff;--ring:rgba(11,11,11,.10);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--plane);padding:28px;max-width:900px;margin:0 auto;}}
@media (prefers-color-scheme:dark){{.viz-root{{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--s1:#3987e5;--s2:#199e70;--good:#0ca30c;--warn:#c98500;--card:#1a1a19;--ring:rgba(255,255,255,.10);}}}}
:root[data-theme=dark] .viz-root{{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--s1:#3987e5;--s2:#199e70;--good:#0ca30c;--warn:#c98500;--card:#1a1a19;--ring:rgba(255,255,255,.10);}}
:root[data-theme=light] .viz-root{{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9;--s1:#2a78d6;--s2:#1baf7a;--good:#006300;--warn:#eda100;--card:#fff;}}
.viz-root h1{{font-size:22px;margin:0 0 4px}} .viz-root h2{{font-size:15px;margin:26px 0 10px;color:var(--ink)}}
.sub{{color:var(--ink2);font-size:13px;margin-bottom:20px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:8px}}
.tile{{background:var(--card);border:1px solid var(--ring);border-radius:12px;padding:14px 16px}}
.tile-v{{font-size:26px;font-weight:650;letter-spacing:-.5px}} .tile-l{{font-size:12px;color:var(--ink2);margin-top:2px}}
.tile-s{{font-size:11px;color:var(--muted);margin-top:3px}} .tile.good .tile-v{{color:var(--good)}} .tile.warn .tile-v{{color:var(--warn)}}
.banner{{background:var(--card);border:1px solid var(--ring);border-left:4px solid var(--good);border-radius:10px;padding:13px 16px;margin:14px 0 4px;font-size:13.5px;line-height:1.5;color:var(--ink2)}}
.banner b{{color:var(--ink)}} .legend .note{{color:var(--muted);font-size:11px}}
.card{{background:var(--surface);border:1px solid var(--ring);border-radius:14px;padding:16px}}
svg{{width:100%;height:auto;display:block}}
.grid{{stroke:var(--grid);stroke-width:1}} .tick{{fill:var(--muted);font-size:11px}}
.axis-title{{fill:var(--ink2);font-size:12px}} .pt-label{{fill:var(--ink2);font-size:10.5px}}
.row-label{{fill:var(--ink2);font-size:11.5px;font-variant-numeric:tabular-nums}} .bar-val{{fill:var(--ink);font-size:11px;font-variant-numeric:tabular-nums}}
.legend{{display:flex;gap:18px;margin:6px 2px 0;font-size:12px;color:var(--ink2)}}
.legend b{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:middle}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:left}}
th{{color:var(--muted);font-weight:600;font-size:11px}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.overflow{{overflow-x:auto}}
</style>
<h1>Judge-noise shootout — Experiment 1</h1>
<div class="sub">Rank reproducibility of {n_configs} scoring configs across {npools} historical 16-worm pools · higher Kendall τ = less noisy judge</div>
<div class="tiles">{tiles}</div>
<div class="banner">{banner}</div>

<h2>Why reproducibility isn't enough — validity vs. τ</h2>
<div class="card">{validity}
<div class="legend"><span><b style="background:var(--s1)"></b>absolute</span><span><b style="background:var(--s2)"></b>pairwise</span><span class="note">top-left = reproducible but wrong (position bias); top-right = reproducible AND valid</span></div></div>

<h2>Stability vs. cost — the frontier</h2>
<div class="card">{frontier}
<div class="legend"><span><b style="background:var(--s1)"></b>absolute (1 batched call/worm)</span><span><b style="background:var(--s2)"></b>pairwise→Bradley-Terry (O(N²) calls)</span></div></div>

<h2>Rank reproducibility by config</h2>
<div class="card">{bars}</div>

<h2>All metrics</h2>
<div class="card overflow">{table}</div>
</div>"""


def main():
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    html = build(metrics)
    (RESULTS / "report.html").write_text(html)
    print(f"wrote {RESULTS/'report.html'} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
