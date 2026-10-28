"""The tasting bench — a human judge for a wormlet lineage.

Run as its OWN process, deliberately not as routes on server/app.py:

  * The sim process leaks to ~2.2 GB over about two days and dies above
    ~1.9 GB if anything asks it to do real work — reading the lineage
    through /api/generations killed it twice on 2026-09-11. Serving a
    judging page is real work.
  * A verdict must still be enterable when the sim is wedged, restarting,
    or waiting on its own watchdog. Judging is the critical path of a
    human-paced lineage; it cannot share a fate with the simulation.
  * There is no simulation in here at all, so there is nothing to leak.

It owns no state. Everything is files under WORMLET_DATA_DIR:

    ballots/<flask>/gen-NNNN/<worm>.json   written by server/judge.py
    verdicts/<flask>/gen-NNNN.json         written here
    commonplace.jsonl                      append-only, written here

WHAT THE PAGE ASKS FOR, AND WHY IT ASKS SO LITTLE
-------------------------------------------------
Selection is rank-based: evolution.nes_update weights children through
rank_weights, which for ten worms is

    [1.331  0.638  0.233  -0.055  -0.278  -0.374 x5]

The top three carry 100% of the positive gradient weight and ranks 6-10
are mathematically identical. So the page asks for an ordered top few and
nothing else. Asking anyone to rank ten would be asking them to supply
information the code cannot read.

Stars are a keepsake, not a score — see the human-backend block in
server/judge.py for why letting them move fitness would let the arithmetic
contradict the person's own stated order.

BLIND BY DEFAULT
----------------
Worms are shown as A, B, C... and their names are revealed only after the
verdict is filed. The lineage has named worms and a person who recognises
"last generation's winner" is no longer judging the poem. Set
WORMLET_BENCH_BLIND=0 to turn it off.
"""
from __future__ import annotations

import html
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from server import judge

DATA_ROOT = Path(os.environ.get("WORMLET_DATA_DIR", "/var/wormlet/data"))
BALLOTS = DATA_ROOT / "ballots"
VERDICTS = DATA_ROOT / "verdicts"
COMMONPLACE = DATA_ROOT / "commonplace.jsonl"

# How many of each worm's windows the page shows. The trade is measured, in
# judge.py's SAMPLE_FRACTION note: fewer windows read means a less
# reproducible ranking (mean Kendall tau 0.37 at 5 windows vs 0.60 at the
# full 25% sample). Five keeps a flask-generation at roughly 530 words —
# about two minutes — at a known cost in ranking noise. Raise it if you want
# a steadier critic and have the patience.
SHOW_N = int(os.environ.get("WORMLET_BENCH_SHOW", "5"))
# How many ranks the page accepts. Five is the whole of the information the
# NES step can use; three is where almost all of it is.
MAX_RANKS = int(os.environ.get("WORMLET_BENCH_RANKS", "5"))
BLIND = os.environ.get("WORMLET_BENCH_BLIND", "1") != "0"
# Local model for the coherence axis. Unset => no model, and coherence
# mirrors the human call (documented dead column, see judge.py).
OLLAMA_MODEL = os.environ.get("WORMLET_BENCH_OLLAMA_MODEL", "").strip()

app = FastAPI(title="wormlet tasting bench")


# --- ballots ----------------------------------------------------------------

def _pending() -> tuple[str, int, Path] | None:
    """Oldest flask-generation that has ballots and no verdict.

    OLDEST, not newest: a lineage waits at corpus exhaustion, so there is
    normally exactly one pending generation. If a second ever appears,
    judging them out of order would apply a verdict to the wrong
    generation's genomes.
    """
    found: list[tuple[int, str, Path]] = []
    if not BALLOTS.is_dir():
        return None
    for flask_dir in sorted(BALLOTS.iterdir()):
        if not flask_dir.is_dir():
            continue
        for gen_dir in sorted(flask_dir.iterdir()):
            if not gen_dir.is_dir() or not gen_dir.name.startswith("gen-"):
                continue
            try:
                gen = int(gen_dir.name[4:])
            except ValueError:
                continue
            if (VERDICTS / flask_dir.name / f"gen-{gen:04d}.json").exists():
                continue
            if any(gen_dir.glob("*.json")):
                found.append((gen, flask_dir.name, gen_dir))
    if not found:
        return None
    gen, flask, path = sorted(found)[0]
    return flask, gen, path


def _load_ballots(gen_dir: Path) -> list[dict]:
    out = []
    for p in sorted(gen_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue  # a torn or half-written ballot is skipped, not fatal
    return out


# --- the coherence pass (optional, local model only) ------------------------

def _coherence_cache(flask: str, gen: int) -> Path:
    return BALLOTS / flask / f"gen-{gen:04d}" / ".coherence.json"


def _coherence(flask: str, gen: int, ballots: list[dict]) -> tuple[dict, str | None]:
    """{worm: {token_offset: coherence}}, plus an error string to show the
    person if the local model could not be reached.

    Cached to disk: the page is reloaded and the model is slow, and a second
    pass would also give a second answer. Computing it once per generation
    makes what is shown stable across reloads.
    """
    if not OLLAMA_MODEL:
        return {}, None
    cache = _coherence_cache(flask, gen)
    if cache.exists():
        try:
            return json.loads(cache.read_text()), None
        except (OSError, ValueError):
            pass
    out: dict[str, dict[str, int]] = {}
    try:
        for b in ballots:
            sampled = [(w["idx"], w["text"].split()) for w in b["windows"]]
            # Model passed explicitly, not via WORMLET_JUDGE_MODEL: setting a
            # process-global env var around a call is not safe in a server
            # that can be handling two requests at once.
            scored = judge.score_windows_openai(sampled,
                                                b.get("corpus", "hamlet"),
                                                model=OLLAMA_MODEL)
            out[b["worm"]] = {str(i): c for i, (_e, c) in scored.items()}
    except Exception as e:  # unreachable box, timeout, bad response shape
        return {}, f"{type(e).__name__}: {e}"
    try:
        judge._write_atomic(cache, json.dumps(out))
    except OSError:
        pass  # cache is an optimisation; a read-only dir must not break the page
    return out, None


def _shortlist(ballot: dict, coh: dict, seed: int) -> list[dict]:
    """Which of a worm's windows the person is shown.

    With a local model: its most coherent windows, so the eye lands on the
    fragments that are language rather than on noise. Without one: a
    DETERMINISTIC random sample, seeded by generation and worm name — not
    the first N, which would show everyone the opening of every poem and
    systematically judge worms on how they start.

    Either way this only decides what is READ. Fitness is computed over
    every sampled window in the ballot, so the denominator — and with it
    the FITNESS_WINDOW_FLOOR volume semantics — is untouched by what the
    page chose to display.
    """
    windows = ballot["windows"]
    if len(windows) <= SHOW_N:
        return windows
    per_worm = coh.get(ballot["worm"]) or {}
    if per_worm:
        ranked = sorted(windows,
                        key=lambda w: -int(per_worm.get(str(w["idx"]), 0)))
        return sorted(ranked[:SHOW_N], key=lambda w: w["idx"])
    rng = random.Random(f"{seed}:{ballot['worm']}")
    return sorted(rng.sample(windows, SHOW_N), key=lambda w: w["idx"])


# --- routes -----------------------------------------------------------------

@app.get("/healthz")
def healthz():
    p = _pending()
    return {"ok": True, "pending": None if p is None else
            {"flask": p[0], "generation": p[1]}}


@app.get("/", response_class=HTMLResponse)
def bench():
    p = _pending()
    if p is None:
        return HTMLResponse(_page_shell(
            "<h1>nothing to judge</h1><p class=q>The lineage is still "
            "eating. This page will have a generation on it when the "
            "corpus next runs out.</p>"))
    flask, gen, gen_dir = p
    ballots = _load_ballots(gen_dir)
    if not ballots:
        return HTMLResponse(_page_shell("<h1>ballots unreadable</h1>"))
    coh, coh_err = _coherence(flask, gen, ballots)

    # Presentation order is shuffled per generation, so position on the page
    # is not worm identity across generations.
    order = sorted(ballots, key=lambda b: b["worm"])
    random.Random(f"{flask}:{gen}").shuffle(order)

    cards = []
    for n, b in enumerate(order):
        label = chr(ord("A") + n) if BLIND else html.escape(b["worm"])
        lines = "".join(
            f'<li data-idx="{w["idx"]}">{html.escape(w["text"])}</li>'
            for w in _shortlist(b, coh, gen))
        cards.append(
            f'<article class=w data-worm="{html.escape(b["worm"])}">'
            f'<header><span class=pick></span><h2>{label}</h2></header>'
            f'<ol class=lines>{lines}</ol></article>')

    banner = ""
    if coh_err:
        banner = ('<p class="q warn">No local model — fragments are a random '
                  f'sample, not the most coherent ones, and the coherence '
                  f'column of this generation will mirror your call. '
                  f'({html.escape(coh_err)})</p>')
    elif not OLLAMA_MODEL:
        banner = ('<p class="q warn">No local model configured — coherence '
                  'will mirror your call.</p>')

    body = f"""
<h1>{html.escape(flask)} · generation {gen}</h1>
<p class=q>Tap your best, then second, then third. Up to {MAX_RANKS}.
Everything you do not rank is tied — the selection step cannot tell
them apart. Tap a line to keep it.</p>
{banner}
<div class=grid>{''.join(cards)}</div>
<div class=bar><button id=go disabled>file verdict</button>
<span id=status></span></div>
<script>
const FLASK={json.dumps(flask)}, GEN={gen}, MAX={MAX_RANKS};
const picks=[], stars={{}};
function paint(){{
  document.querySelectorAll('article.w').forEach(a=>{{
    const i=picks.indexOf(a.dataset.worm);
    a.classList.toggle('picked', i>=0);
    a.querySelector('.pick').textContent = i>=0 ? (i+1) : '';
  }});
  document.getElementById('go').disabled = picks.length===0;
}}
document.querySelectorAll('article.w').forEach(a=>{{
  a.querySelector('header').onclick=()=>{{
    const w=a.dataset.worm, i=picks.indexOf(w);
    if(i>=0) picks.splice(i,1); else if(picks.length<MAX) picks.push(w);
    paint();
  }};
  a.querySelectorAll('.lines li').forEach(li=>{{
    li.onclick=()=>{{
      const w=a.dataset.worm, idx=+li.dataset.idx;
      stars[w]=stars[w]||[];
      const j=stars[w].indexOf(idx);
      if(j>=0) stars[w].splice(j,1); else stars[w].push(idx);
      li.classList.toggle('star');
    }};
  }});
}});
document.getElementById('go').onclick=async()=>{{
  const btn=document.getElementById('go'); btn.disabled=true;
  const s=document.getElementById('status'); s.textContent='filing...';
  const r=await fetch('/verdict',{{method:'POST',
    headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{flask:FLASK,generation:GEN,top:picks,starred:stars}})}});
  const j=await r.json();
  if(!r.ok){{ s.textContent='failed: '+(j.detail||r.status); btn.disabled=false; return; }}
  s.textContent='filed.';
  document.querySelector('.grid').innerHTML =
    '<p class=q>' + j.revealed.map((w,i)=>(i+1)+'. '+w).join('<br>') + '</p>';
}};
</script>"""
    return HTMLResponse(_page_shell(body))


@app.post("/verdict")
async def verdict(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"detail": "bad json"}, status_code=400)
    flask, gen = str(body.get("flask", "")), body.get("generation")
    top = [str(w) for w in (body.get("top") or [])]
    if not flask or not isinstance(gen, int) or not top:
        return JSONResponse({"detail": "flask, generation and top required"},
                            status_code=400)

    gen_dir = BALLOTS / flask / f"gen-{gen:04d}"
    ballots = _load_ballots(gen_dir)
    known = {b["worm"] for b in ballots}
    known_windows = {(b["worm"], w["idx"]) for b in ballots for w in b["windows"]}
    if not known:
        return JSONResponse({"detail": f"no ballots for {flask} gen-{gen:04d}"},
                            status_code=404)
    unknown = [w for w in top if w not in known]
    if unknown:
        # A name not on this generation's ballot would score nobody and
        # silently flatten the field; refuse it rather than file it.
        return JSONResponse({"detail": f"not on this ballot: {unknown}"},
                            status_code=400)
    if len(set(top)) != len(top):
        return JSONResponse({"detail": "duplicate worm in top"}, status_code=400)

    out = VERDICTS / flask / f"gen-{gen:04d}.json"
    if out.exists():
        # Overwriting is how you would accidentally re-judge a generation
        # that has already been selected on. The rollover reads the verdict
        # that was there when it ran.
        return JSONResponse({"detail": "already judged"}, status_code=409)

    starred = {str(k): [int(i) for i in v]
               for k, v in (body.get("starred") or {}).items() if v}
    # A star must name a window that is actually on that worm's ballot.
    # Without this the keepsake silently records a blank line: the text is
    # looked up by (worm, offset), and an offset that was never sampled
    # looks up to "". Caught by a test that starred plausible-looking
    # offsets which happened not to be in the 25% sample.
    stray = [(w, i) for w, idxs in starred.items() for i in idxs
             if (w, i) not in known_windows]
    if stray:
        return JSONResponse({"detail": f"starred windows not on the ballot: "
                                       f"{stray}"}, status_code=400)
    coh, _ = _coherence(flask, gen, ballots)
    judge._write_atomic(out, json.dumps({
        "flask": flask, "generation": gen,
        "judge": os.environ.get("WORMLET_JUDGE_HUMAN", "unnamed"),
        "decided_at": time.time(),
        "top": top, "starred": starred,
        **({"coherence": coh} if coh else {}),
        "shown_per_worm": SHOW_N,
        "blind": BLIND,
    }, indent=1))
    _append_commonplace(flask, gen, gen_dir, starred)
    return {"ok": True, "revealed": top}


def _append_commonplace(flask: str, gen: int, gen_dir: Path,
                        starred: dict[str, list[int]]) -> None:
    """Append-only keepsake of every line the person kept. Separate from the
    generation artifacts on purpose: artifacts are the lineage's record and
    get pruned with it, this outlives any particular run."""
    if not starred:
        return
    text_of = {}
    for b in _load_ballots(gen_dir):
        for w in b["windows"]:
            text_of[(b["worm"], w["idx"])] = w["text"]
    try:
        with COMMONPLACE.open("a") as fh:
            for worm, idxs in starred.items():
                for i in idxs:
                    fh.write(json.dumps({
                        "at": time.time(), "flask": flask, "generation": gen,
                        "worm": worm, "idx": i,
                        "text": text_of.get((worm, i), ""),
                    }) + "\n")
    except OSError:
        pass  # the verdict is already filed; a keepsake must not undo it


@app.get("/book", response_class=HTMLResponse)
def book():
    """Everything ever starred, newest first."""
    rows = []
    if COMMONPLACE.exists():
        for line in reversed(COMMONPLACE.read_text().splitlines()):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows.append(f'<li>{html.escape(r.get("text", ""))}'
                        f'<span class=q> — {html.escape(r.get("flask", ""))} '
                        f'gen {r.get("generation")}</span></li>')
    return HTMLResponse(_page_shell(
        "<h1>commonplace book</h1><ol class=lines>"
        + ("".join(rows) or "<li class=q>nothing kept yet</li>") + "</ol>"))


def _page_shell(body: str) -> str:
    return f"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>tasting bench</title>
<style>
:root{{color-scheme:dark}}
body{{background:#17150f;color:#e8e0cc;font:16px/1.5 Georgia,serif;
 margin:0;padding:1rem;max-width:46rem}}
h1{{font-size:1.1rem;font-weight:normal;letter-spacing:.04em;margin:0 0 .3rem}}
h2{{font-size:1rem;margin:0;font-weight:normal}}
.q{{color:#8e866f;font-size:.85rem;margin:.2rem 0 1rem}}
.warn{{color:#c08a4a}}
article.w{{border:1px solid #352f22;border-radius:4px;margin:.6rem 0;
 background:#1d1a13}}
article.w header{{display:flex;gap:.6rem;align-items:center;padding:.6rem .8rem;
 cursor:pointer;user-select:none;border-bottom:1px solid #352f22}}
article.w.picked{{border-color:#c08a4a}}
.pick{{display:inline-block;min-width:1.5rem;height:1.5rem;line-height:1.5rem;
 text-align:center;border:1px solid #4a4231;border-radius:50%;font-size:.8rem}}
article.w.picked .pick{{background:#c08a4a;color:#17150f;border-color:#c08a4a}}
ol.lines{{margin:0;padding:.4rem .8rem .6rem 2rem}}
ol.lines li{{padding:.35rem 0;cursor:pointer}}
ol.lines li.star{{color:#e8c07a}}
ol.lines li.star::after{{content:" \\2605";font-size:.8em}}
.bar{{position:sticky;bottom:0;background:#17150f;padding:.8rem 0;
 display:flex;gap:.8rem;align-items:center}}
button{{font:inherit;background:#c08a4a;color:#17150f;border:0;
 border-radius:3px;padding:.5rem 1.2rem;cursor:pointer}}
button:disabled{{background:#3a3426;color:#6a6252;cursor:default}}
</style>
{body}
"""
