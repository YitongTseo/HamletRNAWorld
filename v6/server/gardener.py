"""Per-generation gardener's log.

Three Claude calls per generation:

  Round 1  — show the gardener a catalog of past generations (numbers,
             scores, one-line log summaries) and let them pick up to 5
             logs they want to read.
  Round 2  — show those logs in full, plus the same catalog, and let
             them pick a few more logs + up to 5 specific (gen, worm)
             pairs whose poetry they want to sample.
  Round 3  — show all selected logs, all selected sampled poems, and
             this generation's top fragments. Ask for a 2-sentence log
             entry, or the literal word PASS to rest this generation.

The gardener is Opus 4.7 with adaptive thinking. The briefing document
(docs/specs/2026-05-26-gardener-briefing.md) is included verbatim in the
system prompt and is identical across every call across every
generation, so prompt caching covers most of the input cost. Selection
rounds use structured JSON output; the final write is free-form prose
or the exact string PASS.

Public API:
    maybe_write_log(gen_dir, scored_by_worm, fitness_by_worm,
                    generation_num, group_name, best_score_history)
"""
from __future__ import annotations

import datetime
import json
import os
import re
from pathlib import Path

import anthropic

from server.judge import ScoredWindow

V6_ROOT = Path(__file__).resolve().parent.parent
BRIEFING_PATH = V6_ROOT / "docs" / "specs" / "2026-05-26-gardener-briefing.md"

MODEL = "claude-opus-4-7"
MAX_LOG_TOKENS = 300            # 2 sentences fits comfortably; hard cap
MAX_SELECTION_TOKENS = 200      # JSON selection blobs are tiny

MAX_LOGS_PER_ROUND = 5
MAX_POEMS_TO_SAMPLE = 5
TOP_WORMS_TO_SHOW = 3
TOP_WINDOWS_PER_WORM = 3


def _briefing() -> str:
    try:
        return BRIEFING_PATH.read_text()
    except Exception:
        return "(briefing document missing; speak from your own context)"


# ----- catalog + corpus access ---------------------------------------------

def _read_log(gen_root: Path, gen: int) -> str | None:
    """Return the prose of the gardener's log for `gen`, or None if absent
    (gardener PASSed or the gen predates the feature)."""
    path = gen_root / f"gen-{gen:04d}" / "gardeners_log.md"
    return path.read_text().strip() if path.exists() else None


def _one_line_summary(text: str | None) -> str:
    if text is None:
        return "(rested — no log)"
    # Strip the timestamp line if present; take the first sentence.
    body = re.sub(r"^#.*?\n+", "", text, count=1).strip()
    first = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0]
    return first[:200]


def _catalog(gen_root: Path, current_gen: int, best_score_history: list[float]) -> str:
    """A compact markdown index of every prior generation, one line each.
    Includes best score and a one-line excerpt of the gardener's log."""
    if current_gen <= 1:
        return "(this is generation 1 — no prior generations exist)"
    lines = []
    for g in range(1, current_gen):
        score = best_score_history[g - 1] if g - 1 < len(best_score_history) else None
        score_str = f"{score:.2f}" if score is not None else "—"
        excerpt = _one_line_summary(_read_log(gen_root, g))
        lines.append(f"- gen {g} · best={score_str} · {excerpt}")
    return "\n".join(lines)


def _format_this_gen_fragments(
    scored_by_worm: dict[str, list[ScoredWindow]],
    fitness_by_worm: dict[str, float],
) -> str:
    top_worms = sorted(fitness_by_worm.items(), key=lambda kv: -kv[1])[:TOP_WORMS_TO_SHOW]
    lines = []
    for name, fit in top_worms:
        lines.append(f"### {name} — fitness {fit:.2f}")
        windows = scored_by_worm.get(name, [])
        ranked = sorted(
            windows,
            key=lambda w: -(1.5 * (w.emotional / 100.0) ** 2.5 + (w.coherence / 100.0) ** 2.5),
        )[:TOP_WINDOWS_PER_WORM]
        for w in ranked:
            lines.append(f"- (E={w.emotional} C={w.coherence}) “{' '.join(w.tokens)}”")
        lines.append("")
    return "\n".join(lines).rstrip()


def _sample_window_from_gen(gen_root: Path, gen: int, worm: str) -> str | None:
    """Pull the single top-scoring window for (gen, worm) from disk and
    return a one-line render, or None if the data isn't present."""
    scores_path = gen_root / f"gen-{gen:04d}" / worm / "scores.jsonl"
    if not scores_path.exists():
        return None
    best = None
    best_q = -1.0
    with open(scores_path) as f:
        for line in f:
            try:
                s = json.loads(line)
            except Exception:
                continue
            q = 1.5 * (s["emotional"] / 100.0) ** 2.5 + (s["coherence"] / 100.0) ** 2.5
            if q > best_q:
                best_q = q
                best = s
    if not best:
        return None
    text = " ".join(best.get("tokens") or [])
    return f"gen {gen} · {worm} · (E={best['emotional']} C={best['coherence']}) “{text}”"


# ----- system prompts ------------------------------------------------------

GARDENER_TONE = """You are the gardener of the ecosystem described in the briefing above. You write a short private log — at most two sentences — after each generation. You may, at any generation, respond with the single word PASS instead of writing, and we will record that you chose to rest. The briefing tells you what to look for and what tone to take."""

ROUND1_INSTRUCTIONS = """You are about to write a log entry for the current generation. Before you do, you may read past gardener's logs.

Below is a catalog of every prior generation: the best fitness score that generation reached and a one-line excerpt of that generation's log (or "(rested)" if the gardener chose to PASS).

Pick up to {max_logs} generations whose full logs you want to read. Choose based on what would help you write today's entry — perhaps the most recent few, perhaps the ones whose excerpts surprised you, perhaps a stretch of the timeline you want to see continuity across. If no prior context would help you today, return an empty list.

Respond ONLY with the JSON object specified by the output format. No prose."""

ROUND2_INSTRUCTIONS = """You have read the logs from round 1. Below they appear in full, in chronological order.

Now you may pick:
  - up to {max_logs} additional generations' logs to read, and
  - up to {max_poems} specific (generation, worm) pairs whose top-scoring poem window you want to see.

The catalog from round 1 is included again. Pick worms and generations that would let you confirm or deepen whatever observation you're forming. If the round-1 logs gave you what you needed, return empty lists.

Respond ONLY with the JSON object specified by the output format. No prose."""

ROUND3_INSTRUCTIONS = """You have done your reading. Below are all the logs you asked for, the sampled poem windows you selected, and the top fragments from THIS generation.

Now write your log entry. Two sentences at most. You may respond with the single word PASS to rest this generation.

You do not need to summarize what the worms did — the data does that. The log is for observations: a specific scientific concern (σ feels off, decay is too short, UMAP is the wrong fit, …), a specific aesthetic notice (Carol's vocabulary is drifting toward sea-words, Frank produced a line that scans), or a meta-observation about the trajectory across generations.

Caring but firm. Level-headed scientist who is rooting for the experiment but will say plainly when something isn't working."""


# ----- Round 1: pick logs to read ------------------------------------------

ROUND1_SCHEMA = {
    "type": "object",
    "properties": {
        "logs_to_read": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "maxItems": MAX_LOGS_PER_ROUND,
            "description": "Generation numbers whose full gardener's log you want to read.",
        },
    },
    "required": ["logs_to_read"],
    "additionalProperties": False,
}


def _round1_select(client, catalog: str, current_gen: int, briefing: str) -> list[int]:
    sys = [
        {"type": "text", "text": briefing, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": GARDENER_TONE},
    ]
    user = (
        f"Catalog of prior generations (current gen = {current_gen}):\n\n{catalog}\n\n"
        + ROUND1_INSTRUCTIONS.format(max_logs=MAX_LOGS_PER_ROUND)
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_SELECTION_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": ROUND1_SCHEMA}},
        system=sys,
        messages=[{"role": "user", "content": user}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        return json.loads(text).get("logs_to_read", [])[:MAX_LOGS_PER_ROUND]
    except Exception:
        return []


# ----- Round 2: pick more logs + poems -------------------------------------

ROUND2_SCHEMA = {
    "type": "object",
    "properties": {
        "logs_to_read": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "maxItems": MAX_LOGS_PER_ROUND,
        },
        "poems_to_read": {
            "type": "array",
            "maxItems": MAX_POEMS_TO_SAMPLE,
            "items": {
                "type": "object",
                "properties": {
                    "gen": {"type": "integer", "minimum": 1},
                    "worm": {"type": "string"},
                },
                "required": ["gen", "worm"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["logs_to_read", "poems_to_read"],
    "additionalProperties": False,
}


def _round2_select(client, catalog: str, round1_logs_rendered: str, briefing: str) -> dict:
    sys = [
        {"type": "text", "text": briefing, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": GARDENER_TONE},
    ]
    user = (
        f"Catalog (for reference):\n\n{catalog}\n\n"
        f"Round 1 logs you asked for:\n\n{round1_logs_rendered}\n\n"
        + ROUND2_INSTRUCTIONS.format(max_logs=MAX_LOGS_PER_ROUND, max_poems=MAX_POEMS_TO_SAMPLE)
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_SELECTION_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": ROUND2_SCHEMA}},
        system=sys,
        messages=[{"role": "user", "content": user}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        out = json.loads(text)
        return {
            "logs_to_read": out.get("logs_to_read", [])[:MAX_LOGS_PER_ROUND],
            "poems_to_read": out.get("poems_to_read", [])[:MAX_POEMS_TO_SAMPLE],
        }
    except Exception:
        return {"logs_to_read": [], "poems_to_read": []}


# ----- Round 3: write the log ---------------------------------------------

def _round3_write(
    client,
    briefing: str,
    all_logs_rendered: str,
    poem_samples_rendered: str,
    this_gen_fragments: str,
    current_gen: int,
    group_name: str,
) -> str:
    sys = [
        {"type": "text", "text": briefing, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": GARDENER_TONE},
    ]
    user = (
        f"You are writing the log for generation {current_gen} of group \"{group_name}\".\n\n"
        f"Logs you read:\n\n{all_logs_rendered or '(none)'}\n\n"
        f"Sampled poem windows you asked for:\n\n{poem_samples_rendered or '(none)'}\n\n"
        f"Top fragments from THIS generation:\n\n{this_gen_fragments or '(none yet)'}\n\n"
        + ROUND3_INSTRUCTIONS
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_LOG_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=sys,
        messages=[{"role": "user", "content": user}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


# ----- public entry point --------------------------------------------------

def _render_logs(gen_root: Path, gens: list[int]) -> str:
    """Render selected gardener's logs as one markdown blob, chronological."""
    out = []
    for g in sorted(set(gens)):
        text = _read_log(gen_root, g)
        if text is None:
            out.append(f"## gen {g} — (rested, no log)\n")
        else:
            out.append(f"## gen {g}\n\n{text}\n")
    return "\n".join(out)


def _render_poem_samples(gen_root: Path, picks: list[dict]) -> str:
    """Render selected (gen, worm) samples as a markdown list."""
    out = []
    for p in picks:
        try:
            gen = int(p["gen"])
            worm = str(p["worm"])
        except (KeyError, ValueError, TypeError):
            continue
        rendered = _sample_window_from_gen(gen_root, gen, worm)
        if rendered:
            out.append(f"- {rendered}")
    return "\n".join(out)


def maybe_write_log(
    gen_dir: Path,
    scored_by_worm: dict[str, list[ScoredWindow]],
    fitness_by_worm: dict[str, float],
    generation_num: int,
    group_name: str,
    best_score_history: list[float] | None = None,
) -> Path | None:
    """Run the three-call gardener flow and (maybe) write a log file.
    Returns the path to the written log, or None if PASSed/skipped/failed.

    `best_score_history` is the per-generation list of best scores from
    GenerationState (used to populate the catalog). May be None on first
    generation."""
    if os.environ.get("WORMLET_GARDENER", "1") == "0":
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    briefing = _briefing()
    gen_root = gen_dir.parent  # data/generations/<group>/
    history = list(best_score_history or [])
    catalog = _catalog(gen_root, generation_num, history)
    this_gen_fragments = _format_this_gen_fragments(scored_by_worm, fitness_by_worm)

    try:
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"[GARDENER] anthropic client init failed: {e}", flush=True)
        return None

    # Round 1: pick logs (skipped if this is gen 1, no history to read).
    try:
        if generation_num > 1:
            r1_picks = _round1_select(client, catalog, generation_num, briefing)
            r1_rendered = _render_logs(gen_root, r1_picks)
        else:
            r1_picks, r1_rendered = [], "(no prior generations)"
    except Exception as e:
        print(f"[GARDENER] round 1 failed: {e}", flush=True)
        r1_picks, r1_rendered = [], "(round 1 unavailable)"

    # Round 2: pick more logs + poems.
    try:
        if generation_num > 1:
            r2 = _round2_select(client, catalog, r1_rendered, briefing)
            all_log_picks = list(set(r1_picks) | set(r2["logs_to_read"]))
            all_logs_rendered = _render_logs(gen_root, all_log_picks)
            poems_rendered = _render_poem_samples(gen_root, r2["poems_to_read"])
        else:
            all_logs_rendered = ""
            poems_rendered = ""
    except Exception as e:
        print(f"[GARDENER] round 2 failed: {e}", flush=True)
        all_logs_rendered = r1_rendered
        poems_rendered = ""

    # Round 3: write the actual log.
    try:
        text = _round3_write(
            client, briefing, all_logs_rendered, poems_rendered,
            this_gen_fragments, generation_num, group_name,
        )
    except Exception as e:
        print(f"[GARDENER] round 3 failed: {e}", flush=True)
        return None

    if not text:
        return None
    if text.strip().upper() == "PASS" or text.strip().split()[0:1] == ["PASS"]:
        (gen_dir / "gardeners_log.skipped").write_text(
            "the gardener chose not to write this generation.\n"
        )
        return None

    # Stamp the log with generation + UTC timestamp at the top of the file.
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = f"# gen {generation_num} — {ts}\n\n"
    log_path = gen_dir / "gardeners_log.md"
    log_path.write_text(header + text + "\n")
    return log_path


# ----- Meta-gardener (observes all flasks) ---------------------------------
#
# Same three-call shape as the per-flask gardener above. Differences:
#   - Catalog renders per-epoch lines with per-flask best scores instead of
#     one score per row, and the past-log paths are under data/generations/
#     meta/gen-NNNN/ rather than under any single flask.
#   - This-epoch fragments are gathered from ALL flasks (top worm × top
#     window per flask) so the meta-gardener sees the whole garden at once.
#   - Sampled poems can be drawn from any (flask, gen, worm) triple, not
#     just a single flask.

META_GARDENER_TONE = """You are the meta-gardener observing all four flasks of the ecosystem described in the briefing above. You write a short shared log — at most two sentences — once per epoch (after all flasks have completed a generation). You may, at any epoch, respond with the single word PASS instead of writing, and we will record that you chose to rest."""


def _meta_log_root(flasks_root: Path) -> Path:
    """Where meta-gardener logs live. flasks_root is the v6/data/generations
    parent; meta logs are siblings of the per-flask gen subdirs."""
    return flasks_root / "meta"


def _read_meta_log(meta_root: Path, gen: int) -> str | None:
    path = meta_root / f"gen-{gen:04d}" / "gardeners_log.md"
    return path.read_text().strip() if path.exists() else None


def _meta_catalog(meta_root: Path, flask_histories: dict[str, list[float]], current_epoch: int) -> str:
    """One line per past epoch: per-flask best scores + meta-log excerpt."""
    if current_epoch <= 1:
        return "(this is epoch 1 — no prior meta-logs exist)"
    flask_names = sorted(flask_histories.keys())
    lines = []
    for ep in range(1, current_epoch):
        scores = []
        for fn in flask_names:
            h = flask_histories[fn]
            if ep - 1 < len(h):
                scores.append(f"{fn}={h[ep - 1]:.2f}")
        excerpt = _one_line_summary(_read_meta_log(meta_root, ep))
        scores_str = " · ".join(scores) if scores else "—"
        lines.append(f"- epoch {ep} · {scores_str} · {excerpt}")
    return "\n".join(lines)


def _format_all_flasks_fragments(flasks) -> str:
    """One section per flask, showing the flask's top worm + top window."""
    from server.judge import ScoredWindow  # avoid circular import at module load
    out: list[str] = []
    for flask in flasks:
        # Pull the top worm's best window from the current generation's
        # scores.jsonl. Read fresh from disk so we don't need to plumb
        # per-flask scoring through the meta-gardener entry signature.
        cur_gen = flask.state.generation if flask.state else 0
        if cur_gen < 1:
            out.append(f"### {flask.display}: (no completed generation yet)")
            continue
        gen_dir = Path(__file__).resolve().parent.parent / "data" / "generations" / flask.name / f"gen-{cur_gen:04d}"
        out.append(f"### {flask.display} — gen {cur_gen}")
        # Read each worm's fitness from fitness.json, find top fitness, then
        # pick its top window from scores.jsonl.
        top_name, top_fit = None, -1.0
        for worm_dir in sorted(gen_dir.glob("*/")):
            fpath = worm_dir / "fitness.json"
            if not fpath.exists():
                continue
            try:
                fit = json.loads(fpath.read_text()).get("fitness", 0.0)
            except Exception:
                continue
            if fit > top_fit:
                top_fit = fit
                top_name = worm_dir.name
        if top_name is None:
            out.append("- (no fitness data)")
            out.append("")
            continue
        scores_path = gen_dir / top_name / "scores.jsonl"
        best, best_q = None, -1.0
        if scores_path.exists():
            with open(scores_path) as f:
                for line in f:
                    try:
                        s = json.loads(line)
                    except Exception:
                        continue
                    q = 1.5 * (s["emotional"] / 100.0) ** 2.5 + (s["coherence"] / 100.0) ** 2.5
                    if q > best_q:
                        best_q = q
                        best = s
        out.append(f"- top worm: {top_name} (fitness {top_fit:.2f})")
        if best:
            text = " ".join(best.get("tokens") or [])
            out.append(f"  - best window (E={best['emotional']} C={best['coherence']}): “{text}”")
        out.append("")
    return "\n".join(out).rstrip()


META_ROUND1_SCHEMA = {
    "type": "object",
    "properties": {
        "epochs_to_read": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "maxItems": MAX_LOGS_PER_ROUND,
            "description": "Past epoch numbers whose full meta-log you want to read.",
        },
    },
    "required": ["epochs_to_read"],
    "additionalProperties": False,
}

META_ROUND2_SCHEMA = {
    "type": "object",
    "properties": {
        "epochs_to_read": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "maxItems": MAX_LOGS_PER_ROUND,
        },
        "poems_to_read": {
            "type": "array",
            "maxItems": MAX_POEMS_TO_SAMPLE,
            "items": {
                "type": "object",
                "properties": {
                    "flask": {"type": "string"},
                    "gen": {"type": "integer", "minimum": 1},
                    "worm": {"type": "string"},
                },
                "required": ["flask", "gen", "worm"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["epochs_to_read", "poems_to_read"],
    "additionalProperties": False,
}


def _meta_render_logs(meta_root: Path, epochs: list[int]) -> str:
    out = []
    for ep in sorted(set(epochs)):
        t = _read_meta_log(meta_root, ep)
        if t is None:
            out.append(f"## epoch {ep} — (rested, no log)\n")
        else:
            out.append(f"## epoch {ep}\n\n{t}\n")
    return "\n".join(out)


def _meta_render_poems(generations_root: Path, picks: list[dict]) -> str:
    out = []
    for p in picks:
        try:
            flask = str(p["flask"])
            gen = int(p["gen"])
            worm = str(p["worm"])
        except (KeyError, ValueError, TypeError):
            continue
        rendered = _sample_window_from_gen(generations_root / flask, gen, worm)
        if rendered:
            out.append(f"- {flask} · {rendered}")
    return "\n".join(out)


def maybe_write_meta_log(flasks, generation_num: int, keepalive=None) -> Path | None:
    """Three-call meta-gardener. Observes all flasks for one epoch, writes
    a short shared log (or PASSes). Returns the written log path or None."""
    if os.environ.get("WORMLET_GARDENER", "1") == "0":
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    if not flasks:
        return None

    generations_root = V6_ROOT / "data" / "generations"
    meta_root = _meta_log_root(generations_root)

    flask_histories = {f.name: list(f.state.best_score_history) if f.state else []
                       for f in flasks}
    catalog = _meta_catalog(meta_root, flask_histories, generation_num)
    fragments = _format_all_flasks_fragments(flasks)
    briefing = _briefing()

    try:
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"[META-GARDENER] client init failed: {e}", flush=True)
        return None

    sys_blocks = [
        {"type": "text", "text": briefing, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": META_GARDENER_TONE},
    ]

    # Round 1
    r1_picks: list[int] = []
    r1_rendered = "(no prior epochs)"
    if generation_num > 1:
        try:
            if keepalive: keepalive()
            user = (
                f"Catalog of prior epochs (current epoch = {generation_num}):\n\n{catalog}\n\n"
                + ROUND1_INSTRUCTIONS.format(max_logs=MAX_LOGS_PER_ROUND).replace("logs", "meta-logs")
            )
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_SELECTION_TOKENS,
                thinking={"type": "adaptive"},
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": META_ROUND1_SCHEMA}},
                system=sys_blocks,
                messages=[{"role": "user", "content": user}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), "{}")
            r1_picks = json.loads(text).get("epochs_to_read", [])[:MAX_LOGS_PER_ROUND]
            r1_rendered = _meta_render_logs(meta_root, r1_picks)
        except Exception as e:
            print(f"[META-GARDENER] round 1 failed: {e}", flush=True)

    # Round 2
    all_log_picks = list(r1_picks)
    poem_picks: list[dict] = []
    if generation_num > 1:
        try:
            if keepalive: keepalive()
            user = (
                f"Catalog (for reference):\n\n{catalog}\n\n"
                f"Meta-logs from round 1:\n\n{r1_rendered}\n\n"
                + ROUND2_INSTRUCTIONS.format(max_logs=MAX_LOGS_PER_ROUND, max_poems=MAX_POEMS_TO_SAMPLE)
            )
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_SELECTION_TOKENS,
                thinking={"type": "adaptive"},
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": META_ROUND2_SCHEMA}},
                system=sys_blocks,
                messages=[{"role": "user", "content": user}],
            )
            text = next((b.text for b in resp.content if b.type == "text"), "{}")
            out = json.loads(text)
            all_log_picks = list(set(r1_picks) | set(out.get("epochs_to_read", [])[:MAX_LOGS_PER_ROUND]))
            poem_picks = out.get("poems_to_read", [])[:MAX_POEMS_TO_SAMPLE]
        except Exception as e:
            print(f"[META-GARDENER] round 2 failed: {e}", flush=True)

    all_logs_rendered = _meta_render_logs(meta_root, all_log_picks) if all_log_picks else ""
    poems_rendered = _meta_render_poems(generations_root, poem_picks) if poem_picks else ""

    # Round 3: write the log
    try:
        if keepalive: keepalive()
        user = (
            f"You are writing the meta-gardener's log for epoch {generation_num}.\n\n"
            f"Meta-logs you asked for:\n\n{all_logs_rendered or '(none)'}\n\n"
            f"Sampled poem windows:\n\n{poems_rendered or '(none)'}\n\n"
            f"Top fragments from THIS epoch (one section per flask):\n\n{fragments or '(none yet)'}\n\n"
            + ROUND3_INSTRUCTIONS
        )
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_LOG_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            system=sys_blocks,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "").strip()
    except Exception as e:
        print(f"[META-GARDENER] round 3 failed: {e}", flush=True)
        return None

    if not text:
        return None
    epoch_dir = meta_root / f"gen-{generation_num:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    if text.strip().upper() == "PASS" or text.strip().split()[0:1] == ["PASS"]:
        (epoch_dir / "gardeners_log.skipped").write_text(
            "the meta-gardener chose not to write this epoch.\n"
        )
        return None
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = f"# epoch {generation_num} — {ts}\n\n"
    log_path = epoch_dir / "gardeners_log.md"
    log_path.write_text(header + text + "\n")
    return log_path
