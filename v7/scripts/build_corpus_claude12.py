"""Build a 12-dim smell-space cache for a non-Hamlet corpus by having Claude
rate every vocabulary word on 11 named semantic axes (0..1). PC11 stays the
POS-grammar channel at runtime, exactly as for Hamlet; the 12th column here
is a zero filler the loader discards.

Why Claude ratings and not embeddings: the Anthropic API has no embeddings
endpoint, the original nomic+UMAP toolchain (torch/sentence-transformers/
umap-learn) has no FreeBSD path, and named axes make the smell space
interpretable — channel 3 of a worm's nose literally means "body/flesh".
Hamlet keeps its original UMAP space untouched; this deviation is recorded
in the fidelity ledger (wormjail/BEHAVIOUR-LOG.md).

Output: cache/corpus_smell12_<corpus>.json, shaped like corpus_umap.json
(words / umap12 / emotion_keys / emotions) so the embedding loader needs
only a path parameterisation, plus axes/model provenance.

Run (host, one-off; key from env, never argv):
    doas sh -c '. /etc/homelab/secrets.env; export ANTHROPIC_API_KEY; \
        python3 scripts/build_corpus_claude12.py laozi'

Anthropic structured-output schemas reject maxItems/minimum/etc (CLAUDE.md
rule 6): bounds are enforced in the prompt and clamped client-side, and a
row-count mismatch retries the batch.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus import library  # noqa: E402
# tokenisation happens in the corpus loaders now
from corpus.nrc_emotions import get_emotion_vector  # noqa: E402

V7_ROOT = Path(__file__).resolve().parent.parent
MODEL = "claude-opus-5"
BATCH = 60
API_URL = "https://api.anthropic.com/v1/messages"

# 11 semantic axes (PC0..PC10). Chosen to discriminate within BOTH the Tao
# Teh King and Beowulf; the order here is the channel order for the life of
# the corpus cache — never reorder an existing cache's axes.
AXES = [
    "dread — darkness, fear, death, the grave",
    "warmth — affection, kinship, comfort, generosity",
    "nature — earth, water, sky, weather, plants, beasts",
    "body — flesh, blood, hands, breath, physical sensation",
    "motion — journeys, speed, crossing, pursuit",
    "power — command, kingship, strength, dominion",
    "sacred — mystery, the divine, fate, the nameless",
    "speech — song, telling, naming, counsel, boast",
    "time — age, memory, endings, the ancient",
    "conflict — battle, struggle, opposition, weapons",
    "stillness — quiet, emptiness, rest, yielding",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
    "required": ["ratings"],
    "additionalProperties": False,
}

SYSTEM = (
    "You rate words for a simulated organism's chemosensory system. For each "
    "word, output 11 numbers in [0,1]: how strongly the word evokes each axis, "
    "in this exact order:\n"
    + "\n".join(f"{i}. {a}" for i, a in enumerate(AXES))
    + "\nJudge the word's typical sense in the source text named by the user. "
    "0 means no association, 1 means the axis is the word's core meaning. "
    "Most words score low on most axes. Output exactly one row of exactly 11 "
    "numbers per input word, in input order, no more and no fewer."
)


def _post(payload: dict, key: str) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(5):
        req = urllib.request.Request(API_URL, data=body, headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "server-side-fallback-2026-07-01",
            "content-type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 529) and attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"API {e.code}: {e.read()[:500]}") from e
    raise RuntimeError("unreachable")


def rate_batch(words: list[str], corpus_title: str, key: str) -> list[list[float]]:
    prompt = (f"Source text: {corpus_title}.\nWords ({len(words)}):\n"
              + "\n".join(f"{i}: {w}" for i, w in enumerate(words)))
    for attempt in range(3):
        resp = _post({
            "model": MODEL,
            "max_tokens": 16000,
            "output_config": {"effort": "low",
                              "format": {"type": "json_schema", "schema": SCHEMA}},
            "fallbacks": "default",
            "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }, key)
        if resp.get("stop_reason") == "refusal":
            raise RuntimeError(f"refusal on batch starting {words[0]!r}: "
                               f"{resp.get('stop_details')}")
        text = next(b["text"] for b in resp["content"] if b["type"] == "text")
        rows = json.loads(text)["ratings"]
        if len(rows) == len(words) and all(len(r) == 11 for r in rows):
            # Clamp client-side — the schema can't carry numeric bounds.
            return [[min(1.0, max(0.0, float(v))) for v in r] for r in rows]
        print(f"  row mismatch ({len(rows)}/{len(words)}), retry {attempt + 1}")
    raise RuntimeError(f"batch failed 3x at {words[0]!r}")


def main() -> None:
    corpus = sys.argv[1] if len(sys.argv) > 1 else ""
    if corpus not in ("laozi", "beowulf"):
        sys.exit("usage: build_corpus_claude12.py {laozi|beowulf} "
                 "(hamlet keeps its original UMAP space)")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY not in env")

    # Lines arrive as token lists (the corpus contract, matching hamlet).
    lines, _flags = library.get_sentences_with_flags(corpus, "full")
    vocab = sorted({t.lower() for ln in lines for t in ln
                    if any(c.isalpha() for c in t)})
    title = library.TITLES[corpus]
    print(f"{corpus}: {len(vocab)} word types, {len(AXES)} axes, model {MODEL}")

    rows: list[list[float]] = []
    for i in range(0, len(vocab), BATCH):
        chunk = vocab[i:i + BATCH]
        rows.extend(r + [0.0] for r in rate_batch(chunk, title, key))
        print(f"  {min(i + BATCH, len(vocab))}/{len(vocab)}")

    emotions = {w: get_emotion_vector(w) for w in vocab}
    ekeys = sorted(next(iter(emotions.values())).keys()) if emotions else []
    out = {
        "n_words": len(vocab), "n_dims": 12,
        "model": MODEL, "axes": AXES,
        "words": vocab,
        "umap12": rows,  # key kept for loader compatibility; col 12 is filler
        "emotion_keys": ekeys,
        "emotions": {w: [v[k] for k in ekeys] for w, v in emotions.items()},
    }
    dest = V7_ROOT / "cache" / f"corpus_smell12_{corpus}.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
