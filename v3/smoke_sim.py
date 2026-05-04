"""Headless smoke test for the v3 letter-bead simulation core."""
from __future__ import annotations

import argparse
import time

import numpy as np

from corpus.hamlet import get_raw_and_tokens
from embeddings import load as load_embedder
from sim.world import World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passage", default="opening")
    ap.add_argument("--n-sentences", type=int, default=5)
    ap.add_argument("--embedder", default="random")
    ap.add_argument("--steps", type=int, default=60_000)
    args = ap.parse_args()

    sentences_raw, tokens = get_raw_and_tokens(args.passage, args.n_sentences)
    embedder = load_embedder(args.embedder)
    print(f"Embedder: {embedder.name}, dim={embedder.dim}")
    for raw in sentences_raw:
        print(f"  {raw}")

    w = World(
        sentences_raw=sentences_raw,
        tokens_per_sentence=tokens,
        embedder=embedder,
    )
    s = w.state
    print(
        f"\nletter-beads N={s.n}, strands={s.n_chains}, "
        f"tokens={s.n_tokens}, reactive tokens={int(s.token_reactive.sum())}"
    )

    total_steps = args.steps
    log_every = max(1, total_steps // 12)
    t0 = time.time()
    for _ in range(0, total_steps, log_every):
        w.step(log_every)
        elapsed = time.time() - t0
        x_lead = s.pos[0, 0]
        x_tail = s.pos[-1, 0]
        n_bp = len(s.base_pairs)
        print(
            f"step {s.step_count:6d} | bp={n_bp:3d} | lead.x={x_lead:+.2f} "
            f"tail.x={x_tail:+.2f} | t={elapsed:.2f}s"
        )

    print("\nFinal pair-bonds (most-opposite token pairs first):")
    if len(s.base_pairs) == 0:
        print("  (none)")
    else:
        pairs = s.base_pairs.copy()
        ti = s.token_id[pairs[:, 0]]
        tj = s.token_id[pairs[:, 1]]
        affs = s.token_affinity[ti, tj]
        order = np.argsort(-affs)
        for k in order:
            i, j = int(pairs[k][0]), int(pairs[k][1])
            tok_i = s.tokens[int(ti[k])]
            tok_j = s.tokens[int(tj[k])]
            print(
                f"  {s.chars[i]!r}({i:3d}, '{tok_i}') -- "
                f"{s.chars[j]!r}({j:3d}, '{tok_j}')   affinity={affs[k]:.3f}"
            )


if __name__ == "__main__":
    main()
