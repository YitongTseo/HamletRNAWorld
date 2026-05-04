"""Entry point: load Hamlet text + an embedder, build the world, launch viz."""
from __future__ import annotations

import argparse

from corpus.hamlet import get_raw_and_tokens
from embeddings import load as load_embedder
from render.app import run
from sim.world import World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passage", default="opening",
                    help="which Hamlet passage to extrude (opening|soliloquy)")
    ap.add_argument("--n-sentences", type=int, default=50,
                    help="cap on sentences from the passage")
    ap.add_argument("--embedder", default="distilbert",
                    help="which embedder to use (distilbert|random)")
    ap.add_argument("--font", default=None,
                    help="path to a .ttf/.otf monospace font (auto-detect if omitted)")
    args = ap.parse_args()

    sentences_raw, tokens_per_sentence = get_raw_and_tokens(args.passage, args.n_sentences)
    print(f"Loaded {len(sentences_raw)} sentences "
          f"({sum(len(t) for t in tokens_per_sentence)} tokens, "
          f"{sum(len(s) for s in sentences_raw)} characters total)")
    for i, raw in enumerate(sentences_raw):
        print(f"  [{i}] {raw}")

    print(f"Loading embedder: {args.embedder} ...")
    embedder = load_embedder(args.embedder)
    print(f"  embedder dim = {embedder.dim}")

    world = World(
        sentences_raw=sentences_raw,
        tokens_per_sentence=tokens_per_sentence,
        embedder=embedder,
    )
    s = world.state
    print(f"Built world: {s.n} letter-beads in {s.n_chains} strands "
          f"({s.n_tokens} tokens, "
          f"{int(s.token_reactive.sum())} reactive).")
    run(world, font_path=args.font)


if __name__ == "__main__":
    main()
