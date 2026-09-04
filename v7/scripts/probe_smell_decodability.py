#!/usr/bin/env python3
"""Is noun-ness even PRESENT in the smell the worm receives?

Written 2026-09-05, after the `all` vs `chemo` control A/B (1,071 and 1,063
generations) showed that noun SHARE never moved off the corpus base rate in
either arm, on a noise-free deterministic objective. Arm A learned to eat 6x
more words and stayed at 23.1% nouns; the corpus opening passage is 22.6%
nouns. The worms got better at eating and never got selective.

Before designing another evolutionary A/B it is worth asking the cheaper
question. Evolution can only find selectivity if the information is in the
channels to begin with. The worm's entire knowledge of a word is 12 floats:

    word -> 12-dim vector -> PC_NEURON_PAIRS[i] -> amphid neuron pair i

and per `sim/chemosensory_mapping.compute_pca_activation`, the word's identity
sets only the MAGNITUDE of each channel. The left/right split that actually
steers the body is `direction_factor`, which is pure geometry — where the word
is relative to the nose — and is identical for a noun and a preposition sitting
in the same place. So the only way to steer toward nouns is for the 12
magnitudes to differ between nouns and non-nouns.

This script measures whether they do. No simulation, no evolution, no judge:
just "train the best classifier you can on the 12 channels and see if it can
tell a noun from a non-noun".

    AUC ~ 0.50  the channels carry no noun information. No search over the
                connectome can ever produce noun selectivity, because the
                signal does not reach the worm. The control result is
                explained and the architecture is the thing to change.
    AUC >> 0.50 the information IS there and evolution failed to use it.
                The bottleneck is downstream — the sensorimotor loop or the
                search — and an evolutionary A/B is worth running.

Deliberately dependency-free (numpy only): the venv has no sklearn and this
should stay runnable on the FreeBSD host. Logistic regression by gradient
descent, AUC by rank statistic, both cross-validated.

Run:
    ./.venv/bin/python scripts/probe_smell_decodability.py
    ./.venv/bin/python scripts/probe_smell_decodability.py --corpus beowulf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

V7 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V7))

from server import embedding, pos_scorers  # noqa: E402


# ----- metrics ----------------------------------------------------------

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney rank identity. Ties get
    average ranks, so a constant scorer gives exactly 0.5 rather than 1.0."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 12,
                 l2: float = 1e-2) -> np.ndarray:
    """Ridge-penalised logistic regression by IRLS (Newton). Converges in ~8
    iterations where plain gradient descent needed thousands — with 200 RBF
    features and six target tags, that was the difference between the probe
    taking minutes and taking an hour. Returns the weight vector for X with a
    bias column already appended by the caller."""
    n, d = X.shape
    w = np.zeros(d)
    ridge = l2 * np.eye(d)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        s = np.clip(p * (1.0 - p), 1e-6, None)
        # Newton step: (X' S X + lambda I) dw = X'(y - p) - lambda w
        H = (X * s[:, None]).T @ X / n + ridge
        g = X.T @ (y - p) / n - l2 * w
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def rbf_features(X: np.ndarray, centres: np.ndarray, gamma: float) -> np.ndarray:
    """Random-centre RBF expansion. Gives the linear model a fair shot at any
    NONLINEAR noun/non-noun boundary — a linear probe alone would understate
    what a 300-neuron network could in principle extract."""
    d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)


def cv_auc(X: np.ndarray, y: np.ndarray, folds: int, rng: np.random.Generator,
           n_rbf: int = 0) -> tuple[float, float]:
    """K-fold cross-validated AUC. Returns (mean, std)."""
    idx = rng.permutation(len(y))
    out = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te, assume_unique=False)
        Xtr, Xte = X[tr], X[te]
        if n_rbf:
            c = Xtr[rng.choice(len(Xtr), size=n_rbf, replace=False)]
            gamma = 1.0 / max(Xtr.shape[1] * Xtr.var(), 1e-9)
            Xtr = rbf_features(Xtr, c, gamma)
            Xte = rbf_features(Xte, c, gamma)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
        Xte = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
        w = fit_logistic(Xtr, y[tr])
        out.append(auc(y[te], Xte @ w))
    return float(np.mean(out)), float(np.std(out))


# ----- data -------------------------------------------------------------

def load_smell_table(corpus: str) -> dict[str, np.ndarray]:
    """word -> 12-dim smell vector, exactly the table the worm's nose reads.

    hamlet uses cache/corpus_umap.json (the `umap12` field is what
    server/embedding.py serves under WORMLET_ENCODER=umap); the other corpora
    have their own corpus_smell12_<name>.json.
    """
    if corpus == "hamlet":
        d = json.loads((V7 / "cache" / "corpus_umap.json").read_text())
        vecs = np.asarray(d["umap12"], dtype=np.float64)
    else:
        d = json.loads((V7 / "cache" / f"corpus_smell12_{corpus}.json").read_text())
        vecs = np.asarray(d["umap12" if "umap12" in d else "smell12"],
                          dtype=np.float64)
    return {w: vecs[i] for i, w in enumerate(d["words"])}


def load_tokens(corpus: str, passage: str) -> list[str]:
    """The EDIBLE token stream the worm actually crawls over, in order.

    get_sentences_with_flags drops the set-dressing (speaker names, ACT/SCENE
    cues) that is rendered but cannot be eaten or smelled — including it would
    put words in the sample that never reach the nose.
    """
    if corpus == "hamlet":
        from corpus.hamlet import get_sentences_with_flags
        sents, edible = get_sentences_with_flags(passage)
        return [w for s, ok in zip(sents, edible) if ok for w in s]
    from corpus import library
    return list(library.get_tokens(corpus))


def build_dataset(corpus: str, passage: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (X, tags, tokens) over the RUNNING TEXT, not the vocabulary.

    Two things here are load-bearing and were both wrong in the first draft of
    this script, in the direction that flatters the encoder:

    1. TOKENS, not types. score_nouns runs over the token stream, where
       function words ('the', 'and', 'of') dominate by frequency. Scoring one
       row per vocabulary entry weights a word used once the same as 'the'.
    2. tag_sequence, not tag_word. The objective tags IN CONTEXT. Tagging
       isolated words sends NLTK's perceptron to NOUN for almost anything —
       it reports Hamlet's vocabulary as 77% nouns where the running text is
       22.6%. Probing against the out-of-context label measures a category
       the fitness function never used.
    """
    table = load_smell_table(corpus)
    toks = load_tokens(corpus, passage)
    tags = pos_scorers.tag_sequence(toks)
    rows, keep_tags, keep_toks = [], [], []
    for tok, tag in zip(toks, tags):
        v = table.get(embedding._norm(tok))   # the nose's own lookup key
        if v is None:
            continue          # not in the smell table = the worm smells nothing
        rows.append(v)
        keep_tags.append(tag)
        keep_toks.append(tok)
    return np.asarray(rows), keep_tags, keep_toks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="hamlet")
    ap.add_argument("--passage", default="opening",
                    help="opening = what run-control.sh scrolls; act1 = live")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X, tags, toks = build_dataset(a.corpus, a.passage)

    print(f"corpus {a.corpus} / passage {a.passage}: {len(toks)} tokens "
          f"({len(set(toks))} distinct), {X.shape[1]} smell channels")
    print(f"channel range: {X.min():.3f}..{X.max():.3f}  "
          f"mean per-channel std {X.std(axis=0).mean():.3f}")
    print()

    # The headline question, plus the same probe for other POS classes as a
    # control: if NOUN is at chance but VERB is not, that is a fact about the
    # tag, not about the encoder.
    print(f"{'target':>8}  {'base rate':>9}  {'linear AUC':>16}  {'RBF AUC':>16}")
    for tag in ("NOUN", "VERB", "ADJ", "ADV", "DET", "."):
        y = np.array([1.0 if t == tag else 0.0 for t in tags])
        if y.sum() < 30 or (len(y) - y.sum()) < 30:
            continue
        lin, lin_sd = cv_auc(X, y, a.folds, np.random.default_rng(a.seed))
        nl, nl_sd = cv_auc(X, y, a.folds, np.random.default_rng(a.seed), n_rbf=200)
        print(f"{tag:>8}  {100 * y.mean():8.1f}%  "
              f"{lin:.3f} +/- {lin_sd:.3f}  {nl:.3f} +/- {nl_sd:.3f}")

    # What noun share would a worm achieve if it ate greedily by the best
    # available noun score? This converts AUC into the units the control A/B
    # reports, so the two are directly comparable.
    print()
    y = np.array([1.0 if t == "NOUN" else 0.0 for t in tags])
    # OUT-OF-FOLD scores. Fitting on all rows and then ranking those same rows
    # overstates the reachable share badly at this sample size (258 tokens in
    # the opening passage), and the reachable share is the number that gets
    # compared against the control A/B — so it has to be honest.
    s = np.empty(len(y))
    idx = np.random.default_rng(a.seed).permutation(len(y))
    for f in range(a.folds):
        te = idx[f::a.folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = fit_logistic(np.hstack([(X[tr] - mu) / sd, np.ones((len(tr), 1))]), y[tr])
        s[te] = np.hstack([(X[te] - mu) / sd, np.ones((len(te), 1))]) @ w
    base = 100 * y.mean()
    print(f"noun base rate (eat indiscriminately): {base:.1f}%")
    for frac in (0.5, 0.25, 0.1):
        k = max(1, int(len(s) * frac))
        top = np.argsort(-s)[:k]
        print(f"  eat only the top {100 * frac:4.0f}% smelliest words: "
              f"{100 * y[top].mean():.1f}% nouns "
              f"({100 * y[top].mean() - base:+.1f} pts vs base)")
    print()
    print("Reference: the control A/B measured 23.1% (arm A, gen 1047-1071)")
    print("and 23.8% (arm B) after ~1,070 generations of selection for nouns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
