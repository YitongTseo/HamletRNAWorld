"""Word → chemosensory neuron firing, driven by a 12-dim PCA over the
DistilBERT/nomic embedding of the full Hamlet corpus.

The mapping is one principal component per amphid neuron pair. The pair
ordering below is canonical and load-bearing: PC index i drives
PC_NEURON_PAIRS[i] and nothing else. Don't reorder casually — the on-disk
PCA artifact (data/corpus_pca.json, built by scripts/build_corpus_pca.py)
is computed against this ordering and the residual / smell payloads in
sim.world depend on it.

Biology note: all twelve are real C. elegans amphid sensilla, bilateral
pairs sitting in the worm's nose. They're the chemoreceptors the actual
animal uses to taste salts, food odors, repellents, CO2, temperature, etc.
Each L/R member of a pair is normally activated together; the spatial L vs R
asymmetry that drives steering comes from `direction_factor` in
compute_pca_activation below, not from any difference in baseline activation.

Earlier versions (v6.0 and the v5 lineage) mapped 8 NRC emotion categories
to these neurons via a hand-designed `EMOTION_TO_NEURONS` table. That
approach left ~85% of Hamlet's vocabulary silent (only emotion-bearing
words fired anything). Replaced in v6.1 by the PCA-of-corpus approach
below, which gives every word a vector. The emotion table and the
nrc_emotions corpus are gone — see git history if you want to revive them.
"""

from __future__ import annotations


# Canonical ordering of 12 chemosensory neuron pairs in the C. elegans
# amphid sensilla. PC index i drives PC_NEURON_PAIRS[i].
PC_NEURON_PAIRS: list[tuple[str, str]] = [
    ("ASEL", "ASER"),  # PC0: salt / primary valence
    ("AWAL", "AWAR"),  # PC1: food odor
    ("AWBL", "AWBR"),  # PC2: approach
    ("AWCL", "AWCR"),  # PC3: CO2 / safety
    ("ADLL", "ADLR"),  # PC4: polymodal
    ("ADFL", "ADFR"),  # PC5: food-chemo
    ("ASHL", "ASHR"),  # PC6: pain / avoid
    ("ASIL", "ASIR"),  # PC7: hunger / arousal
    ("ASJL", "ASJR"),  # PC8: taste / novel
    ("ASKL", "ASKR"),  # PC9: protect
    ("ASGL", "ASGR"),  # PC10: integrate
    ("AFDL", "AFDR"),  # PC11: thermosensory
]


def compute_pca_activation(
    pca_vec,                       # iterable of 12 floats in [0, 1]
    direction_factor: float = 0.5,
    intensity: float = 1.0,        # multiplier (e.g. distance_factor)
) -> dict[str, float]:
    """For a single word's PCA vector, return per-neuron firing levels.

    L_i = pca_vec[i] * direction_factor * intensity
    R_i = pca_vec[i] * (1 - direction_factor) * intensity

    Used both for in-range smells (direction-aware) and for the residual
    afterglow of eaten words (direction_factor=0.5 so L and R get the same
    contribution — the word is already inside the worm, no direction of
    arrival anymore)."""
    out: dict[str, float] = {}
    dirL = direction_factor
    dirR = 1.0 - direction_factor
    for i, (L, R) in enumerate(PC_NEURON_PAIRS):
        if i >= len(pca_vec):
            break
        v = pca_vec[i] * intensity
        out[L] = v * dirL
        out[R] = v * dirR
    return out
