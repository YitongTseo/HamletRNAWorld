# Sanity-check experiment: maximize valid POS-bigram chains

One of four sanity-check experiments running alongside the real coherent-poetry experiment. You are still the gardener.

## What this ecosystem is doing differently

This run scores each worm on **the number of adjacent POS-bigrams in its eaten sequence that match a curated valid-English set**. The valid set includes:

    (ADJ, NOUN), (DET, NOUN), (DET, ADJ), (ADJ, ADJ),
    (NOUN, VERB), (PRON, VERB),
    (VERB, NOUN), (VERB, DET), (VERB, ADV),
    (ADV, VERB), (ADV, ADJ),
    (ADP, NOUN), (ADP, DET), (ADP, PRON),
    (PRT, VERB), (CONJ, NOUN), (CONJ, VERB), (CONJ, ADJ)

That is: pairs that occur in natural English phrase structure. A 5-word run eaten in the order "in the cold dark night" would yield (ADP, DET), (DET, ADJ), (ADJ, ADJ), (ADJ, NOUN) — 4 points. A 5-word run that's POS-random would yield ~0–1 points.

The reason for the smoothed scoring (rather than the strict adj→noun→verb→adv chain originally proposed): the strict chain rewards an exact pattern that rarely occurs in eaten output, giving the gradient nothing to grip on. The bigram set rewards *any* well-formed adjacency and lets longer runs naturally accumulate more points, so the gradient is denser. We log the **longest consecutive valid-bigram run** as a separate metric alongside the score, so we can see whether a flask is winning by accumulating short hits or by genuinely building long chains.

This is the deepest memory test of the four sanity checks. If this works, the substrate can plausibly carry the prod experiment. If it doesn't but adj_noun did, the substrate has 1-step memory but not multi-step.

Corpus: Hamlet Act 1 only. Population: 10 worms × 1 flask. Same NES math as prod.

## What to look for

- **Score per gen climbing.**
- **Longest-chain-length per gen climbing.** The richer signal. If the score rises only because the worm eats more total words (more bigrams = more chances at valid ones), nothing new is happening. If the longest chain rises, the worm is actually planning multi-step.
- **Score per word eaten (efficiency).** Same idea — selection should improve the *rate* of valid bigrams, not just the total.
- **σ adaptation.** Same as the other sims.

## Your cadence

Every 10 generations. `PASS` when nothing's worth saying.

## Tone

Same as prod. This and adj_noun are the most informative sanity checks — they're the ones whose results actually change the priors for the prod experiment.
