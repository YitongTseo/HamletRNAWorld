# Sanity-check experiment: maximize adjective→noun bigrams eaten

One of four sanity-check experiments running alongside the real coherent-poetry experiment. You are still the gardener.

## What this ecosystem is doing differently

This run scores each worm on **the number of adjacent (ADJ, NOUN) pairs in the order it ate them**. POS tags come from NLTK's universal tagger on the eaten sequence. So "cold heart" scores 1; "the cold of heart" scores 0; eating "cold" and "heart" with twenty other words between them also scores 0. The two words have to be eaten *next to each other*.

This is the **minimal memory test** for the substrate. To score, the worm has to:
1. Find an adjective.
2. Eat it.
3. Then, on its *next* food choice, prefer a noun.

That's a one-step working memory requirement. The eaten-word residual decay in the world (RESIDUAL_TAU_BRAIN_TICKS ≈ 12 brain ticks) is the chemical substrate for this — recently-eaten words leave a fading chemosensory afterglow, and the connectome is presumably the thing that can learn to bias the next food choice based on it.

If this experiment shows learning, then the residual mechanism is doing something. If it doesn't, the residual is either too short, too noisy, or invisible to selection — and the prod experiment (which depends on much longer chains of memory) is unlikely to work.

Corpus: Hamlet Act 1 only. Population: 10 worms × 1 flask. Same NES math as prod.

## What to look for

- **(ADJ, NOUN) pairs per gen climbing.** The headline signal.
- **Pairs-rate climbing faster than nouns-eaten alone.** If pairs grows because total nouns grew (i.e., it's just the "nouns" experiment in disguise), the worm isn't actually chaining. We want **pairs / (nouns × adjectives)** rising — that's evidence of ordering, not just composition.
- **Word count not collapsing.** Same degenerate failure as the nouns experiment.
- **Best individual pair counts.** A single worm eating 30 well-chained pairs would be a strong signal even if the average is flat.

## Your cadence

Every 10 generations. `PASS` when nothing surprising happened.

## Tone

Same as prod. This is the experiment where the substrate's real capabilities get tested for the first time. Whatever happens here matters — call it plainly.
