# Generational Evolution — design spec

Status: draft, not yet implemented.
Owner: Yitong.
Last touched: 2026-05-26.

## What we're building

When all six worms finish eating one full pass of Shakespeare's *Hamlet*, the
generation ends. An LLM reads each worm's resulting poem, scores it, and the
top performers become the parents of the next generation. The new worms are
clones of the previous champions with directed mutation pressure (NES, not
random walk), so the population should evolve toward producing more
emotionally impactful and coherent poetry over time. Everything from each
generation — poems, scores, weights, seeds — gets committed to git for later
analysis.

## Generation lifecycle

```
gen N:                            gen N+1:
+--------------+   end-of-corpus  +--------------+
| 6 worms eat  | ───────────────► | LLM judge    |
| full Hamlet  |                  | scores poems |
+--------------+                  +------┬-------+
       ▲                                 │
       │                                 ▼
       │                          +--------------+
       │                          | NES update   |
       │                          | parent gets  |
       │                          | nudged       |
       │                          +------┬-------+
       │                                 │
       │      commit gen N to git, spawn │
       └─────── gen N+1 from new parent ◄┘
```

End-of-corpus detection: switch the food refill from "loop forever" to "fill
once". When all 6 worms' `food_remaining == 0` (or after a hard timeout, see
Open Q1), the sim pauses, kicks off scoring, then resets.

## Corpus change

Today `corpus.hamlet.get_sentences(passage="opening")` returns 52 sentences /
~340 tokens. We add `passage="full"` that tokenizes the entire
`hamlet_gutenberg.txt` (~32K tokens). One generation = one full pass.

Empirical eating rate is ~1.5 words/sec/worm → ~6 hours per generation →
~4 generations/day. That tempo is also what makes the LLM cost work; see
budget below.

## Punctuation cleanup

Applied when we write `poem_clean.txt` (not to the live `poem.txt`, so we can
re-derive cleaned versions if rules change later).

Rule: collapse runs of consecutive punctuation tokens to a single mark
(keeping the first), preserve trailing punctuation.

| Input                          | Output                  |
| ------------------------------ | ----------------------- |
| `he . . .`                     | `he .`                  |
| `I 'tis ! ? . .`               | `I 'tis !`              |
| `to be , or not to be . . . .` | `to be , or not to be .`|

Punctuation set: any token in `{. , ! ? ; : — - ' " ( )}`. (Confirm full set
against the tokenizer in `corpus/hamlet.py` before implementation.)

## LLM judge

**Model:** Claude Haiku 4.5. Sonnet is overkill for "rate this fragment
1-100" and 3-5× the cost.

**Windowing:** 15-token windows, **non-overlapping** (stride 15). For a 32K
poem that's ~2100 windows/worm. Original proposal was stride 3; we dropped
it because each word would be evaluated in 5 overlapping windows for ~10%
more signal at 5× the cost — Claude already integrates surrounding context.

**Prompt structure (one call per worm, prompt-cached):**

```
[system, cached]
You are a careful judge of poetry. For each 15-token window I number
below, output one line of the form:
  <window_idx>,<emotional>,<coherence>
where emotional and coherence are integers 1-100. Emotional measures
artistic / emotional / poetic impact. Coherence measures whether the line reads
as language rather than noise. Output ONLY the score lines, nothing
else.


Hey claude to be clear ... this example you provided below is the unadultrated Hamlet... we will be feeding
the worms' poems right? so they should be all cut up... just making sure we're on the same page with that!
[user]
0: Who's there ? Nay , answer me . Stand , and unfold
1: yourself . Long live the King ! Bernardo ? He . You come
2: most carefully upon your hour . 'Tis now struck twelve . Get thee
...
2099: <last 15 tokens>
```

... should be something closer to: 

[user]
0: night , come us this the beating heart . of him come speak
1: Touching may let are What And that's t'illume
2: come to . , answer ? and Long , answer ? and
...
2099: <last 15 tokens>

**Why this is cheap:** the rubric (~300 tokens) is identical across all
calls → prompt cache hit. The unique input is the windowed poem
(~45K tokens including line numbers). Output is `2100 × ~8 tokens ≈ 17K`
tokens of compact CSV.

**Back-of-envelope per generation (Haiku 4.5):**

| Component          | Cost            |
| ------------------ | --------------- |
| Input, 6 worms     | ~$0.27          |
| Output, 6 worms    | ~$0.10          |
| Cached rubric      | negligible      |
| **Per generation** | **~$0.40**      |
| **Per month** (4/day) | **~$48**     |

If $48/mo is too steep we have three knobs: bigger stride (stride 30 halves
output cost), batch scoring (group multiple windows per output line), or
sample (score every Nth window only). All easy to dial later.

^^ actually yes maybe we should sample one of every 10 poems at random! to save a dime... it'll be 
too much data anyways, why not cut costs a little? 

## Fitness aggregation

For each window with scores `(E, C)` both in `[0, 100]`:

```
window_fitness = 1.5 · (E/100)^γ + 1.0 · (C/100)^γ
```

with `γ = 2.5`. The exponent makes the top windows dominate (a 90 contributes
~10× more than a 50). The 1.5× weight on emotional matches your stated
preference. Worm fitness = sum over all its windows.

Selection: rank-based, not raw-score-based. Rank worms 0..5 within a
generation; weights for NES update are `rank_weight = [4, 2, 1, 0, -1, -2]`
or similar (zero-sum, top-heavy). Rank protects against runaway scores from
the LLM.

## NES-style breeding (the "directional drift")

Each worm's weights are a flat vector `θ ∈ ℝ^d`. For each generation:

1. **Parent θ₀** = last generation's parent (or the best worm of gen 0).
2. **Spawn 6 children:** `θ_i = θ₀ + σ · ε_i` where `ε_i ~ N(0, I_d)`.
   Keep one elite: `θ_0 = θ₀` (`ε_0 = 0`).
3. **Score** all 6 children via LLM judge → `f_i`.
4. **Update parent for next generation:**

   ```
   θ₀' = θ₀ + (η / (N · σ)) · Σ_i rank_weight_i · ε_i
   ```

   where `N = 6`, `η ≈ 0.1` (learning rate), `σ ≈ 0.02` initially.

5. **Adapt σ:** if best-of-gen > best-of-prev-gen, `σ ← 0.8 · σ`. Else
   `σ ← 1.3 · σ`. Clip to `[0.005, 0.1]`.

This is the OpenAI Evolution Strategies (Salimans et al. 2017) update, which
performs surprisingly well with small populations because it estimates a
fitness gradient from the noise+score correlation rather than relying on
random walk.

Why this beats vanilla cloning+noise: with 6 worms you get directional
information for free — the ε_i vectors that scored high tell you which way
in weight-space to nudge the parent. Pure cloning+noise throws this away.

## Data layout

```
data/generations/gen-0001/
  metadata.json            # corpus_hash, judge_model, sigma, lr, parent_gen, started_at, ended_at
  Alice/
    poem_raw.txt           # what the worm ate, untouched
    poem_clean.txt         # after punctuation cleanup
    weights.json           # this worm's final weights
    seed.txt
    scores.jsonl           # one line per window: {idx, tokens, emotional, coherence}
    fitness.json           # aggregate fitness, rank
  Bob/ … Frank/
  selection.json           # chosen parent for next gen + NES update vector
```

Per-generation flow:
1. Generation ends → freeze sim.
2. Write all the above to `data/generations/gen-NNNN/`.
3. `git add data/generations/gen-NNNN && git commit -m "gen NNNN: …"`.
4. `git push` to a private remote (TBD: GitHub private repo or local bare repo).
5. After successful push, delete `poem_raw.txt` locally (largest file); keep
   the next parent's `weights.json` cached in `data/current/parent.json`.
6. Spawn gen NNNN+1.

Total per-gen disk usage: ~200KB raw + ~150KB clean + ~50KB scores +
weights, × 6 worms ≈ ~2 MB/gen. After cleanup, ~1 MB/gen retained in git.

## Open questions

1. **Hard timeout per generation.** If a worm gets stuck and never finishes
   the corpus, what's the max wall-clock per generation? Suggest 12h then
   force-end and score what we have.
   ^ to be clear, the worms are not finishing the corpus... we are just scrolling the corpus past the worms...
   ^ so hopefully all of the corpuses will scroll past the worms at basically the same rate and the simulations will end for all of the worms at the same time... can you double check this in the code? the only time a worm will get stuck is if the whole simulation itself gits stuck.
2. **Git remote.** Private GitHub repo, or a bare repo on the same server, or
   both? Affects the push step.
   ^I'd like it to go directly into the HamletRNAWorld Repo... i think there should already be a git repository setup for this directory... 
3. **Cold start.** Generation 0's "parent" weights = current per-worm
   weights, or a single chosen seed worm? I lean toward: gen 0 = today's six
   worms run to completion, gen 1's parent = winner of gen 0.
   ^ I think cold start is to restart the whole simulation with just the default weights of the worms... 
4. **Anthropic key.** Where does `ANTHROPIC_API_KEY` live? Suggest add to
   `/home/web/.wormlet.env` (already used by the systemd unit).
   ^ can add anywhere as long as it's safely gitignored... 
5. **Watchdog interaction.** Scoring will pause sim for several seconds (LLM
   round-trip). The tick watchdog will fire if pause > 20s. Need to either
   (a) increase the threshold during scoring, (b) update `_LAST_TICK_AT`
   from the scorer to keep watchdog quiet, or (c) move scoring to a
   background thread and let sim keep running on the *new* generation while
   scoring the previous one. (c) is most elegant.
   ^ i'd like the simulations to all stop once a few seconds after the final closing words of the play float above the screen.
   The ideally i'd like an indicator to show up on the screen with a progress bar as the scoring takes place. 
   Only after the scoring is completed i'd like the worms to get back up and start twisting and twirling...
   It'd probalby be best to freeze the watchdog as the worms are frozen.
6. **Tokenizer alignment.** The LLM sees space-separated tokens; the worm
   eats whatever `corpus/hamlet.py` produces. Confirm 1:1 correspondence
   before windowing, or scores will be off by ~1 token at a time.
   ^ I don't understand this? 

## Out of scope (for this spec)

- A web UI for browsing generations / scores.
- Cross-generation analysis dashboards.
- Anything sexual/recombination — staying on asexual self-fertilization as
  agreed.
- Compute-bound mini-evals (running children in fast-forward sim before
  the LLM judges) — possible future optimization.

## Plan of attack (when this spec is approved)

1. Add `passage="full"` to `corpus/hamlet.py` and a way to detect "corpus
   exhausted" in the world.
2. Write `server/poem_clean.py` (pure function, easy to unit-test).
3. Write `server/judge.py` — Claude API call with prompt caching + retry.
4. Write `server/evolution.py` — fitness aggregation + NES update.
5. Write `server/generations.py` — orchestrates the end-of-gen handoff,
   git commit, spawn-next.
6. Wire into `server/app.py:sim_loop` (or a sibling task).
7. Tests: deterministic fitness on a canned poem; NES math correctness;
   punctuation cleanup edge cases.

Each step is a few hundred LOC and independently testable. Implementing in
the order above keeps each change reviewable.

I'd also like to make this not just for 6 worm population. FI'd like to be for 4 groups of worms (since there are four cores on this server which I hope to run in parallel) and for each 4 groups of worms to be an independent NEC experiment. I'd like there to be 10 worms rather than 6 in each of the 4 groups. To be clear that means each independent core should produce its own gradient estimation and evolve completely separately from the others... like 4 silos if that makes sense... but i'd like the wordswordsworms.org display to show the live actions of all four cores is that possible?

