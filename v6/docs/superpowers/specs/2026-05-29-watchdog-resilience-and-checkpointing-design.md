# Watchdog resilience + mid-generation checkpointing

**Date:** 2026-05-29
**Status:** implemented (pending coordinated deploy — see Deploy notes)

## Problem

On 2026-05-29 ~01:26 UTC the `wormlet-app` process was restarted by the
external systemd healthcheck watchdog after 3 consecutive 5s `curl /` timeouts
(transient event-loop starvation under the 32-worm load — not a crash, OOM, or
exception). The restart ran `_truncate_partial_gen_poems()`, discarding the
in-progress **generation 0** and restarting it from word 0.

Root-cause findings:
- Not a code crash: no reboot (host up 3 days), no kernel OOM, no Python traceback.
- The watchdog had restarted the app **5×/3 days** — a recurring false-positive.
- Healthcheck failures were transient and uncorrelated with rollovers.
- A latent risk: the external healthcheck is **not rollover-aware**. Now that the
  judge makes real API calls (post `ANTHROPIC_API_KEY` fix), rollovers take real
  time and could be falsely killed mid-rollover.
- Progress loss is **by design**: there is no mid-generation checkpoint, so any
  restart mid-gen restarts the generation. For gen 0 that means starting over.

## Changes

### 1. Rollover-aware watchdog
- App writes `/run/wormlet/rollover.lock` (systemd `RuntimeDirectory=wormlet`,
  tmpfs) for the duration of `_trigger_generation_rollover()`.
- `wormlet-healthcheck` skips probing/counting/restarting while the lock exists.
- tmpfs ⇒ the lock can never survive a process death and wedge the watchdog off.

### 2. Lenient healthcheck
- `wormlet-healthcheck`: probe `/healthz` (cheap JSON) instead of `/`;
  `TIMEOUT 5→10`, `THRESHOLD 3→20` ⇒ ~20 min of *sustained* unresponsiveness
  before a restart, vs ~3 min before.

### 3. Lightweight mid-generation checkpoint (Approach C)
Chosen over (A) deterministic replay — too slow to replay multi-hour gens on
boot — and (B) full pickle snapshot — heavy and fragile across code deploys.

- `TextScroller.snapshot()/restore()`: serialize corpus progress only
  (`_sent_idx`, `_line_id`, `_elapsed`, `_next_spawn`, `_dead`, `_active`).
- `sim_loop` writes `<worm>/checkpoint.json` every `CHECKPOINT_INTERVAL_S`
  (default 60s, env `WORMLET_CHECKPOINT_INTERVAL_S`), atomic temp+rename.
- Startup: per worm, if a checkpoint's `generation` matches the flask's current
  generation → restore scroll position and **keep `poem.txt`**; otherwise
  truncate the poem and start the generation fresh (the prior behavior — the
  safe fallback for missing/corrupt/stale-gen checkpoints).
- Rollover (`_respawn_flask`) deletes the checkpoint so the next generation
  can't resume from a stale one.
- **Accepted trade-off:** worm body + brain transients reset (respawn at center,
  re-settle in ~1s). Corpus position, eaten words, on-screen words, and the full
  poem are preserved. Determinism (`test_determinism.py`) is unaffected — an
  uninterrupted run is byte-identical to before.

### 4. X-ray labels off by default
- `xrayLabelsVisible` default `true→false` in `network-panel.js`, `magnifier.js`,
  `index.js`. The x-ray/live-body view + magnifier lens start uncluttered;
  neuron names remain available in the connectome graph view (`x`) and via the
  `l` toggle. No persistence, so the default always applies.

## Tests
- `tests/test_checkpoint.py`: scroller round-trip exact; restored scroller
  continues identically; eaten words never respawn.
- Decision-logic cases (keep / stale-truncate / no-ckpt-truncate / corrupt-safe)
  validated via the live `_restore_or_reset_worm`.
- End-to-end SIGKILL crash → restore verified on a temp data dir.
- `test_determinism.py`, `test_generations.py`, `test_smoke_multi.py` green.

## Deploy notes
- #2 (healthcheck script) and #4 (viewer JS, on browser reload) are live without
  an app restart.
- #1 and #3 require an app restart + `daemon-reload` (done) to activate.
  Restarting *now* would lose the current in-flight gen-0 (the running process
  predates checkpointing). Recommended: restart at the next clean generation
  boundary (after gen-0 rolls over and banks its poems) for zero loss. The
  lenient watchdog (#2) already protects gen-0 in the meantime.
