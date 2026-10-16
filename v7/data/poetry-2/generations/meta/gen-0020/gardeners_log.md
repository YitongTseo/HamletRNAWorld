# epoch 20 — 2026-10-16T01:44:13Z

Both flasks are now pinned near their σ floor (F1 0.067, F2 0.020) while the winner still changes every single generation — Dave→Liam in Flask 2, four different winners in four gens in Flask 1 — which means the 1/5 rule keeps shrinking the step because judge noise is making fresh children *look* like failures. The fix took for float drift but the signal is still too small to climb against the noise; σ_MIN may need raising, or the judge needs more windows per worm than SAMPLE_FRACTION=0.25 gives.
