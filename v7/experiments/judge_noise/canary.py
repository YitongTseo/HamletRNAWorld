"""Tiny live validation of the judge interfaces before the full run."""
from __future__ import annotations
import random
from pool_builder import POOLS, load_pool, represent, pool_id
from judges import absolute_batch, pairwise, bradley_terry

proc, flask, gen = POOLS[0]
pool = load_pool(proc, flask, gen)
names = sorted(pool)
print(f"pool {pool_id(proc, flask, gen)}: {len(names)} worms; "
      f"windows/worm = {[len(pool[n]) for n in names[:4]]}...")

rng = random.Random(0)
a = represent(pool[names[0]], "stratified", 8, rng)
print(f"\nrepresenting {names[0]} with 8 stratified windows (idxs {[w.idx for w in a]})")

print("\n-- absolute @ temp=0 (x2, should be ~identical) --")
s1 = absolute_batch(a, 0.0); s2 = absolute_batch(a, 0.0)
print("run1:", {k: v for k, v in list(s1.items())[:4]})
print("run2:", {k: v for k, v in list(s2.items())[:4]})

print("\n-- absolute @ temp=1 (x2, expect variation) --")
t1 = absolute_batch(a, 1.0); t2 = absolute_batch(a, 1.0)
print("run1:", {k: v for k, v in list(t1.items())[:4]})
print("run2:", {k: v for k, v in list(t2.items())[:4]})

print("\n-- pairwise (both orders) --")
b = represent(pool[names[1]], "stratified", 8, rng)
ab = pairwise(a, b, 0.0); ba = pairwise(b, a, 0.0)
print(f"{names[0]} vs {names[1]}: A-first winner={ab} ; swapped winner={ba}")

print("\n-- bradley_terry smoke --")
wins = {(names[0], names[1]): 2, (names[1], names[0]): 1, (names[2], names[0]): 3}
bt = bradley_terry(names[:3], wins)
print({k: round(v, 3) for k, v in bt.items()})
print("\nCANARY OK")
