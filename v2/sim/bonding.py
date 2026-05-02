"""Dynamic Watson-Crick base-pair formation and breaking."""
from __future__ import annotations

import numpy as np

from .sequence import pair_table
from .state import SimState


class Bonder:
    """Periodically scans the chain and updates the dynamic base-pair list.

    Formation: free, complementary pairs within cutoff form with probability
    `bp_form_prob`. Each bead can pick up at most one new bond per pass, and
    closer candidates are preferred (sorted by distance).

    Breaking: existing bonds break deterministically when stretched beyond
    `bp_break_length`, plus a small per-pass baseline rate to allow
    re-arrangement out of local minima.
    """

    def __init__(self, state: SimState, seed: int = 1):
        self.s = state
        self._rng = np.random.default_rng(seed)
        self._pair_tbl = pair_table()

    def maybe_update(self) -> None:
        """Run an update pass if step_count is on the bonding interval."""
        if self.s.step_count % self.s.params.bonding_interval != 0:
            return
        self._break_pass()
        self._form_pass()

    def _break_pass(self) -> None:
        s = self.s
        p = s.params
        if len(s.base_pairs) == 0:
            return
        i = s.base_pairs[:, 0]
        j = s.base_pairs[:, 1]
        dr = s.pos[j] - s.pos[i]
        r = np.sqrt((dr * dr).sum(axis=1))
        random_break = self._rng.random(len(r)) < p.bp_break_rate
        breaks = (r > p.bp_break_length) | random_break
        if not breaks.any():
            return
        s.bp_partner[i[breaks]] = -1
        s.bp_partner[j[breaks]] = -1
        s.base_pairs = s.base_pairs[~breaks]

    def _form_pass(self) -> None:
        s = self.s
        p = s.params
        free_mask = s.bp_partner == -1
        free_idx = np.where(free_mask)[0]
        if len(free_idx) < 2:
            return

        pos_f = s.pos[free_idx]
        dr = pos_f[None, :, :] - pos_f[:, None, :]
        r2 = np.einsum("ijk,ijk->ij", dr, dr)

        cutoff2 = p.bp_form_cutoff * p.bp_form_cutoff
        # Upper triangle only — each pair once.
        a_local, b_local = np.where(np.triu(r2 < cutoff2, k=1))
        if len(a_local) == 0:
            return

        i_glob = free_idx[a_local]
        j_glob = free_idx[b_local]

        sep_ok = np.abs(j_glob - i_glob) >= p.bp_min_separation
        pair_ok = self._pair_tbl[s.bases[i_glob], s.bases[j_glob]]
        keep = sep_ok & pair_ok
        if not keep.any():
            return

        i_glob = i_glob[keep]
        j_glob = j_glob[keep]
        r_cand = np.sqrt(r2[a_local[keep], b_local[keep]])

        # Prefer closer candidates; resolve conflicts greedily.
        order = np.argsort(r_cand)
        claimed = np.zeros(s.n, dtype=bool)
        new_pairs: list[tuple[int, int]] = []
        for k in order:
            if self._rng.random() >= p.bp_form_prob:
                continue
            a = int(i_glob[k])
            b = int(j_glob[k])
            if claimed[a] or claimed[b]:
                continue
            claimed[a] = True
            claimed[b] = True
            new_pairs.append((a, b))
            s.bp_partner[a] = b
            s.bp_partner[b] = a

        if new_pairs:
            arr = np.array(new_pairs, dtype=np.int32)
            s.base_pairs = np.concatenate([s.base_pairs, arr], axis=0)
