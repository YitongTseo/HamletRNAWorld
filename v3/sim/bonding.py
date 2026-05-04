"""Letter-level pair bonding gated by token-level embedding affinity.

A letter bead can hold at most one pair partner — but because words have
many letters, a word with N letters can pick up to N pair partners (with
up to N other words). That's how multi-partner pairing emerges.

Gates applied to a candidate letter pair (i, j):
  1. Both letters' parent tokens are reactive (no stop words / punct /
     whitespace).
  2. Letters belong to different tokens (a word doesn't pair with itself).
  3. Sequence-position gate (within the same chain only): |i - j| >= min_sep.
  4. Distance < bp_form_cutoff.
  5. token_affinity[token(i), token(j)] >= affinity_threshold.

Formation probability is bp_form_prob × token_affinity (so the most-opposite
token-pairs bond most readily).
"""
from __future__ import annotations

import numpy as np

from .state import SimState


class Bonder:
    def __init__(self, state: SimState, seed: int = 1):
        self.s = state
        self._rng = np.random.default_rng(seed)

    def maybe_update(self) -> None:
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

        # Reactive letter mask: parent token exists and is reactive.
        token_id = s.token_id
        reactive_letter = np.zeros(s.n, dtype=bool)
        if len(s.token_reactive):
            valid = token_id >= 0
            reactive_letter[valid] = s.token_reactive[token_id[valid]]

        free_mask = (s.bp_partner == -1) & reactive_letter
        free_idx = np.where(free_mask)[0]
        if len(free_idx) < 2:
            return

        pos_f = s.pos[free_idx]
        dr = pos_f[None, :, :] - pos_f[:, None, :]
        r2 = np.einsum("ijk,ijk->ij", dr, dr)

        cutoff2 = p.bp_form_cutoff * p.bp_form_cutoff
        a_local, b_local = np.where(np.triu(r2 < cutoff2, k=1))
        if len(a_local) == 0:
            return

        i_glob = free_idx[a_local]
        j_glob = free_idx[b_local]

        # Different tokens (a word doesn't pair with itself).
        ti = token_id[i_glob]
        tj = token_id[j_glob]
        diff_token = ti != tj

        # Sequence-separation gate within a single chain.
        same_chain = s.chain_id[i_glob] == s.chain_id[j_glob]
        sep_ok = np.where(
            same_chain,
            np.abs(j_glob - i_glob) >= p.bp_min_separation,
            True,
        )

        # Token-level affinity.
        aff = s.token_affinity[ti, tj]
        aff_ok = aff >= p.affinity_threshold

        keep = diff_token & sep_ok & aff_ok
        if not keep.any():
            return

        i_glob = i_glob[keep]
        j_glob = j_glob[keep]
        aff = aff[keep]
        r_cand = np.sqrt(r2[a_local[keep], b_local[keep]])

        # Prefer closer candidates; resolve conflicts greedily. Each letter
        # can claim at most one new partner per pass.
        order = np.argsort(r_cand)
        claimed = np.zeros(s.n, dtype=bool)
        new_pairs: list[tuple[int, int]] = []
        exponent = float(p.bp_form_exponent)
        for k in order:
            # p = bp_form_prob * affinity^exponent. With exponent > 1, weak
            # pairs become disproportionately reluctant, while strong pairs
            # stay almost as eager as under linear scaling.
            prob = p.bp_form_prob * (float(aff[k]) ** exponent)
            if self._rng.random() >= prob:
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
