"""Tests for the per-flask (1+1)-ES embedder (server/flask_embedder.py)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import embedding
from server import flask_embedder as fe
from server.flask_embedder import FlaskEmbedderState, flask_eps, step, build_model


def test_eps_deterministic_and_shaped():
    a = flask_eps(3, 7)
    b = flask_eps(3, 7)
    assert a.shape == (embedding.GENOME_SIZE,)
    assert np.allclose(a, b)                       # same (seed, gen) -> same eps
    assert not np.allclose(flask_eps(3, 7), flask_eps(3, 8))   # gen varies
    assert not np.allclose(flask_eps(3, 7), flask_eps(4, 7))   # seed varies


def test_gen0_sets_incumbent_baseline():
    st = FlaskEmbedderState.load_or_init(Path("/nonexistent"), "f", seed=1)
    assert st.generation == 0
    assert st.incumbent_score is None
    # candidate at gen 0 is the incumbent itself (eps = 0).
    assert np.allclose(st.candidate_vec(), st.theta_vec())
    step(st, 5.0)
    assert st.incumbent_score == 5.0
    assert st.generation == 1
    # a fresh candidate (theta + sigma*eps) was drawn for gen 1.
    assert not np.allclose(st.candidate_vec(), st.theta_vec())


def test_accept_on_improve_reject_on_worse():
    st = FlaskEmbedderState.load_or_init(Path("/nonexistent"), "f", seed=1)
    step(st, 5.0)                       # baseline incumbent = 5.0
    cand_before = st.candidate_vec().copy()
    theta_before = st.theta_vec().copy()
    # candidate BEATS incumbent -> theta moves to the candidate.
    step(st, 9.0)
    assert st.incumbent_score == 9.0
    assert np.allclose(st.theta_vec(), cand_before)     # accepted
    assert not np.allclose(st.theta_vec(), theta_before)
    # next candidate is worse -> incumbent unchanged.
    theta_after_accept = st.theta_vec().copy()
    step(st, 1.0)
    assert st.incumbent_score == 9.0
    assert np.allclose(st.theta_vec(), theta_after_accept)   # rejected


def test_sigma_grows_on_frequent_accepts_shrinks_on_rejects():
    st = FlaskEmbedderState.load_or_init(Path("/nonexistent"), "f", seed=1)
    step(st, 0.0)  # baseline
    s0 = st.sigma
    # Steadily improving scores -> mostly accepts -> success_rate > 1/5 -> grow.
    for v in range(1, 12):
        step(st, float(v))
    assert st.sigma > s0
    # Now a long run of rejects -> success_rate < 1/5 -> shrink.
    st2 = FlaskEmbedderState.load_or_init(Path("/nonexistent"), "g", seed=2)
    step(st2, 100.0)  # high baseline
    s0b = st2.sigma
    for _ in range(12):
        step(st2, 0.0)   # never beats baseline
    assert st2.sigma < s0b


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "flask_1" / "embedder.json"
        st = FlaskEmbedderState.load_or_init(p, "flask_1", seed=4)
        step(st, 3.0)
        step(st, 7.0)
        st.save(p)
        st2 = FlaskEmbedderState.load_or_init(p, "flask_1", seed=4)
        assert st2.generation == st.generation
        assert st2.sigma == st.sigma
        assert st2.incumbent_score == st.incumbent_score
        assert np.allclose(st2.theta_vec(), st.theta_vec())
        assert np.allclose(st2.candidate_vec(), st.candidate_vec())


def test_build_model_is_primed_and_matches_genome():
    st = FlaskEmbedderState.load_or_init(Path("/nonexistent"), "f", seed=1)
    m = build_model(st)
    assert m._E_table is not None                 # primed
    # its params ARE the candidate genome
    assert np.allclose(m.params.flatten(), st.candidate_vec())
