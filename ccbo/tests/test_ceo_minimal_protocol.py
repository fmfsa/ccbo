"""Protocol tests for the CEO-on-MinimalBench adapter (scripts/run_ceo_minimal.py).

Guards the noisy/noiseless seam that PR-review finding 1 flagged: the NOISY
target path must produce genuine stochastic draws of the true latent SEM
(they feed CEO's GP targets and graph-posterior evidence), while the
NOISELESS path must be the deterministic closed-form E[Y|do] oracle and
consume no randomness (it feeds only trajectory scoring). Also guards the
alias metadata contract (finding 5). No third_party/CEO dependency.
"""

import importlib.util
import json
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_ceo_minimal", os.path.join(_REPO, "scripts", "run_ceo_minimal.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rcm = _load()

ARMS = {  # one multi-var and one single-var arm per SCM
    "ParallelParent": [("X1",), ("X1", "X2")],
    "FrontDoor": [("X1",), ("X1", "M")],
    "MediatedChain": [("X2",), ("X1", "X2")],
}


def _mk(scm, es, noisy, rs):
    factory = rcm.make_target_factory(scm)
    return factory(noisy=noisy, random_state=rs,
                   initial_structural_equation_model=None,
                   structural_equation_model=None, graph=None,
                   exploration_set=es, all_vars=None, T=1)


@pytest.mark.parametrize("scm", list(ARMS))
def test_noisy_path_is_stochastic_and_not_the_oracle(scm):
    es = ARMS[scm][0]
    levels = np.zeros(len(es))
    rs = np.random.RandomState(0)
    target = _mk(scm, es, True, rs)
    a = target("Y_0", levels, {})
    b = target("Y_0", levels, {})
    oracle = rcm.SCMS[scm]["oracle"]({v: 0.0 for v in es})
    assert float(a["Y"][0]) != float(b["Y"][0]), "consecutive draws identical"
    assert float(a["Y"][0]) != pytest.approx(oracle, abs=1e-12), \
        "noisy draw equals the oracle exactly"
    non_int = [v for v in rcm.SCMS[scm]["nodes"]
               if v not in es and v != "Y"]
    for v in non_int:
        assert float(a[v][0]) != float(b[v][0]), \
            f"non-intervened {v} did not vary across draws"


@pytest.mark.parametrize("scm", list(ARMS))
def test_noiseless_path_equals_oracle_and_consumes_no_rng(scm):
    for es in ARMS[scm]:
        levels = np.linspace(-1.0, 1.0, len(es))
        iv = {v: float(levels[j]) for j, v in enumerate(es)}
        rs = np.random.RandomState(7)
        probe_before = np.random.RandomState(7).randn(5)
        target = _mk(scm, es, False, rs)
        a = target("Y_0", levels, {})
        b = target("Y_0", levels, {})
        assert float(a["Y"][0]) == float(b["Y"][0])
        assert float(a["Y"][0]) == pytest.approx(
            float(rcm.SCMS[scm]["oracle"](iv)), abs=1e-12)
        # the shared RandomState stream must not have advanced
        assert np.array_equal(rs.randn(5), probe_before)


@pytest.mark.parametrize("scm", list(ARMS))
def test_noisy_mean_matches_oracle(scm):
    es = ARMS[scm][0]
    spec = rcm.SCMS[scm]
    rng = np.random.RandomState(123)
    iv = {v: float(rng.uniform(*spec["domain"][v])) for v in es}
    ys = np.array([rcm.sample_true(scm, iv, rng)["Y"] for _ in range(20_000)])
    se = float(np.std(ys) / np.sqrt(len(ys)))
    assert abs(float(np.mean(ys)) - float(spec["oracle"](iv))) <= 4 * se + 1e-6


def test_initial_data_pair_shares_levels_but_differs_in_response():
    D, Dn = rcm.initial_interventional_data(
        "ParallelParent", [("X1",), ("X1", "X2")], ninit=3, seed=0)
    for es in D:
        for v in es:  # clamped coordinates: identical levels in both sets
            assert np.array_equal(D[es][v], Dn[es][v])
        assert not np.array_equal(D[es]["Y"], Dn[es]["Y"])


def test_materialize_aliases_sets_alias_cond(tmp_path):
    src_csv = tmp_path / "CEO_ParallelParent_A0_seed0.csv"
    src_csv.write_text("trial_number,current_optimal\n0,1.0\n")
    meta = {"scm": "ParallelParent", "cond": "A0", "seed": 0,
            "pool_pids": ["A0"], "baseline_observable_cond": "A0"}
    (tmp_path / "CEO_ParallelParent_A0_seed0.meta.json").write_text(
        json.dumps(meta))
    rcm.materialize_aliases(str(tmp_path))
    dst = tmp_path / "CEO_ParallelParent_A3_seed0.csv"
    assert dst.exists() and dst.read_bytes() == src_csv.read_bytes()
    dst_meta = json.loads(
        (tmp_path / "CEO_ParallelParent_A3_seed0.meta.json").read_text())
    assert dst_meta["cond"] == "A3"
    assert dst_meta["alias_of"] == "A0"
