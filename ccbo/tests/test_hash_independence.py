"""The causal-prior pipeline must not depend on Python's per-process hash
seed.  ``sorted()`` over frozenset-typed C-DAG vertices is only a partial
order, so before the 2026-09-10 fix the adjustment set chosen among equal-size
candidates, the GP input column order and the g-computation step order all
followed ``PYTHONHASHSEED`` (the CBO-family CompleteGraph and Coral priors
differed between processes).  These tests fit the priors in two fresh
interpreters with different hash seeds and require identical moments."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from ccbo import adjustment as adj

_PROBE = textwrap.dedent("""
    import json, sys, warnings, logging
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    import numpy as np
    sys.path.insert(0, "scripts")
    from run_cbo_family import PARTITIONS, _load_graph
    from ccbo.coarsened_graph import CoarsenedGraph
    dataset, n_obs = sys.argv[1], int(sys.argv[2])
    out = {}
    for algo in ("CBO", "QCBO"):
        np.random.seed(0)
        graph, obs, _ = _load_graph(dataset, n_obs)
        cg = CoarsenedGraph(graph, PARTITIONS[(dataset, algo)], dataset, obs,
                            num_mc_samples=500)
        do = cg.get_all_do()
        ranges = cg.get_interventional_ranges()
        for key in sorted(do):
            arm = key.replace("compute_do_", "")
            info = cg._identification_info[key]
            x = []
            for es in cg.get_sets()[0]:
                if "".join(es) == arm:
                    x = [0.5 * (ranges[v][0] + ranges[v][1]) for v in es]
            if not x:
                continue
            m, v = do[key](obs, None, np.array(x))
            out[f"{algo}:{arm}"] = {"mean": float(m), "v": float(v),
                                    "method": info.get("method"),
                                    "adj": info.get("adjustment_fine_vars")}
    print(json.dumps(out, sort_keys=True))
""")


def _run(hash_seed: str, dataset: str, n_obs: int) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed, OMP_NUM_THREADS="1",
               PYTHONPATH=os.getcwd())
    res = subprocess.run([sys.executable, "-c", _PROBE, dataset, str(n_obs)],
                         capture_output=True, text=True, env=env, timeout=1800)
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_vertex_key_is_canonical():
    a, b = frozenset({"D", "T"}), frozenset({"C", "N", "O"})
    assert adj._sorted_v([a, b]) == adj._sorted_v([b, a]) == [b, a]
    assert adj._sorted_v(["X2", frozenset({"A"}), "X1"]) == ["X1", "X2", frozenset({"A"})]


def test_complete_graph_priors_are_hash_independent():
    r0, r1 = _run("0", "CompleteGraph", 60), _run("1", "CompleteGraph", 60)
    assert r0 and r0 == r1


@pytest.mark.slow
def test_coral_priors_are_hash_independent():
    r0, r1 = _run("0", "SimplifiedCoralGraph", 100), _run("7", "SimplifiedCoralGraph", 100)
    assert r0 and r0 == r1
