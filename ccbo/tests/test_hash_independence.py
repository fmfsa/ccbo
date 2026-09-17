"""Causal prior moments must be independent of Python hash ordering."""

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
    from ccbo import minibench as mb
    from ccbo.matched_protocol import observational_data
    from ccbo.run_experiment import get_original_graph
    from ccbo.coarsened_graph import CoarsenedGraph
    dataset, n_obs = sys.argv[1], int(sys.argv[2])
    mb.register_variants()
    out = {}
    for algo in ("CBO", "QCBO"):
        np.random.seed(0)
        obs = observational_data(dataset, 0).iloc[:n_obs]
        graph = get_original_graph(dataset, obs)
        partition = mb.fine_partition(dataset) if algo == "CBO" else mb.coarse_partition(dataset)
        cg = CoarsenedGraph(graph, partition, dataset, obs,
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


def test_parallel_parent_priors_are_hash_independent():
    r0, r1 = _run("0", "ParallelParent", 60), _run("1", "ParallelParent", 60)
    assert r0 and r0 == r1


@pytest.mark.slow
def test_frontdoor_priors_are_hash_independent():
    r0, r1 = _run("0", "FrontDoor", 100), _run("7", "FrontDoor", 100)
    assert r0 and r0 == r1
