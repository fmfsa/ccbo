"""Re-apply the repo's local modifications to third_party/CausalBO_Benchmark.

The benchmark is downloaded (gitignored), and ClusterBench10 depends on one
local extension: a ``quadratic`` relationship type in
``baselines/BO_CBO/graph.py`` giving the target an interior optimum,

    node = intercept + sum_p coefs[p] * (parent_p - centers[p])**2 + noise,

with ``centers`` read from ``relationship_params`` (written by
scripts/generate_clusterbench10.py). Without it the benchmark's unknown-type
fallback silently treats Y as LINEAR, moving the optimum to a box vertex and
failing the generator's oracle checks. This script exists because the original
patch lived only inside the gitignored tree and was lost with it; run it after
every fresh benchmark download, BEFORE generate_clusterbench10*.py.

Idempotent.  Run:  python scripts/patch_benchmark.py
"""

import os
import sys

GRAPH_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                        "third_party", "CausalBO_Benchmark",
                        "baselines", "BO_CBO", "graph.py")

MARKER = 'relationship_type == "quadratic":'

# Inserted before the unknown-type fallback; noise convention copied from the
# stock "linear" branch (ndarray epsilon = deterministic objective path).
QUADRATIC_BLOCK = '''\
                elif relationship_type == "quadratic":
                    def create_quadratic_function(deps, coefs, const_intercept, params):
                        def node_function(epsilon=0, **kwargs):
                            centers = params.get("centers", {})
                            result = const_intercept
                            for parent in deps:
                                if parent in kwargs:
                                    result += coefs[parent] * (kwargs[parent] - centers.get(parent, 0.0)) ** 2

                            noise_std = params.get("noise_std", 0.01)
                            if isinstance(epsilon, (int, float)) and epsilon == 0:
                                noise = np.random.normal(0, noise_std)
                            else:
                                noise = epsilon

                            return result + noise
                        return node_function

                    sem_functions[node] = create_quadratic_function(
                        dependencies, coefficients, intercept, relationship_params
                    )

'''

ANCHOR = ('                else:\n'
          '                    def create_endogenous_function(deps, coefs, const_intercept):')


def main():
    path = os.path.normpath(GRAPH_PY)
    if not os.path.exists(path):
        sys.exit(f"not found: {path} (download the benchmark first)")
    with open(path) as f:
        src = f.read()
    if MARKER in src:
        print(f"already patched: {path}")
        return
    if ANCHOR not in src:
        sys.exit("anchor not found -- benchmark layout changed; patch by hand")
    with open(path, "w") as f:
        f.write(src.replace(ANCHOR, QUADRATIC_BLOCK + ANCHOR, 1))
    print(f"patched quadratic relationship into {path}")


if __name__ == "__main__":
    main()
