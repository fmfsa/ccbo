# QCBO — Results Overview

_Generated 2026-06-17. Protocol: 100 trials, 5 seeds, unit cost. Scoring: CausalBO-benchmark GAP / PA-GAP (higher = better) vs each dataset's shipped `y*`. Errors are ± s.e. (ddof=1)._

**Status:** QCBO (finest + coarse) and baselines **BO, CBO, CEO, CoCaBO, DCBO** are complete on all 6 curated datasets. **MCBO** finished only ToyGraph + Synthetic-2 (stopped early; slow at 100 trials) and is **excluded** from the head-to-head below — to be added later.

---

## 1. QCBO on the standardized benchmark (finest vs coarse)

QCBO-finest **is the CBO algorithm at the identity partition** (Prop. 1): exploration set = all variable subsets up to size `m = min(5, |X|)`, matching the benchmark's own `DiscoveredGraph.get_sets`. Finest→coarse change = the *price of coarsening*.

> **Corrected re-run (2026-06-17).** These numbers replace the earlier ones, which were produced with a `max_intervention_size=1` bug in `ccbo/benchmark.py` that silently restricted finest to **single-variable** arms (so it was *not* CBO). The adapter now defaults to `min(5,|X|)`; suite re-run via `scripts/run_qcbo_benchmark_parallel.py`.

| Dataset | Partition (coarse) | y\* | QCBO | Final Y | GAP@100 | PA-GAP@100 |
|---|---|---|---|---|---|---|
| ToyGraph | {X,Z} | −2.57 | finest | −2.172 ± 0.000 | 0.390 ± 0.056 | 0.176 ± 0.065 |
| | | | coarse | −2.172 ± 0.000 | 0.453 ± 0.068 | 0.238 ± 0.058 |
| Synthetic-2 | {X,Z} | −3.97 | finest | −1.856 ± 0.000 | 0.370 ± 0.036 | 0.082 ± 0.029 |
| | | | coarse | −0.511 ± 0.000 | 0.490 ± 0.007 | 0.018 ± 0.006 |
| Synthetic (CompleteGraph) | {B},{D,E} | −4.45 | finest | −1.739 ± 0.212 | 0.127 ± 0.109 | 0.018 ± 0.017 |
| | | | coarse | −1.737 ± 0.257 | 0.317 ± 0.099 | 0.051 ± 0.015 |
| Healthcare | {Aspirin,Statin} | 4.04 | finest | 5.352 ± 0.041 | 0.208 ± 0.128 | 0.011 ± 0.009 |
| | | | coarse | 5.329 ± 0.015 | 0.360 ± 0.071 | 0.033 ± 0.010 |
| Epidemiology | {L,B} | −3.39 | finest | −2.344 ± 0.031 | 0.542 ± 0.106 | 0.168 ± 0.044 |
| | | | coarse | −2.300 ± 0.026 | 0.473 ± 0.058 | 0.254 ± 0.027 |
| Ecology (max) | {C,N,O},{D,T} | 9.27 | finest | 5.023 ± 0.036 | 0.099 ± 0.067 | 0.001 ± 0.001 |
| | | | coarse | 4.983 ± 0.051 | 0.286 ± 0.083 | 0.020 ± 0.008 |

> **Caveat:** GAP is improvement *relative to each method's initial incumbent*, and `y*` is an extreme/oracle reference (sometimes attained outside the searched domain — e.g. epidemiology's optimizer has L≈736) — read as *competitive ranking*, not an attainment fraction.
>
> **Sample-efficiency effect (new, from the corrected ES):** with the faithful CBO exploration set, finest spreads the 100-trial budget across many arms on high-dim datasets (Synthetic 7 arms, Ecology 31), so its GAP *drops*; coarse keeps few cluster-arms and **beats finest on GAP** for ToyGraph/Synthetic/Healthcare/Ecology at comparable final Y. Coarsening buys sample efficiency, not just robustness.

---

## 2. Head-to-head vs the field (correctly-specified graphs)

**GAP@100** (higher = better; **bold** = best in column):

| Method | ToyGraph | Synth-2 | Synthetic | Healthcare | Epidem. | Ecology |
|---|---|---|---|---|---|---|
| BO | 0.25 ± .07 | 0.08 ± .05 | 0.49 ± .08 | 0.43 ± .08 | 0.55 ± .07 | 0.30 ± .09 |
| CBO | 0.56 ± .03 | 0.27 ± .11 | 0.37 ± .08 | 0.50 ± .02 | 0.38 ± .08 | 0.16 ± .10 |
| CEO | 0.49 ± .07 | 0.00 ± .00 | 0.34 ± .08 | 0.47 ± .03 | 0.38 ± .06 | 0.24 ± .07 |
| CoCaBO | 0.44 ± .06 | **0.67 ± .07** | **0.61 ± .07** | **0.61 ± .08** | **0.78 ± .09** | **0.63 ± .09** |
| DCBO | **0.66 ± .08** | 0.23 ± .14 | 0.46 ± .01 | 0.00 ± .00 | 0.27 ± .03 | 0.40 ± .10 |
| **QCBO-finest** | 0.39 ± .06 | 0.37 ± .04 | 0.13 ± .11 | 0.21 ± .13 | 0.54 ± .11 | 0.10 ± .07 |
| **QCBO-coarse** | 0.45 ± .07 | 0.49 ± .01 | 0.32 ± .10 | 0.36 ± .07 | 0.47 ± .06 | 0.29 ± .08 |

**PA-GAP@100** (higher = better):

| Method | ToyGraph | Synth-2 | Synthetic | Healthcare | Epidem. | Ecology |
|---|---|---|---|---|---|---|
| BO | 0.04 ± .03 | 0.00 ± .00 | 0.21 ± .02 | 0.06 ± .02 | 0.24 ± .03 | 0.06 ± .03 |
| CBO | **0.28 ± .06** | 0.09 ± .03 | 0.11 ± .01 | 0.05 ± .02 | 0.10 ± .05 | 0.01 ± .01 |
| CEO | 0.18 ± .03 | 0.00 ± .00 | 0.06 ± .01 | 0.10 ± .00 | 0.07 ± .01 | 0.00 ± .00 |
| CoCaBO | 0.26 ± .06 | **0.28 ± .03** | **0.38 ± .03** | **0.29 ± .06** | **0.38 ± .04** | **0.27 ± .06** |
| DCBO | 0.18 ± .07 | 0.04 ± .04 | 0.01 ± .01 | 0.00 ± .00 | 0.09 ± .03 | 0.00 ± .00 |
| **QCBO-finest** | 0.18 ± .06 | 0.08 ± .03 | 0.02 ± .02 | 0.01 ± .01 | 0.17 ± .04 | 0.00 ± .00 |
| **QCBO-coarse** | 0.24 ± .06 | 0.02 ± .01 | 0.05 ± .02 | 0.03 ± .01 | 0.25 ± .03 | 0.02 ± .01 |

> Read: QCBO is **competitive / mid-pack** on these (correctly-specified) benchmarks — it is *not* the GAP leader (CoCaBO is strongest). Expected: the benchmark gives everyone the true graph, so QCBO's robustness advantage is not exercised here. (Note: QCBO-finest shares CBO's *algorithm* and the full size-≤m exploration set, but is a separate implementation from the benchmark's own CBO baseline — so the two rows are not numerically identical.) The contribution lands in §3–4 below. **Resolved:** the earlier flag (QCBO-finest Healthcare GAP low vs CBO) was a GAP-metric artifact, not a search failure — GAP measures improvement from the initial incumbent, and QCBO-finest reaches a comparable final Y (5.35 vs CBO 5.38 at task=min).

---

## 3. Tier-2 robustness — arm deletion (ConfoundedCluster "bow") ★ headline

A spurious intra-cluster edge B→C + latent B↔C forms a bow ⇒ `P(Y|do(B))` unidentifiable. Ground-truth arms: μ\*({B}) = −3.75, μ\*({C}) = −2.75, μ\*({B,C}) = −4.25.

| Partition | Assumed DAG | \|ES\| | Final Y | Gap vs correct |
|---|---|---|---|---|
| finest | WrongBC | **2** | −2.749 ± .000 | **+1.002 ± .000** |
| finest | correct | 3 | −3.750 ± .000 | — |
| {A},{B,C} | WrongBC | 2 | −4.232 ± .006 | **+0.000 ± .000** |
| {A},{B,C} | correct | 2 | −4.232 ± .006 | — |

**Finest CBO loses the best arm {B} (\|ES\| 3→2), an unrecoverable +1.0 gap. Coarse partition: the bow is intra-cluster ⇒ invisible in 𝒢π ⇒ byte-identical to the correct run (+0.000).**

---

## 4. Tier-1 robustness — prior bias (CompleteGraph)

Deleting an edge that corrupts priors but deletes no arm (\|ES\| unchanged).

| Partition | Assumed DAG | \|ES\| | Final Y | Gap vs correct |
|---|---|---|---|---|
| finest | NoBC | 7 | −3.404 ± .167 | +0.112 ± .113 |
| finest | NoCD | 7 | −3.571 ± .002 | −0.054 ± .055 |
| finest | correct | 7 | −3.516 ± .054 | — |
| {B},{D,E} | NoBC | 3 | −1.179 ± .137 | +0.000 ± .000 |
| {B},{D,E} | NoCD | 3 | −1.179 ± .137 | +0.000 ± .000 |
| {B},{D,E} | correct | 3 | −1.179 ± .137 | — |

**By 100 trials the finest-partition damage is within noise of zero (prior bias self-heals); coarse is exactly immune. Contrast with Tier-2, which never recovers.**

---

## 5. Price of coarsening (CompleteGraph, exact via MC on the SEM)

| Intervention | μ\* | Note |
|---|---|---|
| do(B, D) | **−3.60** | global optimum (joint arm) |
| do(B, D, E) | −0.57 | forcing E destroys it |
| coarse {B},{D,E} floor = do(B) | −1.31 | optimum splits {D,E} → unreachable |

**The coarse partition cannot represent `do(B,D)` (it splits the {D,E} cluster), so it sits at its floor −1.31. Loss is zero iff some optimum is a union of whole clusters (Prop. 5).**

---

## Bottom line

1. **Robustness is the contribution** — Tier-2: misspecification deletes the optimal arm under finest CBO (+1.0, permanent) but is provably invisible under the matching coarse partition (+0.000). Tier-1: prior bias self-heals.
2. **Competitive, not dominant** on the standardized benchmark (true graphs) — as expected; QCBO-finest runs the CBO algorithm at the identity partition (a separate implementation from the benchmark's own CBO baseline, so not numerically identical).
3. **Coarsening has an exact, characterized price** (Prop. 5), demonstrated on CompleteGraph.
4. Coarsening can also *restore* identifiability (bow becomes intra-cluster).

## Provenance

- QCBO: `results/qcbo_benchmark_results.json` → `paper/tables/benchmark_gap.tex` (`scripts/emit_benchmark_table.py`)
- Head-to-head: per-seed CSVs under `third_party/CausalBO_Benchmark/results/<METHOD>/` → `results/baseline_comparison.json`, `paper/tables/benchmark_comparison{,_pagap}.tex` (`scripts/run_baselines_suite.py`, `scripts/emit_baseline_table.py`)
- Robustness: `paper/tables/{CompleteGraph,ConfoundedCluster}_wrong_edge.tex` + `paper/figures/*_wrong_edge.pdf` (`run_experiments.py`)
- Arm values: `scripts/*_arm_values.py`
- _Pending:_ MCBO on synthetic/healthcare/epidemiology/ecology, then re-emit head-to-head with MCBO.
