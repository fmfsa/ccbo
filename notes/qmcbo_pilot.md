# QMCBO pilot: quotient mechanisms for model-based causal BO

*Feasibility note — prototype + pilot, per the agreed scope. Not a paper draft.*

## The idea in one paragraph

MCBO (Sussex et al., ICLR'23) models each **node's mechanism** as a GP (inputs
= parent values + actions) and composes them along the full DAG; its graph
dependence is therefore *architectural* — a wrong intra-cluster edge corrupts
the surrogate's factorization itself, a deeper failure mode than the prior
bias / arm deletion taxonomy of arm-based CBO. **QMCBO** runs the identical
machinery on the **quotient**: each node's mechanism conditions on the *union
of its parent clusters' members* (per-coordinate cluster mechanisms composed
along the C-DAG), and do() targets are lifted to whole-cluster patterns
(QCBO-coarse's arm space). Intra-cluster and quotient-redundant edges are
invisible to the architecture by construction — the model-based analogue of
QCBO's Prop. 2, for the method class most exposed to graph misspecification.

## What made v1 cheap

MCBO's code touches the graph only through `dag.get_parent_nodes(k)`. QMCBO
v1 is therefore a **pure env_profile transformation** (`ccbo/qmcbo/quotient.py`)
— no GP-network subclassing, the stock `mcbo_trial` runs unchanged. This makes
the invariance claim mechanically checkable: intra-cluster edits of the fine
DAG produce a byte-identical transformed profile (unit-tested), and with
seeding fixed (the vendored `torch.seed()` de-seeding is neutralized at
runtime) byte-identical *trajectories*.

## Setup

Vendored MCBO envs (same SCMs as the CausalBO benchmark datasets):
ToyGraph (partition {X,Z}), Synthetic_2 ({X,Z}), PSAGraph/healthcare
({Aspirin,Statin}). Protocol = the benchmark's own MCBO settings (algo MCBO
= hallucinated GP network + qSimpleRegret, beta 10, noise 0, T=100, 5 seeds).
Causal sufficiency assumed throughout (MCBO's own assumption; see obstacles).
QMCBO differs from MCBO **only** in the model's graph view and the lifted
targets — a pure architecture ablation.

## E0 (smoke, ToyGraph, 2 seeds, T=50): PASS

QMCBO optimizes non-degenerately: improves in the first ~5 trials, converges
to 1.0004 on both seeds. Notably, stock MCBO **never improved on its initial
design** (+0.000 over 50 trials, both seeds) under its own benchmark settings.
QMCBO ~30% faster per unit (106s vs 147s) — fewer mechanisms, fewer targets.

## E1 (feasibility, 3 envs x 2 algos x 5 seeds, T=100): PASS

| env (optimum) | MCBO final | QMCBO final | wall-clock/unit |
|---|---|---|---|
| PSAGraph (-4.13) | -4.341±0.099 | **-4.201±0.055** | 3574s → **1074s (3.3x)** |
| Synthetic_2 (2.152) | 1.845±0.009 | **1.855±0.001** | 766s → 449s (1.7x) |
| ToyGraph (2.172) | **1.422±0.222** | 1.261±0.160 | 573s → 386s (1.5x) |

QMCBO is better on the two larger/harder envs — on PSAGraph (the biggest, 6
nodes) it ends closer to the optimum with half the seed variance at a third
of the compute — and within overlapping error bars on ToyGraph, the one env
where v1's known approximation bites ({X,Z} holds the only intra-cluster
edge, whose within-cluster correlation v1 ignores; the cost shows up exactly
where predicted). QMCBO is faster everywhere (fewer mechanism GPs to refit
per trial, fewer do-targets to optimize over).

Cross-family context (same datasets, different harness — protocol caveat):
the arm-based rows for these datasets live in results/qcbo_benchmark_results.json
and results/clusterbench10_refine.json.

## E2 (robustness headline, intra-cluster misspecification of the model view)

Perturbations (model view only; ground truth fixed): ToyGraph del X->Z inside
{X,Z}; Synthetic_2 rev X->Z; PSAGraph add spurious Aspirin->Statin inside
{A,S}. Same seeds, deterministic runs.

**Result — the model-based Prop.-2 mirror, at machine precision:**

* **QMCBO: byte-identical trajectories, correct-vs-perturbed, on 15/15
  (env, seed) pairs.** Invariance by construction: the quotient view of the
  perturbed DAG is literally the same object (unit-tested), so the entire run
  cannot move.
* **MCBO: trajectories shift on 14/15 pairs** (the one coincidence, ToyGraph
  seed 3, is luck — its node GPs did change parents; nothing constrains the
  outcome). Direction is unpredictable: here the wrong graph sometimes
  *helped* MCBO (ToyGraph 1.807 misspec vs 1.422 correct; PSAGraph -4.026 vs
  -4.341) — optimism under a wrong model got lucky on these SCMs. That is
  precisely the point the QCBO paper makes about invariance being a claim
  about *behavior*: a method whose architecture reads the fine edges moves
  when they are wrong, in a direction no one controls; a quotient method
  provably does the same thing.

## The four obstacles (why this is a pilot, not a paper)

1. **Hidden confounders.** MCBO assumes causal sufficiency (independent
   per-node noise); C-DAGs carry bidirected edges. ClusterBench10's U12
   confounds X2's response exactly where a partial-arm evaluation would need
   it. Correlated-noise cluster mechanisms are the open problem.
2. **Within-cluster correlation.** v1 samples cluster coordinates
   independently given cluster inputs; intra-cluster edges induce residual
   correlation v1 ignores. Upgrade path: joint multi-output cluster GPs.
3. **Partial clamping.** Clamping a subset of a cluster's coordinates while
   the rest respond (the "shadow surrogates made native" result — evaluating
   do(X1,X3) without refinement) needs conditional-MVN logic in the
   propagation samplers. Deferred; the robustness headline doesn't need it.
4. **Regret analysis.** MCBO's guarantees assume calibrated per-mechanism GPs
   on the true graph; the quotient version needs cluster-level smoothness
   assumptions and an account of the abstraction error (cf. MFACBO's delta_l).

## Go / no-go

**GO.** All three gates passed: E0 (non-degenerate optimization), E1
(comparable-or-better finals vs MCBO, 1.5-3.3x faster, lower variance), E2
(machine-precision invariance vs MCBO's 14/15 sensitivity). The construction
is cheap (a profile transformation, zero vendored-code edits), the headline
result is exact rather than statistical, and the efficiency observation is
consistent across every env. The paper's core experiment already exists in
this pilot; what a full paper adds: the ClusterBench10-NC env (2-cluster
richness + our misspec taxonomy), the partial-clamp prototype (obstacle 3 —
the genuinely novel capability), joint multi-output cluster mechanisms
(obstacle 2, fixes the ToyGraph gap), and the confounder story (obstacle 1,
the hard theory). Recommended order: obstacle 2 first (cheap, direct fix to
the one E1 loss), then partial clamping (the new capability), with
confounders as the paper's honest limitation exactly as sufficiency is
MCBO's.

## Framing rules (carried over from the QCBO paper)

No claims against full-knowledge CBO's regret. The claim is within the
model-based family: MCBO's architecture breaks under intra-cluster
misspecification; QMCBO's cannot represent it. Efficiency numbers reported
as observations, not claims.
