# Known issues — RCCBO / RePaRe backlog

RCCBO is excluded from the current paper (the submission covers QCBO only;
online partition discovery is deferred to follow-up work). The issues below
must be resolved before RCCBO results are reported anywhere.

## Baselines

- **GACBO is not run as a head-to-head baseline.** Mukherjee 2024's RJ-MCMC
  sampler over DAGs has no faithful public implementation, and an idealized
  enumeration handed the true graph would be an optimistic stand-in (the former
  `ccbo/baselines/gacbo.py` stub has been removed; see
  `ccbo/baselines/__init__.py`). We instead compare on the CausalBO benchmark
  via its own scorer. The benchmark ships a
  baseline literally named "CCBO" (unrelated to our method) — we renamed ours
  to **QCBO** to avoid the clash.

## RePaRe (vendored partition discovery, `ccbo/repare_lib/`)

1. **Conditioning set ignored in the nonparametric adjacency test**
   ([repare.py:278-280](ccbo/repare_lib/repare.py)). `p_val(x, y, z)` calls
   `dcor.independence.distance_correlation_t_test(x, y)` and never uses `z`,
   so the test is marginal rather than conditional. Edges between clusters
   that are d-separated given the conditioning set will be spuriously kept.
   Fix options: partial distance correlation, or residualize x and y on z
   first; alternatively restrict RePaRe to `assume="gaussian"` (LRT path,
   which does condition correctly) and document the restriction.

2. **Discrete branch is a silent stub**
   ([repare.py:302](ccbo/repare_lib/repare.py)): `assume="discrete"` hardcodes
   `p_val = 1` in the adjacency test, so no edge is ever added for discrete
   data. Implement or remove the branch.

3. **Shadowed loop variable in the refinement test**
   ([repare.py:144-155](ccbo/repare_lib/repare.py)): the inner KS loop reuses
   `j` from the outer loop. Results are still correct (all entries get
   overwritten with the right values) but the outer loop body runs
   `num_feats` times more often than needed.

## RCCBO driver (`ccbo/rccbo/`)

4. **`partition_diff` does not verify the refinement property**
   ([partition_ops.py:52-88](ccbo/rccbo/partition_ops.py)). It assumes the
   new partition refines the old one; if RePaRe returns a non-refinement
   (mixed split/merge), the diff is silently wrong and the monotonicity that
   Prop. 3 (refinement monotonicity) promises is not enforced. Call
   `is_refinement()` and reject or project non-refinements.

5. **Hardcoded seeds bypass the experiment seed**
   ([state_manager.py:186,202,221,233](ccbo/rccbo/state_manager.py)): data
   lifting and initial-data generation use fixed seeds (0, 42, loop index)
   instead of the run seed, so RCCBO runs are not fully controlled by the
   `seed` argument across refinement events.

## Status

- RCCBO is removed from `run_experiments.py` conditions; the code paths are
  kept so the follow-up work can resume from here.
- `ccbo/rccbo/mcbo_baseline.py` was dead code (never imported) and has been
  removed.
