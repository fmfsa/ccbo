# Rerun runbook — engine v3 (post-review fixes)

**For the agent running on the DTU HPC.** Companion to `RERUN_ONLINE_OBS.md`
(the August observation-protocol rerun); this one covers the September
engine fixes that followed the external review.

## 1. Why this rerun exists

Confirmed defects in the static CBO engine (see `MANUSCRIPT_REVISION_MEMO.md`
and the plan in the PR description):

1. **Front-door / g-computation priors substituted means** for the mediator
   / propagated conditionals (`ccbo/adjustment.py`). Now integrated:
   Gauss–Hermite over the fitted mediator law; seeded common-random-number
   draws through the DAG with within-cluster chain rule.
2. **Causal-prior gradients were wrong**: `CausalRBF` inherited the plain-RBF
   input gradients (rank-one term ignored, `gradients_X_diag` = 0), the
   causal mean function had no input gradient, and the σ_f² hyperparameter
   gradient used the full kernel. All three feed L-BFGS. Fixed.
3. **Prior variance** is the predictive `Var[Y | do]` under the fitted
   plug-in model (`lik_var + spread + GP epistemic`; see §11 for why the
   epistemic-free variant was retired); **GP noise fixed at 1e-10**; exact
   byte cache keys; identification gate tri-state and fail-closed.
4. **HQCBO** now charges its split design, and a graph-free variant
   (`HQCBOGF`) is added; the initial design is charged on row 0 for every
   arm; every unit writes a full-precision decision log.
5. New ablation arms `CBONP`, `QCBONP` (same arms, plain GPs) and `BOS`
   (all fine subsets, plain GPs, no observations).

Trajectories change everywhere; every affected result is regenerated.

## 2. What did NOT change — do not "fix" these

* The ε_t observation schedule (division form, clipped) — unchanged.
* The observation protocol: n_obs=100, batches of 20 fresh rows, N_max=150,
  forced first observation / second intervention, HQCBO forced observation
  at refined-phase entry.
* Plateau constants k=5, δ=1e-3; the declared MediatedChain split.
* QDCBO / QMCBO / CEO engines are separate (QDCBO/QMCBO are fixed and rerun
  in the second work package; CEO is not rerun).

## 3. Preconditions

```bash
cd <repo> && git checkout fix/engine-v3 && git pull --ff-only
source "$HOME/venvs/ccbo/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## 4. Stage A — tests (must pass first)

```bash
PYTHONPATH=. python -m pytest -q            # expect 162+ passed, 11 skipped
PYTHONPATH=. python -m pytest -q -m slow    # expect 30+ passed; test_minimal_invariance must pass
```

## 5. Stage B — smoke

```bash
PYTHONPATH=. python scripts/run_minimal_suite.py --scms MediatedChain,ParallelParent \
  --seeds 2 --trials 12 --mediated-trials 12 --jobs 4 --outdir $SCRATCH/smoke_v3
```
Expect 64/64 OK, a `.decisions.json` next to every CSV, row-0 `cum_cost`
equal to the initial-design cost (CBO 12, QCBO 6, BOS 12, BO 6 on
ParallelParent A0; 6 on MediatedChain), HQCBO and HQCBOGF with the same
trigger and a +6 charge at the trigger row.

## 6. Stage C — the full runs

```bash
PYTHONPATH=. python scripts/run_minimal_suite.py --dry-run --seeds 30 --trials 50 --mediated-trials 60 --outdir results/v3/minimal   # 1320
PYTHONPATH=. python scripts/run_cbo_family.py    --dry-run --seeds 10 --trials 40 --outdir results/v3/family_cbo                     # 90
bash scripts/lsf/submit_minimal_v3.sh
bash scripts/lsf/submit_family_v3.sh
```
**Versioned output directories are mandatory** (`results/v3/...`): both
runners skip units whose CSV *and* sidecar exist.

## 7. Stage D — assemble

```bash
mkdir -p results/v3 && for s in ceo_minimal; do tar -C results/v3 -xzf artifacts/traces/$s.tar.gz && (cd results/v3 && sha256sum --check --quiet ../../artifacts/traces/$s.sha256); done
```
(`qdcbo` / `qmcbo` come from their own v3 reruns — second work package.)

## 8. Stage E — regenerate

```bash
PYTHONPATH=. python scripts/emit_minimal_taxonomy.py --dir results/v3/minimal
PYTHONPATH=. python scripts/plot_minimal_misspec.py  --dir results/v3/minimal
PYTHONPATH=. python scripts/plot_minimal_refine.py   --dir results/v3/minimal
PYTHONPATH=. python scripts/plot_minimal_ablations.py --dir results/v3/minimal
PYTHONPATH=. python scripts/plot_cost_indexed.py     --dir results/v3/minimal
PYTHONPATH=. python scripts/summarize_minimal_exact.py --dir results/v3/minimal
PYTHONPATH=. python scripts/emit_minimal_ablation_table.py
PYTHONPATH=. python scripts/plot_family_suite.py     --results-dir results/v3
PYTHONPATH=. python scripts/emit_paired_effects.py   --results-dir results/v3
PYTHONPATH=. python scripts/emit_cost_analysis.py    --results-dir results/v3 --ceo-dir results/v3/ceo_minimal
PYTHONPATH=. python scripts/emit_ceo_comparison.py   --ceo-dir results/v3/ceo_minimal --minimal-dir results/v3/minimal
```

## 9. Stage F — verify, freeze, archive

`scripts/verify_traces.py --results-dir results/v3` must report 0 failures
(structural + invariance checks run now; the published-aggregate checks are
behind `MINIMAL_PUB` / `FAMILY_PUB` sentinels — freeze them from the
regenerated summaries, then re-run). Repack `minimal` and `family_cbo`
(CSV + sidecars + refine_info) with the deterministic `tar` flags in
`artifacts/traces/MANIFEST.md`, refresh the `.sha256` lists and
`SHA256SUMS`, update the manifest blocks.

## 10. Stop-and-report conditions

* Any Stage A failure — above all `test_minimal_invariance`.
* Dry-run totals ≠ 1320 / 90.
* Any unit failing that a relaunch does not fix.
* BO or BOS differing across conditions; QCBO **or QCBONP** differing under a
  protected edit at full precision; a gate `error` state in any sidecar.
* Any temptation to change the ε_t schedule or to reuse `results/v2`.

## 11. Addendum 2026-09-10 — second static rerun (hash order, variance policy)

Two defects surfaced while auditing the 2026-09-09 static results
(`results/v3_prelim/{minimal,family_cbo}`, kept for comparison, never reused):

1. **Hash-seed dependence of the prior.** `ccbo/adjustment.py` sorted
   frozenset-typed C-DAG vertices with a bare `sorted()`, which is only a
   partial order. The adjustment set chosen among equal-size candidates, the
   GP input column order and the g-computation step order therefore followed
   `PYTHONHASHSEED`. On the family suite the CompleteGraph and Coral priors
   differed between interpreters (Coral joint arm mean 57 / 275 / 26 under
   three hash seeds; QCBO's C+N+O arm switched adjustment set). The minimal
   suite, QDCBO (runner pins the seed) and QMCBO were already independent.
   Fix: canonical `_vkey` ordering everywhere (`_sorted_v`,
   `lexicographical_topological_sort`); `ccbo/tests/test_hash_independence.py`
   runs the prior in two interpreters with different seeds. `PYTHONHASHSEED=0`
   is additionally exported by every LSF script and by `reproduce_paper.sh`.
2. **Epistemic-free prior variance kills the surrogate off support.** With
   `VARIANCE_POLICY="total"` the joint/D/T arms of Coral (intervention ranges
   D∈[2000,2080], T∈[2300,2400] against observational support D∈[3,7],
   T∈[4,8]) get prior mean 0 and prior variance ≤ 1e-6 (the outcome GP
   interpolates, `lik_var` at its floor, no between-row spread), while the
   discarded epistemic term is 1e4–7e7. A ~0 variance adjustment makes the
   CausalRBF kernel identically zero: the arm's GP can never move away from
   its (wrong) prior mean, EI's σ=0 branch reports a certain improvement, and
   the optimizer pulls that arm forever — the flat Coral CBO/QCBO curves of
   the preliminary run (0 improvements in 10/10 seeds, 390/400 QCBO pulls on
   the joint arm at y≈9300). The CBO reference implementation averages
   `gp.predict(rows)[1]`, i.e. it *keeps* epistemic + noise. New default
   `VARIANCE_POLICY="predictive"` = `lik_var + spread + epistemic`;
   `"total"` stays selectable (`CCBO_VARIANCE_POLICY=total`) for comparison.

Procedure (same stages as above; sidecars now record the policy and the gate
checks it):

```bash
bash scripts/lsf/submit_minimal_v3.sh                                   # results/v3/minimal    (predictive)
bash scripts/lsf/submit_family_v3.sh                                    # results/v3/family_cbo (predictive)
CCBO_VARIANCE_POLICY=total OUTDIR=$PWD/results/v3_total/family_cbo bash scripts/lsf/submit_family_v3.sh   # comparison
```

`results/v3_prelim/minimal` already *is* the minimal suite under the retired
policy (the canonical-order fix is byte-neutral there — verified on a 16-unit
probe), so no `v3_total/minimal` run is needed. Freeze `MINIMAL_PUB` /
`FAMILY_PUB` from the predictive runs only.
