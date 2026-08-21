# Rerun runbook — online observation protocol fix (+ BO baseline)

**For the agent running on the DTU HPC (P1) node.** Self-contained: everything
you need is here. Read it fully before starting.

---

## 1. Why this rerun exists

The vendored CBO loop (`ccbo/cbo/cbo.py`) is supposed to run an ε-greedy
observe/intervene protocol where each *observe* trial reveals a fresh batch of
previously unseen observational rows that then refresh the causal prior. Three
defects meant it did not:

1. **Every observation returned the same rows.** `observe()` sliced the constant
   `full_obs[initial : initial + batch]`, so with the runners' settings every
   observe trial re-appended `full_obs[100:120]`. No cursor, no budget guard, no
   truncated final batch; `max_N` was used only in the ε denominator, never as a
   pool cap.
2. **Observations never reached the prior.** `CoarsenedGraph.refit_models` only
   delegated — it never updated `_obs_samples` nor cleared the memoized
   `_do_cache`, and the do-estimators from `make_cdag_do_function` close over the
   frame they were built with and *ignore* the `obs` argument they are handed. On
   the CoarsenedGraph path (every paper result) the estimator was frozen at the
   initial 100 rows for the whole run. The fine graphs compounded it: their
   `refit_models` read the passed frame but never wrote back `self.<col>`, so a
   later `fit_all_models()` silently reverted to the initial rows.
3. **Point-wise prior caches were never cleared.** `x_dict_mean` / `x_dict_var`
   are keyed by `str(x)` and lived for the whole run, so even a correctly rebuilt
   estimator returned stale values at any already-queried point.

Net effect: observe trials consumed a trial slot and moved ε (via `N_t`) but were
otherwise **no-ops on the prior**. HQCBO additionally lost phase 1's observations,
because the refined graph was rebuilt from the original 100 rows.

All three are fixed. Trajectories change, so every affected result must be
regenerated.

## 2. What did NOT change — do not "fix" these

* **The ε_t schedule is unchanged and must stay unchanged.** It is the *division*
  form inherited verbatim from the vendored Aglietti et al. (2020) implementation:

  `ε_t = clip[ (Vol(C(D_t^O)) / Vol(D(X))) / (N_t / N_cap), 0, 1 ]`

  This differs from the multiplication-based expression printed in the CBO paper.
  That discrepancy is deliberate, measured (ParallelParent at N=100: 0.114 vs
  0.051), and disclosed in a footnote in `paper/qcbo_aistats.tex`. The only
  addition is the clip, which is behaviour-neutral because `u ~ U[0,1)`.
  **Do not change it to multiplication.** A multiplication variant is a possible
  future sensitivity analysis, not the primary policy.
* **HQCBO still forces an observation at the start of the refined phase.** It now
  consumes the *next unseen* batch, and is skipped when the budget is spent. The
  forcing itself is retained for fidelity to the vendored loop.
* **QDCBO and QMCBO use separate engines** (`ccbo/qdcbo/`, `ccbo/qmcbo/`) and are
  **not** affected. Do not rerun them. Their archived traces feed the combined
  tables unchanged.

## 3. The protocol, stated exactly

* `n_obs = 100` initial observational rows, one shared pool per SCM.
* Each observe action reveals **20 fresh, previously unseen** rows.
* Budget `N_max = 150`, effective cap `N_cap = min(N_max, |pool|)`; at most 50
  rows are acquired online. The final batch is truncated.
* Once `N_t = N_cap` the optimizer **must** intervene — forced or not.
* The observation cursor and the observational frame persist across resume and
  across the HQCBO phase boundary.
* Every observation refreshes the identified functionals on the enlarged dataset
  and rebuilds every arm GP — exactly once (not twice).

## 4. New arm: plain BO

`NonCausal_BO` now runs as a baseline arm on the CBO-vs-QCBO experiments: one
joint arm over all manipulable variables, no causal prior, **no observation
actions**. Two properties to preserve when reading results:

* BO reads no graph structure, so its trajectory is **identical across every
  misspecification condition** of an SCM. That is intentional — it is the
  "invariance for free, no causal prior" reference. On ParallelParent it runs
  4× (A0–A3) and on FrontDoor 2× (B0–B1) producing byte-identical CSVs; that
  redundancy is a **free harness check**. If `ParallelParent_A0_BO_seed*.csv`
  and `ParallelParent_A1_BO_seed*.csv` ever differ, the harness leaked the
  perturbation — stop and report it.
* BO intervenes on every manipulable variable each trial, so under unit
  per-variable costs it pays `|X|` per trial against a causal method's per-arm
  cost. Its curve sits differently on the cost axis by construction.

---

## 5. Preconditions

```bash
cd <repo>
git fetch origin && git checkout main && git pull --ff-only origin main
git log -1 --format='%h %ci %s'          # record this in your report
source "$HOME/venvs/ccbo/bin/activate"
```

If `git pull` reports local modifications, **stop and report** — do not stash or
discard; the machine may hold results that exist nowhere else.

Environment of record: `envs/README.md` (Python 3.10, `numpy<2`, `ananke-causal`
0.5.0). Pin BLAS threads for every run:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## 6. Stage A — tests (must pass before anything else)

```bash
PYTHONPATH=. python -m pytest -q
```
```bash
PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest -q -m slow
```

Expected on the reference machine: **106 passed, 11 skipped, 14 deselected**
fast; **14 passed** slow. Skips depend on optional third-party stacks, so compare
like-for-like rather than demanding the exact skip count.

The slow set includes `test_minimal_invariance.py` — the scientific guard. It
asserts that quotient-redundant perturbations (A1, A2, B1) leave the QCBO
arm/value sequence *and* the observe/intervene decisions identical. **If it
fails, stop.** It means the fix leaked fine-graph information into the prior;
nothing downstream is worth running.

## 7. Stage B — smoke check the protocol

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 python scripts/run_minimal_suite.py \
  --scms MediatedChain --seeds 2 --trials 12 --mediated-trials 12 \
  --jobs 2 --outdir /tmp/obs_smoke
```

Then confirm on the CSVs: `trial`/`best_y`/`cum_cost` monotone and aligned, 13
rows per unit, and `ParallelParent`-style BO/CBO/QCBO/HQCBO units all present.
Discard `/tmp/obs_smoke` afterwards.

## 8. Stage C — the full runs

**Confirm the grid from the runner, not from this document:**

```bash
PYTHONPATH=. python scripts/run_minimal_suite.py --dry-run \
  --seeds 30 --trials 50 --mediated-trials 60 --outdir results/v2/minimal
```
```bash
PYTHONPATH=. python scripts/run_cbo_family.py --dry-run \
  --seeds 10 --trials 40 --outdir results/v2/family_cbo
```

Expected **660** minimal units (450 causal + 210 BO) and **90** family units
(60 + 30 BO). If the printed totals differ, report before submitting.

Then submit:

```bash
bash scripts/lsf/submit_minimal_v2.sh
```
```bash
bash scripts/lsf/submit_family_v2.sh
```

> **Versioned output directories are mandatory.** Both runners skip any unit
> whose CSV already exists with enough rows. Writing into `results/minimal` or
> `results/family_cbo` would silently preserve the *old, wrong* results and
> report success. Use `results/v2/...` as the submit scripts do.

Every unit must finish. The runners report `FAIL <tag> :: <error>` per unit and a
final `N/M units OK` line. Re-launching is safe and resumes.

## 9. Stage D — assemble the combined workdir

The affected suites come from the new runs; the unaffected ones come from the
archives, so they are provably unchanged.

```bash
mkdir -p results/v2
for suite in qdcbo qmcbo ceo_minimal; do
    tar -C results/v2 -xzf artifacts/traces/$suite.tar.gz
    (cd results/v2 && sha256sum --check --quiet ../../artifacts/traces/$suite.sha256)
done
```

`results/v2/` must then hold: `minimal/`, `family_cbo/` (new) and `qdcbo/`,
`qmcbo/`, `ceo_minimal/` (extracted, unchanged).

## 10. Stage E — regenerate summaries, tables, figures

```bash
PYTHONPATH=. python scripts/emit_minimal_taxonomy.py --dir results/v2/minimal
PYTHONPATH=. python scripts/plot_minimal_misspec.py  --dir results/v2/minimal
PYTHONPATH=. python scripts/plot_minimal_refine.py   --dir results/v2/minimal
PYTHONPATH=. python scripts/summarize_minimal_exact.py --dir results/v2/minimal
PYTHONPATH=. python scripts/plot_family_suite.py     --results-dir results/v2
PYTHONPATH=. python scripts/emit_paired_effects.py   --results-dir results/v2
PYTHONPATH=. python scripts/emit_cost_analysis.py    --results-dir results/v2 \
                                                     --ceo-dir results/v2/ceo_minimal
PYTHONPATH=. python scripts/emit_ceo_comparison.py   --ceo-dir results/v2/ceo_minimal \
                                                     --minimal-dir results/v2/minimal
```

Use `scripts/emit_paired_effects.py` for the family table.
`scripts/emit_family_price_table.py` is **deprecated by its own docstring** (its
unpaired layout hid the seed pairing) — do not use it.

Table 1 (`paper/tables/minimal_taxonomy.tex`) now carries two `sup|Δ|` columns,
QCBO and BO. **The BO column must read `0` on every row.** A non-zero entry is a
harness bug, not a finding — report it.

## 11. Stage F — re-archive and verify

Re-archive `minimal` and `family_cbo` from `results/v2/`, refresh
`artifacts/traces/{minimal,family_cbo}.sha256` and `SHA256SUMS`, and update the
per-suite blocks in `artifacts/traces/MANIFEST.md` (grid size, generator command,
date, provenance — note the new 660/90 grids and the BO arm). Then:

```bash
PYTHONPATH=. python scripts/verify_traces.py --results-dir results/v2
```

`verify_traces.py` re-checks the expected file grid, recomputes every published
aggregate at the paper's displayed precision, and re-tests the exact-invariance
pairs. It hard-codes protocol constants and the old file grid in places
(e.g. `n_obs == 100`, `ninit == 3` around line 295) and **will need updating for
the BO arm and the new grid**. Update it to match reality — do not weaken a check
to make it pass, and say in your report exactly what you changed and why.

## 12. Report back — do not edit the manuscript prose

Write `RERUN_ONLINE_OBS_REPORT.md` and commit it on a branch (see §13). Include:

1. Commit SHA run, host, date, Python/package versions.
2. Stage A results verbatim (fast + slow counts); explicitly confirm
   `test_minimal_invariance` passed.
3. Dry-run grid totals vs. units actually completed; every `FAIL` line, if any.
4. Old vs. new headline numbers side by side: Table 1 `sup|Δ|` per condition (both
   columns), Table 2 paired effects, and the final incumbents behind Figs 2–4.
5. **Whether any protected-pair invariance broke.** This is the paper's central
   claim; state it explicitly either way.
6. Observation-protocol diagnostics from the trial logs: distribution of observe
   actions per run, how often the budget was exhausted, and confirmation that no
   run revealed a row twice.
7. Anything you changed in `verify_traces.py`, with justification.
8. A list of every in-text number in `paper/qcbo_aistats.tex` that is now stale.

**List the stale numbers; do not rewrite the manuscript prose.** Table/figure
files are regenerated by the emitters and may be committed. Narrative claims are
for the authors.

## 13. Committing

Work on a branch, never on `main`:

```bash
git checkout -b rerun/online-obs-v2
```

Commit: regenerated `paper/tables/*`, `paper/figures/*`, `artifacts/summaries/*`,
`artifacts/traces/*`, any `verify_traces.py` fix, and the report. Do **not**
commit `results/v2/` itself — `results/` is gitignored and the archives are the
tracked artifact. Open a PR; do not merge it.

## 14. Stop-and-report conditions

Stop and report rather than working around any of these:

* `git pull` blocked by local modifications on the server.
* Any Stage A test failing — above all `test_minimal_invariance`.
* Dry-run grid totals differing from 660 / 90.
* Any unit failing that a relaunch does not fix.
* The BO `sup|Δ|` column non-zero, or per-condition BO CSVs differing within an
  SCM.
* A protected-pair invariance breaking.
* Any temptation to change the ε_t schedule, to reuse `results/minimal` or
  `results/family_cbo`, or to weaken a `verify_traces.py` check.
