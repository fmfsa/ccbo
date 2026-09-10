# Rerun report — online observation protocol fix (+ BO baseline)

Per `RERUN_ONLINE_OBS.md` §12. Executed 2026-08-21 on the DTU HPC cluster.

## 1. Provenance

| | |
|---|---|
| Commit run | `d23022e` 2026-08-21 09:23:13 +0200 "Fix online observation protocol; add BO baseline; runbook for the rerun" |
| Host | `hpclogin9.hpccluster.dtu.dk` (login node); compute via LSF `hpc` queue, 16 cores/job |
| Date | 2026-08-21 (submitted 10:03, minimal done ~10:40, family done ~12:10) |
| Python | 3.10.18 (`~/venvs/ccbo`) |
| Key packages | numpy 1.26.4, pandas 1.5.3, scipy 1.12.0, GPy 1.13.2, emukit 0.5.1, paramz 0.9.6, ananke-causal 0.5.0 |
| BLAS pinning | `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on every run |
| Working tree | clean before pull; `git pull --ff-only` fast-forwarded b8fd9a6 → d23022e with no local modifications |

## 2. Stage A — tests

Fast suite (`PYTHONPATH=. python -m pytest -q`):

```
106 passed, 11 skipped, 14 deselected, 1341 warnings in 192.93s (0:03:12)
```

Slow suite (`PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest -q -m slow`):

```
14 passed, 117 deselected, 26346 warnings in 215.88s (0:03:35)
```

Both match the reference machine's expected counts exactly (106/11 fast, 14
slow). **`ccbo/tests/test_minimal_invariance.py` passed**, including
`test_qcbo_diverges_under_quotient_visible_edit`. No stop condition was hit.

Stage B smoke check (MediatedChain, 2 seeds, 12 trials, `/tmp/obs_smoke`):
8/8 units OK; every CSV had 13 rows, monotone aligned
`trial`/`best_y`/`cum_cost`, all four arms (BO/CBO/QCBO/HQCBO) present, BO
with zero observe rows and the causal arms with 1–3. Directory discarded.

## 3. Grid totals and completions

| Runner | Dry-run total | Completed | FAIL lines |
|---|---|---|---|
| `run_minimal_suite.py` (seeds 30, T 50/60) | 660 (450 causal + 210 BO) | **660/660 units OK** (LSF 29163301) | none |
| `run_cbo_family.py` (seeds 10, T 40) | 90 (60 + 30 BO) | **90/90 units OK** (LSF 29163302) | none |

Dry-run totals matched the runbook's expected 660/90 exactly. No unit
failed; no relaunch was needed. Outputs went to fresh `results/v2/minimal`
and `results/v2/family_cbo` (never the old `results/minimal` /
`results/family_cbo`).

Stage D: `qdcbo`, `qmcbo`, `ceo_minimal` extracted from
`artifacts/traces/*.tar.gz` into `results/v2/` with every per-file SHA-256
verified — those suites are provably the archived traces, unchanged.

## 4. Old vs. new headline numbers

### Table 1 (`paper/tables/minimal_taxonomy.tex`) — sup|Δ| per condition

| Cond. | QCBO old | QCBO new | BO old | BO new |
|---|---|---|---|---|
| A1 (PP) | 0 | **0** | — (arm did not exist) | **0** |
| A2 (PP) | 0 | **0** | — | **0** |
| A3 (PP, negative control) | 21.5 | **21.6** | — | **0** |
| B1 (FD) | 0 | **0** | — | **0** |

The BO column reads 0 on every row, as required: BO CSVs are byte-identical
across every misspecification condition within an SCM (verified per seed on
PP A0–A3 and FD B0–B1) — the harness did not leak the perturbation.

### Table 2 (`paper/tables/family_effects.tex`) — paired effects

CBO-family rows (rerun; base/quotient finals, paired quotient disadvantage):

| Task | Old | New |
|---|---|---|
| Toy ↓ | −2.16±0.00 / −2.13±0.03; **+0.03** [−0.04, +0.11] | −2.16±0.00 / −2.17±0.01; **−0.005** [−0.016, +0.007] |
| Synthetic ↓ | −3.14±0.19 / −1.25±0.07; **+1.89** [+1.40, +2.39] | −3.47±0.04 / −1.28±0.03; **+2.19** [+2.06, +2.31] |
| Coral ↓ | 36.08±0.02 / 36.42±0.01; **+0.34** [+0.30, +0.37] | 36.07±0.02 / 36.45±0.02; **+0.38** [+0.33, +0.44] |

Notable: the Toy paired effect changes sign (small QCBO edge instead of a
small QCBO deficit) but remains a statistical tie (CI spans 0;
worse/tie/better = 7/0/3 seeds). Synthetic and Coral sharpen in the same
direction as before. DCBO/QDCBO and MCBO/QMCBO rows are identical to the old
table (archived traces fed through unchanged): stat −0.29 [−0.41, −0.17],
ind −2.45 [−2.72, −2.19], nonstat −1.69 [−3.57, +0.18], MCBO ToyGraph −0.77
[−1.04, −0.50], PSAGraph +0.0001 [−0.0004, +0.0006].

### Final incumbents behind Figs 2–4 (mean over seeds)

Minimal suite (Figs 2–3):

| Group | Old final | New final |
|---|---|---|
| PP A0 CBO / QCBO | 0.0016 / 0.0021 | 0.0014 / 0.0021 |
| PP A1 CBO (misspec fine prior) | 4.2500 | 4.2500 |
| PP A2 CBO | 0.0016 | 0.0014 |
| PP A3 CBO / QCBO | 0.1441 / 0.0004 | 0.0354 / 0.0005 |
| PP BO (any cond) | — | 0.0017 |
| FD B0 CBO / QCBO | 0.0003 / 0.0012 | 0.0001 / 0.0015 |
| FD B1 CBO | 0.0035 | 0.0019 |
| FD BO (any cond) | — | 0.5994 |
| MC C0 CBO / QCBO / HQCBO | 0.0400 / 4.0000 / 0.0400 | 0.0400 / 4.0000 / 0.0400 |
| MC BO | — | 4.0034 |

Derived headline quantities:

| Quantity | Old | New |
|---|---|---|
| FD paired ΔR50 (B1−B0, CBO) | 5.91 ± 1.06 | **6.18 ± 1.06** |
| MC R60: CBO / QCBO / HQCBO | 5.46±1.61 / 243.23±0.90 / 36.97±1.18 | 5.98±1.88 / 243.19±0.90 / 36.76±1.23 |
| HQCBO trigger trial (mean±se, n=30) | 7.4 ± 0.14 | **7.3 ± 0.10** (all 30 splits accepted) |
| PP A0-vs-A3 QCBO sup|Δ| (neg. control) | 21.5 | 21.56 |

Family suite (Fig 4 top; new BO arm included):

| Task | BO | CBO old → new | QCBO old → new |
|---|---|---|---|
| ToyGraph | −2.17 | −2.16 → −2.16 | −2.13 → −2.17 |
| CompleteGraph | −0.63 | −3.14 → −3.47 | −1.25 → −1.28 |
| SimplifiedCoral | 9279.47 | 36.08 → 36.07 | 36.42 → 36.45 |

Fig 4 middle/bottom (QDCBO, QMCBO): unchanged — regenerated from the
verified archives.

CEO comparison (`paper/tables/ceo_minimal.tex`): CEO columns unchanged (CEO
was not rerun); the CBO/QCBO reference columns refreshed — PP A1 CBO
4.245±0.003 → 4.250±0.000, PP A3 CBO 0.002 → 0.035, FD B1 CBO 0.006 → 0.002.
The Δ(CEO−QCBO) range on protected conditions remains +0.07 to +1.00, so the
manuscript's "better by 0.07–1.00" sentence happens to survive.

Cost accounting (`paper/tables/cost_accounting.tex`): totals shift a few
cost units because the causal arms now spend up to 3 trials observing
(e.g. FD B0 C\* 52.77 → 53.20; family Toy C\* 51.5 → 57.0; family Synthetic
C\* 95.5 → 99.4). The two structural claims survive: Coral CBO's 31-arm
initialization still costs 800 > QCBO's entire total (~217), and on
FrontDoor fine CBO still reaches the well within a smaller total budget
(53.2 vs 100.4).

## 5. Protected-pair invariance — explicit statement

**No protected-pair invariance broke.** On the rerun traces, for every one
of the 30 seeds:

- ParallelParent A0-vs-A1 QCBO: sup|Δ| = 0.0, arm/x decisions identical;
- ParallelParent A0-vs-A2 QCBO: sup|Δ| = 0.0, decisions identical;
- FrontDoor B0-vs-B1 QCBO: sup|Δ| = 0.0, decisions identical;
- the quotient-visible negative control A0-vs-A3 diverges as predicted
  (sup|Δ| = 21.56, decisions differ).

The paper's central claim holds under the corrected online-observation
protocol, now non-vacuously: observe trials genuinely refresh the prior and
the trajectories still coincide exactly on quotient-redundant edits.

## 6. Observation-protocol diagnostics

Observe actions per unit (protocol: n_obs=100, 20 fresh rows per observe,
N_max=150 ⇒ at most ⌈50/20⌉ = 3 observes, final batch truncated to 10):

| Suite / arm | Distribution of observe counts | Budget exhausted (3 observes) |
|---|---|---|
| minimal BO (210 units) | {0: 210} | 0% (by construction) |
| minimal CBO (210) | {1: 4, 2: 6, 3: 200} | 95% |
| minimal QCBO (210) | {1: 4, 2: 5, 3: 201} | 96% |
| minimal HQCBO (30) | {2: 1, 3: 29} | 97% |
| family BO (30) | {0: 30} | 0% |
| family CBO (30) | {1: 10, 3: 20} | 67% |
| family QCBO (30) | {1: 10, 3: 20} | 67% |

(Family observe trials identified as zero-cost-increment rows; the 10
1-observe units are ToyGraph's.) No unit anywhere exceeded 3 observe
actions, i.e. every run intervened once `N_t = N_cap` — the forced-
intervention rule held, including HQCBO's forced observation at the refined
phase (skipped when the budget was spent; no HQCBO unit exceeded the cap).

No-row-revealed-twice: guaranteed structurally by the observation-pool
cursor (each observe consumes `pool[cursor:cursor+batch]` and advances the
cursor, which persists across resume and the HQCBO phase boundary) and
asserted by the Stage A unit tests (`test_observation_pool.py`,
`test_observation_policy.py`, `test_resume_equivalence.py`,
`test_prior_refresh_on_observe.py` — all passing). The traces are
consistent with it: observe counts never exceed the fresh-row budget, and
BO (which never observes) is byte-identical across conditions, ruling out
pool cross-contamination between units.

The ε_t schedule was left exactly as vendored (division form, clipped), per
runbook §2. No change was made or attempted.

## 7. Changes to `scripts/verify_traces.py`

All changes update the gate to the new reality; no check was weakened:

1. **Minimal grid 450 → 660**: `minimal_units()` now includes the BO arm on
   every condition (the runbook's new-arm spec). Grid-completeness message
   updated accordingly.
2. **New check — BO byte-identity**: per seed, BO CSVs must be byte-identical
   across PP A0–A3 and across FD B0–B1 (the "free harness check" of runbook
   §4; the basis of Table 1's all-zero BO column).
3. **New check — observe budget cap**: every causal-arm unit ≤ 3 observe
   actions; every BO unit exactly 0. Encodes protocol §3's forced-
   intervention rule at the trace level.
4. **Refreshed published aggregates** (the gate compares recomputed values
   to the paper's displayed numbers, which this rerun redefines):
   FD B0 CBO final 0.0003 → 0.0001; FD B1 0.0035 → 0.0019; paired ΔR50
   5.9 → 6.2; HQCBO trigger mean 7.4 → 7.3. MC finals (0.040/4.000/0.040)
   unchanged.
5. **Family grid 60 → 90** with the BO arm, and published finals updated to
   the rerun values (BO −2.17/−0.63/9279.47, CBO −2.16/−3.47/36.07, QCBO
   −2.17/−1.28/36.45).

QDCBO/QMCBO/CEO blocks untouched (suites unchanged). Result on
`results/v2`: **61 checks, 0 failures**; identical result on a fresh
extraction of the refreshed archives, whose per-file SHA-256 lists all
verify.

## 8. Stale in-text numbers in `paper/qcbo_aistats.tex`

Line numbers refer to the manuscript of record at `d23022e` (1569 lines).
**Prose is untouched per the runbook — this is the authors' checklist.**
Tables and figures under `paper/tables/` / `paper/figures/` are already
regenerated and committed.

Numbers that are now WRONG in prose:

| Line | Printed | Correct after rerun |
|---|---|---|
| 690 | "Fine CBO ends at mean $4.245$" (PP A1) | **4.250** (se 0.000); the "Monte Carlo and finite-budget optimization error" sentence at line 692 loses its object — the measured value now sits on the analytic floor 4.25 exactly at 3 dp |
| 714 | "$\Delta R_{50}=6.1\pm1.0$" (FD paired) | **6.2 ± 1.1** |
| 715 | "similar final value ($0.006$ versus $0.001$)" (FD B1 vs B0 CBO) | **0.002 versus 0.000** at the same 3-dp convention (raw 0.0019 vs 0.0001) — authors may prefer to re-precision this sentence |
| 729 | "HQCBO refines at trial $7.5\pm0.1$" | **7.3 ± 0.1** |
| 732–733 | "$R_{60}=5.60\pm1.69$ (CBO), $243.23\pm0.91$ (QCBO), $37.22\pm1.21$ (HQCBO)" | **5.98 ± 1.88, 243.19 ± 0.90, 36.76 ± 1.23** (analytic floor $T\Delta(\Pi)=237.6$ at line 733 is unchanged, but the printed pairing must be refreshed) |
| 764–765 | "CEO 0.77 versus committed CBO's $4.245$" (A1) | CEO value stands; CBO reference becomes **4.250** |
| 754–757 | "…tie on the remaining tasks" (mixed sentence) | Toy is still a statistical tie but the point estimate flips sign (−0.005, CI spans 0); wording survives, the sign of any quoted point estimate does not |

Numbers/claims that are results-derived and must be RE-VERIFIED by the
authors but happen to hold on the new traces:

| Line | Claim | Status on new traces |
|---|---|---|
| 696, 716, 1440–1442 | "$\sup|\Delta|=0$" on protected conditions (A1, B1, whole-suite) | Holds exactly (this report §5) |
| 696–698 | A3 changes the QCBO trajectory | Holds (sup|Δ| = 21.6, table value 21.5 → 21.6) |
| 728 | MC finals "$0.040$, $4.000$, $0.040$" | Unchanged |
| 766 | CEO "better by $0.07$–$1.00$" on protected conditions | Range unchanged on refreshed table |
| 768–770 | "($0.135$ versus $4.000$) … HQCBO reaches $0.040$" | Unchanged |
| 758–760, 1463–1467 | Coral: CBO's 31-arm init (800 cost units) exceeds QCBO's total budget | Holds (800 > ~217) |
| 1467–1469 | FrontDoor: fine CBO reaches the well within a smaller total budget | Holds (53.2 vs 100.4; note C\* moved 52.8 → 53.2) |
| 1463–1465 | Synthetic: CBO's advantage disappears at matched total cost | Direction re-confirmed by regenerated `cost_accounting.tex`; C\* moved 95.5 → 99.4 — authors should re-read the matched-budget row before keeping the sentence |
| 1422–1423 (footnote) | "$\epsilon_t\approx0.11$ and $0.05$" (division vs multiplication on PP at $N_t{=}100$) | Values derive from the shared observational pool, which was not regenerated; still, re-check at press time since the footnote is the disclosure of the deliberately-kept division schedule |
| 1185–1186, 1209–1210 | uninformative-tier statements (A3; FD every condition) | Re-confirmed in run logs (`[uninformative: …]` markers present) |
| 1292–1301 | HQCBO prefix identity; forced observation at refined phase | Consistent with traces and passing slow tests |
| 1196–1198 | A2/B1 "corruption measurably costs nothing" contrast | Holds (A2 ≈ A0 CBO finals 0.0014; B1 pays ΔR50 6.2) |

Qualitative sentences bound to regenerated tables/figures (lines 701, 737,
754–760, 1472–1477, 1497–1500, 1547–1549): re-read against the committed
new versions; the table/figure files themselves are already updated.

Not affected (do not touch): all QDCBO/QMCBO numbers (lines 752–753, 1526–
1534), protocol constants (n_obs=100, batch 20, N_max=150, seeds/trials/
costs, tolerances), and analytic values (4.25, 3.96, 237.6, 0.04, etc.).
Older sibling drafts (`qcbo_aistats_source/reordered/restructured/
scaffold.tex`) still carry previous-generation numbers throughout and were
deliberately left alone.

## 9. Addendum (post-review): BO in Figure 4, and why BO collapses on Coral

Review of the first push caught that `plot_family_suite.py` still loaded
only CBO/QCBO in its top row, so the 30 new family BO traces were run but
never drawn. Fixed: BO is now plotted in the Toy and Synthetic panels
(grey dotted), and the Coral panel carries an off-axis annotation
("BO ≈ 9.28×10³") instead of a curve. The same pass fixed the Fig 3
trigger annotation, which rounded the mean trigger to "7" (`.0f`); it now
prints 7.3 (`.1f`).

**Why BO ends at ~9279 on Coral — verified, not a bug.** The benchmark's
declared intervention ranges for SimplifiedCoralGraph put T in [2300, 2400]
and D in [2000, 2080], while the SEM's natural regime is T ≈ 4–8 and
D ≈ 3–7 (observational data). A causal arm can intervene on N alone and
leave T and D at natural values (Y ≈ 36); BO's arm is by definition a joint
intervention on all five manipulable variables, so every BO query must
clamp T and D into those off-distribution boxes. Monte-Carlo evaluation of
the SEM over the joint box gives a floor of ≈ 9273.9 (best corner:
C=0.3, D=2000, N=−2, O=3, T=2300; 400-point random search never beats
9288.7). BO's recorded finals are min 9274.2 / mean 9279.5 ± 2.0 — i.e.
**BO found the optimum of its own feasible set to within ~0.3**. The
250× gap versus CBO/QCBO is structural (the price of being forced to act
on every variable), not an optimizer or harness failure, and is why the
Coral panel annotates BO rather than flattening the axis. Toy and
Synthetic behave oppositely and are plotted normally: on Toy (2 manipulable
variables, benign ranges) BO matches the causal arms (−2.17); on Synthetic
it lands mid-field (−0.63 vs CBO's −3.47).

Not addressed here, deliberately: the archived MCBO ToyGraph plateau
(17/20 seeds flat from trial 0) predates this rerun, is recorded as an
open instrumented-run TODO in `RUN_TODO.md`, and QMCBO reruns are out of
scope per runbook §2.

## 10. Deviations from the runbook

None affecting results. Two operational notes: (i) Stage D extraction was
performed while Stage C was still running (it touches only the unaffected
archived suites); (ii) an interim WIP commit (`c739671`) pushed the
minimal-suite artifacts before the family job finished, at the user's
request. Everything else followed the runbook order, and no stop-and-report
condition in §14 was triggered.
