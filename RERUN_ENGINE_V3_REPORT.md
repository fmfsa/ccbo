# Rerun report — engine v3 (post-review fixes)

Companion to `RERUN_ONLINE_OBS_REPORT.md` (August). Executed 2026-09-09 on the
DTU HPC (branch `fix/engine-v3`, off `rerun/online-obs-v2`). The manuscript
change-list is `MANUSCRIPT_CHANGES.md`; this file records what was run, what
was verified, and how the numbers moved.

## 1. Provenance

| | |
|---|---|
| Engine commits | WP1 `afdacea` (estimators, gradients, noise, gate, cache); WP2–4 `f2a98e0` (arms, cost axis, logs, emitters, gate); WP5 `9f25b41` (QDCBO/QMCBO) |
| Minimal / family runs | LSF 29365450 / 29365451, submitted 2026-09-09 14:50 at engine SHA `f2a98e0` |
| QDCBO / QMCBO runs | LSF 29365482 / 29365483, submitted 15:03 at engine SHA `9f25b41` |
| Host / env | `hpclogin9`, LSF `hpc` queue, 16 cores per job; `~/venvs/ccbo` (py3.10.18, numpy 1.26.4, GPy 1.13.2, emukit 0.5.1, ananke 0.5.0); `~/venvs/mcbo` (py3.9, botorch 0.7.2) for QMCBO; BLAS pinned to 1 thread |
| Third-party stacks | `third_party/DCBO` @ 85a9bdf, `third_party/mcbo` @ 0d0650e (fetched 2026-09-09, gitignored, unmodified — corrections are applied by `ccbo/qdcbo/stock_fixes.py` at import sites) |
| Results | `results/v3/{minimal,family_cbo,qdcbo,qmcbo}` (fresh) + `results/v3/ceo_minimal` (extracted from the unchanged archive, checksums OK) |

## 2. Stage A — tests

Fast suite on the final engine: **170 passed, 3 skipped** (the 3 QMCBO joint-network tests, run separately in the mcbo env: **11 passed**). Slow suite: **21 passed** — `test_minimal_invariance` included (and extended to QCBONP and BO/BOS identity). QDCBO module: **22 passed**, parametrised over stock and corrected semantics (finest-partition identity holds in both). New test files: `test_estimators`, `test_causal_kernel_gradients`, `test_causal_acquisition_gradients`, `test_id_gate_tristate`, `test_cache_keys`, `test_noise_policy`, `test_decision_log`, `test_cost_charging`, `test_hqcbo_gf`, plus additions to `test_minibench_structure`, `test_minimal_invariance`, `test_qdcbo`, `test_qmcbo`.

Key numerical checks behind the tests: `CausalRBF` input gradients agree with central differences to 1e-9 (previously `gradients_X_diag` returned zeros and `dvar/dx` had the wrong sign); the σ_f² hyperparameter gradient now matches finite differences; the oracle-conditional FrontDoor integral reproduces 0.33590 exactly where mean substitution gives 0; g-computation draws recover `x²+1` where mean propagation gives `x²`; the likelihood variance stays at 1e-10 through `optimize()`.

## 3. Smoke and grid

Smoke (2 seeds, 12 trials): 64 MediatedChain+ParallelParent units and 24 FrontDoor units OK, sidecars present, row-0 costs as designed (CBO 12, QCBO 6, BOS 12, BO 6 on PP; 6 on MC), HQCBO and HQCBOGF sharing trigger 7 with a +6 split charge; family ToyGraph 3 units OK (row 0 = 20).

| Suite | Dry-run | Completed | FAIL |
|---|---|---|---|
| minimal (`run_minimal_suite.py`, 30 seeds) | 1320 (PP 720, FD 360, MC 240) | **1320/1320** | 0 |
| family (`run_cbo_family.py`, 10 seeds) | 90 | «pending» | — |
| qdcbo (E1 + E2, 20 seeds) | 240 | «pending» | — |
| qmcbo (E1 + E2, 20 seeds) | 160 | «pending» | — |

## 4. Verification gate (`scripts/verify_traces.py --results-dir results/v3`)

Minimal section: grid, sidecars, row-0 = charged initial design (3·Σ|arm|), observe caps (BO/BOS 0, others ≤ 3), no identification-gate error in any unit, split design charged at the trigger, HQCBO/HQCBOGF triggers equal, BO and BOS byte-identical across every condition, **QCBO and QCBONP identical at full decision-log precision under A1, A2, B1 (QCBONP also under A3)**, A3 control sup|Δ| = 21.50 for QCBO — all PASS; frozen published aggregates all PASS. Family / qdcbo / qmcbo sections: «pending».

## 5. Old (v2, 2026-08-21) vs new (v3) — MinimalBench

| Quantity | v2 | v3 |
|---|---|---|
| Table 1 sup|Δ| QCBO A1 / A2 / A3 / B1 | 0 / 0 / 21.6 / 0 | 0 / 0 / **21.5** / 0 |
| Table 1 sup|Δ| BO (all rows) | 0 | 0 |
| PP A0 CBO final / R50 | 0.0014±0.0003 / 4.17±0.37 | **0.0002±0.0001 / 3.61±0.36** |
| PP A0 QCBO final / R50 | 0.0021±0.0004 / 7.28±1.04 | **0.0002±0.0001 / 6.82±1.03** |
| PP A1 CBO final / R50 | 4.2500 / 216.16±1.12 | 4.2500 / **216.10±1.09** |
| PP A3 CBO / QCBO final | 0.0354 / 0.0005 | 0.0354 / 0.0005 (unchanged: constant-fallback arms) |
| FD B0 CBO final | 0.00011±0.00008 | **0.00003±0.00001** |
| FD B1 CBO final | 0.0019±0.0007 | **0.0039±0.0022** |
| FD paired ΔR50 (CBO) | 6.18±1.06 | **5.91±1.06** [3.73, 8.08] |
| FD B0 QCBO final / R50 | 0.0015 / 11.18 | 0.0015 / 11.18 (unchanged: constant-fallback arm) |
| MC finals CBO / QCBO / HQCBO | 0.040 / 4.000 / 0.040 | 0.040 / 4.000 / 0.040; **HQCBOGF 0.040** |
| MC R60 CBO / QCBO / HQCBO | 5.98±1.88 / 243.19±0.90 / 36.76±1.23 | **5.79±1.78 / 243.15±0.90 / 35.34±1.00**; HQCBOGF **37.61±1.37** |
| HQCBO trigger (n=30) | 7.3±0.10 | **7.0±0.0** (every seed triggers at trial 7) |
| HQCBOGF − HQCBO (paired) | — | Δfinal +3.5e-7 (≈0); ΔR60 **+2.27 [+0.90, +3.64]** |

Arms whose priors are all on the constant fallback (FD QCBO; PP A3 CBO/QCBO) are unchanged between v2 and v3 — exactly as expected, since the gradient defect only affected non-constant causal priors.

New ablation results (v3 only; final ± se, R_T):

| Arm | PP A0 | FD B0 | MC C0 |
|---|---|---|---|
| CBONP (same arms, no prior) | 0.0004±0.0001, 23.03±2.61 | 0.0011±0.0003, 8.17±0.92 | 0.040, 8.44±1.96 |
| QCBONP | 0.0006±0.0002, 19.09±2.17 | 0.0009±0.0003, 15.63±1.89 | 4.000, 245.32±1.11 |
| BOS (all subsets, no prior, no obs) | 0.0005±0.0002, 17.49±2.08 | 0.0013±0.0004, 12.58±1.53 | 0.040, 12.00±2.76 |
| BO (joint arm) | 0.0037±0.0035, 5.83±1.23 | 0.40±0.15, 30.62±6.59 | 4.008±0.008, 238.55±0.85 |

Readings: the causal prior is worth 2.6 [1.5, 3.8] regret units to CBO and 2.2 [1.0, 3.4] to QCBO on MediatedChain, and cuts PP A0 CBO regret from 23.0 (no prior) to 3.6; CBONP is *identical* under FrontDoor B0/B1 (ΔR50 = 0), so the 5.9-unit paired increase is entirely the price of a corrupted prior; BOS owns the {X1} arm and reaches y* on MediatedChain without any graph (R60 12.0 vs CBO 5.8), while BO's joint arm sits on the fixed-partition floor.

## 6. Protected-pair invariance — explicit statement

**No protected-pair invariance broke.** On all 30 seeds, QCBO's and QCBONP's decision logs (observe/intervene draws, arms, intervention values at full double precision, outcomes, incumbents) are identical under A1, A2 and B1; QCBONP is additionally identical under A3 (the A3 edit changes only the quotient prior, which QCBONP does not use); QCBO diverges under A3 (sup|Δ| = 21.50). BO and BOS are byte-identical across all conditions of each SCM.

## 7. Family, QDCBO, QMCBO

«pending — filled when LSF 29365451 / 29365482 / 29365483 complete»

## 8. What changed in `verify_traces.py`

Grid 660 → 1320 with sidecars; row-0 init-cost check; split-charge and shared-trigger checks; BOS added to byte-identity; QCBONP added to the invariance checks (plus its A3 invariance); full-precision decision-log signatures compared in addition to the CSV columns; gate-error audit; QDCBO/QMCBO sidecar presence, no-stock-quirks audit and decision-level E2 invariance; all published constants behind `None` sentinels until frozen (MINIMAL_PUB frozen 2026-09-09). No check was weakened.
