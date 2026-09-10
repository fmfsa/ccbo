# Rerun report — engine v3 (post-review fixes)

Companion to `RERUN_ONLINE_OBS_REPORT.md` (August). Executed 2026-09-09/10 on
the DTU HPC (branch `fix/engine-v3`, off `rerun/online-obs-v2`). The manuscript
change-list is `MANUSCRIPT_CHANGES.md`; this file records what was run, what
was verified, and how the numbers moved. **Read §9 first**: the static suites
(minimal, family) were rerun on 2026-09-10 with engine v3.1 (`f75a19c`) after
two defects were found in the 2026-09-09 run; the numbers below are from the
final run, with the retired preliminary run shown for comparison where it
differs.

## 1. Provenance

| | |
|---|---|
| Engine commits | WP1 `afdacea` (estimators, gradients, noise, gate, cache); WP2–4 `f2a98e0` (arms, cost axis, logs, emitters, gate); WP5 `9f25b41` (QDCBO/QMCBO); **v3.1 `f75a19c`** (hash-independent identification order, predictive prior variance; §9) |
| Minimal / family runs | **LSF 29367764 / 29367765, submitted 2026-09-10 00:13 at engine SHA `f75a19c`** (policy `predictive`); the retired 2026-09-09 runs LSF 29365450 / 29365451 (`f2a98e0`, policy `total`) are kept in `results/v3_prelim/`; family comparison run under `total`: LSF 29367766 → `results/v3_total/family_cbo` |
| QDCBO / QMCBO runs | LSF 29365482 / 29365483, submitted 15:03 at engine SHA `9f25b41` |
| Host / env | `hpclogin9`, LSF `hpc` queue, 16 cores per job; `~/venvs/ccbo` (py3.10.18, numpy 1.26.4, GPy 1.13.2, emukit 0.5.1, ananke 0.5.0); `~/venvs/mcbo` (py3.9, botorch 0.7.2) for QMCBO; BLAS pinned to 1 thread |
| Third-party stacks | `third_party/DCBO` @ 85a9bdf, `third_party/mcbo` @ 0d0650e (fetched 2026-09-09, gitignored, unmodified — corrections are applied by `ccbo/qdcbo/stock_fixes.py` at import sites) |
| Results | `results/v3/{minimal,family_cbo,qdcbo,qmcbo}` (fresh) + `results/v3/ceo_minimal` (extracted from the unchanged archive, checksums OK) |

## 2. Stage A — tests

Fast suite on the final engine (v3.1): **173 passed** (the QMCBO joint-network tests run separately in the mcbo env: **11 passed**). Slow suite: **40 passed** — `test_minimal_invariance` included (and extended to QCBONP and BO/BOS identity), plus the Coral hash-independence test. QDCBO module: **22 passed**, parametrised over stock and corrected semantics (finest-partition identity holds in both). New test files: `test_estimators`, `test_causal_kernel_gradients`, `test_causal_acquisition_gradients`, `test_id_gate_tristate`, `test_cache_keys`, `test_noise_policy`, `test_decision_log`, `test_cost_charging`, `test_hqcbo_gf`, `test_hash_independence`, plus additions to `test_minibench_structure`, `test_minimal_invariance`, `test_qdcbo`, `test_qmcbo`.

Key numerical checks behind the tests: `CausalRBF` input gradients agree with central differences to 1e-9 (previously `gradients_X_diag` returned zeros and `dvar/dx` had the wrong sign); the σ_f² hyperparameter gradient now matches finite differences; the oracle-conditional FrontDoor integral reproduces 0.33590 exactly where mean substitution gives 0; g-computation draws recover `x²+1` where mean propagation gives `x²`; the likelihood variance stays at 1e-10 through `optimize()`.

## 3. Smoke and grid

Smoke (2 seeds, 12 trials): 64 MediatedChain+ParallelParent units and 24 FrontDoor units OK, sidecars present, row-0 costs as designed (CBO 12, QCBO 6, BOS 12, BO 6 on PP; 6 on MC), HQCBO and HQCBOGF sharing trigger 7 with a +6 split charge; family ToyGraph 3 units OK (row 0 = 20). v3.1 smoke (2026-09-10): Coral QCBO and CBO seed 0, 8 trials, predictive policy — QCBO pulls C+N+O on all 7 intervention trials and improves 37.61 → 36.40; CBO pulls N, N+O, C+N+O and improves 37.61 → 36.06 (the retired policy never left the initial-design incumbent).

| Suite | Dry-run | Completed | FAIL |
|---|---|---|---|
| minimal (`run_minimal_suite.py`, 30 seeds; LSF 29367764, v3.1) | 1320 (PP 720, FD 360, MC 240) | **1320/1320** | 0 |
| family (`run_cbo_family.py`, 10 seeds; LSF 29367765, v3.1) | 90 | **90/90** | 0 |
| family, `total`-policy comparison (LSF 29367766) | 90 | **90/90** | 0 |
| retired 2026-09-09 static runs (LSF 29365450 / 29365451) | 1320 / 90 | 1320/1320, 90/90 | 0 |
| qdcbo (E1 + E2, 20 seeds) | 240 | **240/240** | 0 |
| qmcbo (E1 + E2, 20 seeds) | 160 | **160/160** | 0 |

## 4. Verification gate (`scripts/verify_traces.py --results-dir results/v3`)

Minimal section: grid, sidecars, row-0 = charged initial design (3·Σ|arm|), observe caps (BO/BOS 0, others ≤ 3), no identification-gate error in any unit, split design charged at the trigger, HQCBO/HQCBOGF triggers equal, BO and BOS byte-identical across every condition, **QCBO and QCBONP identical at full decision-log precision under A1, A2, B1 (QCBONP also under A3)**, A3 control sup|Δ| = 21.50 for QCBO — all PASS; frozen published aggregates all PASS. Family: grid + sidecars, row-0 = charged design (10·Σ|arm|), frozen finals — PASS. New in v3.1: every minimal and family sidecar records `variance_policy == predictive` and a single engine SHA per suite — PASS. QDCBO: grid, sidecars, corrected semantics, E2 invariance at value and decision level, frozen finals — PASS. QMCBO: grid, E2 invariance at value and decision level (X, score) on 20/20 seeds for both environments, frozen finals and E2 counts — PASS. CEO: unchanged archive — PASS. **Total: 85 checks, 0 failures** (v3.1 run of record; the retired 2026-09-09 run passed 81/81 of the then-current gate), on `results/v3` and on a fresh extraction of the repacked archives (`SHA256SUMS` and all per-file lists verified). A3 negative control sup|Δ| = 21.43; MINIMAL_PUB / FAMILY_PUB re-frozen 2026-09-10 from the v3.1 runs.

## 5. Old (v2, 2026-08-21) vs new (v3) — MinimalBench

Columns: v2 (August engine), v3-prelim (2026-09-09, policy `total`, **retired**, `results/v3_prelim`), **v3 (2026-09-10, engine v3.1, policy `predictive`, `results/v3` — the numbers of record)**.

| Quantity | v2 | v3-prelim (retired) | **v3** |
|---|---|---|---|
| Table 1 sup|Δ| QCBO A1 / A2 / A3 / B1 | 0 / 0 / 21.6 / 0 | 0 / 0 / 21.5 / 0 | **0 / 0 / 21.4 / 0** |
| Table 1 sup|Δ| BO (all rows) | 0 | 0 | **0** |
| PP A0 CBO final / R50 | 0.0014±0.0003 / 4.17±0.37 | 0.0002±0.0001 / 3.61±0.36 | **0.0004±0.0001 / 3.65±0.35** |
| PP A0 QCBO final / R50 | 0.0021±0.0004 / 7.28±1.04 | 0.0002±0.0001 / 6.82±1.03 | **0.0003±0.0001 / 6.83±1.02** |
| PP A1 CBO final / R50 | 4.2500 / 216.16±1.12 | 4.2500 / 216.10±1.09 | **4.2500 / 216.15±1.11** |
| PP A3 CBO / QCBO final | 0.0354 / 0.0005 | 0.0354 / 0.0005 | **0.0354 / 0.0005** (constant-fallback arms: unchanged) |
| FD B0 CBO final | 0.00011±0.00008 | 0.00003±0.00001 | **0.00006±0.00004** |
| FD B1 CBO final | 0.0019±0.0007 | 0.0039±0.0022 | **0.0039±0.0022** |
| FD paired ΔR50 (CBO) | 6.18±1.06 | 5.91±1.06 [3.73, 8.08] | **6.31±1.04 [4.19, 8.44]** |
| FD B0 QCBO final / R50 | 0.0015 / 11.18 | 0.0015 / 11.18 | **0.0015 / 11.18** (constant-fallback arm: unchanged) |
| MC finals CBO / QCBO / HQCBO / HQCBOGF | 0.040 / 4.000 / 0.040 / — | 0.040 / 4.000 / 0.040 / 0.040 | **0.040 / 4.000 / 0.040 / 0.040** |
| MC R60 CBO / QCBO / HQCBO / HQCBOGF | 5.98±1.88 / 243.19±0.90 / 36.76±1.23 / — | 5.79±1.78 / 243.15±0.90 / 35.34±1.00 / 37.61±1.37 | **5.51±1.64 / 243.15±0.90 / 35.52±1.05 / 37.74±1.41** |
| HQCBO trigger (n=30) | 7.3±0.10 | 7.0±0.0 | **7.0±0.0** (every seed triggers at trial 7; HQCBOGF identical) |
| HQCBOGF − HQCBO (paired) | — | Δfinal ≈0; ΔR60 +2.27 [+0.90, +3.64] | **Δfinal +1.3e-7 (≈0); ΔR60 +2.23 [+0.86, +3.59]** |

Arms whose priors are all on the constant fallback (FD QCBO; PP A3 CBO/QCBO) are unchanged between v2 and v3, as expected: the gradient defect only affected non-constant causal priors. The prior-free arms (BO, BOS, CBONP, QCBONP) are byte-identical between v3-prelim and v3 (the variance policy never enters them), so the prelim→v3 movement in the table isolates the effect of the epistemic term on the minimal suite: small, and within one standard error everywhere.

New ablation results (v3; final ± se, R_T):

| Arm | PP A0 | FD B0 | MC C0 |
|---|---|---|---|
| CBONP (same arms, no prior) | 0.0004±0.0001, 23.03±2.61 | 0.0011±0.0003, 8.17±0.92 | 0.040, 8.44±1.96 |
| QCBONP | 0.0006±0.0002, 19.09±2.17 | 0.0009±0.0003, 15.63±1.89 | 4.000, 245.32±1.11 |
| BOS (all subsets, no prior, no obs) | 0.0005±0.0002, 17.49±2.08 | 0.0013±0.0004, 12.58±1.53 | 0.040, 12.00±2.76 |
| BO (joint arm) | 0.0037±0.0035, 5.83±1.23 | 0.40±0.15, 30.62±6.59 | 4.008±0.008, 238.55±0.85 |

Readings: the causal prior is worth 2.9 [1.8, 4.1] regret units to CBO and 2.2 [1.0, 3.4] to QCBO on MediatedChain, and cuts PP A0 CBO regret from 23.0 (no prior) to 3.7; CBONP is *identical* under FrontDoor B0/B1 (ΔR50 = 0), so the 6.3-unit paired increase is entirely the price of a corrupted prior; BOS owns the {X1} arm and reaches y* on MediatedChain without any graph (R60 12.0 vs CBO 5.5), while BO's joint arm sits on the fixed-partition floor.

## 6. Protected-pair invariance — explicit statement

**No protected-pair invariance broke.** On all 30 seeds, QCBO's and QCBONP's decision logs (observe/intervene draws, arms, intervention values at full double precision, outcomes, incumbents) are identical under A1, A2 and B1; QCBONP is additionally identical under A3 (the A3 edit changes only the quotient prior, which QCBONP does not use); QCBO diverges under A3 (sup|Δ| = 21.43). BO and BOS are byte-identical across all conditions of each SCM. This holds for the final (predictive-policy) run; it also held for the retired preliminary run.

## 7. Family, QDCBO, QMCBO

### 7.1 QDCBO (LSF 29365482) — complete

240/240 units, zero failures, every unit with a decision sidecar and `stock_quirks=false`. Gate: grid, sidecars, corrected semantics, **QDCBO E2-invariant on all 60 units at value and at full-precision decision level** — all PASS; `QDCBO_PUB` frozen.

| Setup (T=3, min) | v2 DCBO / QDCBO | v3 DCBO / QDCBO | paired QDCBO − DCBO (v3) [95 % CI] |
|---|---|---|---|
| stat | -6.14 / -6.43 | **-6.12±0.05 / -6.40±0.03** | -0.27 [-0.39, -0.15] |
| ind | -3.12 / -5.57 | **-3.13±0.06 / -5.55±0.13** | -2.43 [-2.69, -2.17] |
| nonstat | 8.03 / 6.33 | **7.01±0.83 / 5.78±0.02** | -1.23 [-2.98, +0.52] |

The corrected transition fitting (regress each child on its lagged parents instead of the stock slice mix-up) moves the nonstationary setup most — both the DCBO baseline and QDCBO improve, and the quotient advantage there is no longer significant at 95 % — while stat and ind are essentially unchanged.

### 7.2 Family (LSF 29367765, engine v3.1, policy `predictive`) — complete

90/90 units, zero failures, sidecars present, row 0 = charged 10-point design. Finals (BO; CBO / QCBO, mean ± se over 10 seeds) and the paired quotient disadvantage QCBO − CBO (95 % paired-t CI, `emit_paired_effects.py`); the retired 2026-09-09 run (policy `total`) and the `total`-policy comparison run under canonical ordering (LSF 29367766) are shown for context:

| Task | v2 (CBO / QCBO; paired) | v3-prelim `total` (retired) | `total` comparison run | **v3 `predictive` (record)** |
|---|---|---|---|---|
| ToyGraph | −2.164±0.000 / −2.169±0.005; −0.005 [−0.016, +0.007] | −2.164 / −2.169; −0.005 [−0.016, +0.007] | same | **−2.164±0.000 / −2.169±0.005; −0.005 [−0.016, +0.007]**; BO −2.171±0.005 |
| CompleteGraph | −3.472±0.041 / −1.285±0.029; +2.19 [+2.06, +2.31] | −3.557±0.011 / −1.314±0.003; +2.24 [+2.22, +2.27] | −3.557 / −1.314; +2.24 | **−3.567±0.002 / −1.314±0.003; +2.25 [+2.25, +2.26]**; BO −0.628±0.001 |
| SimplifiedCoralGraph | 36.065±0.018 / 36.449±0.017; +0.38 [+0.33, +0.44] | 37.091±0.139 / 37.540±0.137; +0.45 [+0.11, +0.79] — **flat: 0 improvements in 10/10 seeds for both arms** | 37.091 / 37.540; +0.45 — still flat (0/10) | **36.042±0.009 / 36.401±0.009; +0.36 [+0.32, +0.39]**; BO 9278.12±2.14; improvements per unit CBO 1–2, QCBO 1–8 |

Readings. Toy remains a statistical tie (7 worse / 3 better of 10 seeds). Synthetic's fine-CBO advantage is now essentially deterministic across seeds (10/10, CI width 0.01): QCBO's two arms {B} and {D,E} cannot reach the fine optimum. Coral is back to the v2 picture — both causal arms improve on their initial design in every seed and QCBO pays a small, significant price for its three cluster arms (+0.36) — and the `total`-policy runs show why the preliminary result was an artefact: with the epistemic term removed the off-support arms are unlearnable (§9.2), and this reproduces under canonical ordering (comparison run: identical flat finals). BO on Coral stays at the floor of its joint box (9278 ± 2; the joint arm is forced into D ∈ [2000, 2080], T ∈ [2300, 2400]). Cost table: Coral CBO charges 800 for its 31-arm design vs 100 for QCBO's three arms.

### 7.3 QMCBO (LSF 29365483) — complete

160/160 units, zero failures, per-iteration decision sidecars. The engine change here is only the zero-range normalisation guard (never triggered) plus logging, so the trajectories reproduce the August archive to the displayed precision: ToyGraph MCBO 1.39 ± 0.13 / QMCBO 2.16 ± 0.00, PSAGraph −5.15 / −5.15; E2: ToyGraph MCBO 9/20 identical (mean |Δfinal| 0.277), QMCBO 20/20; PSAGraph MCBO 0/20 (0.001), QMCBO 20/20. New in v3: QMCBO's E2 invariance is now established at **decision level** — the chosen intervention X and its score are identical on every iteration for 20/20 seeds in both environments. The ToyGraph MCBO plateau noted in the review persists (it is a property of the stock MCBO baseline, unchanged here) and remains the open instrumented-run item in `RUN_TODO.md`.

## 8. What changed in `verify_traces.py`

Grid 660 → 1320 with sidecars; row-0 init-cost check; split-charge and shared-trigger checks; BOS added to byte-identity; QCBONP added to the invariance checks (plus its A3 invariance); full-precision decision-log signatures compared in addition to the CSV columns; gate-error audit; QDCBO/QMCBO sidecar presence, no-stock-quirks audit and decision-level E2 invariance; all published constants behind `None` sentinels until frozen (MINIMAL_PUB / FAMILY_PUB frozen 2026-09-09, unfrozen and re-frozen 2026-09-10 from the v3.1 runs); v3.1 adds the sidecar variance-policy and single-engine-SHA checks (85 checks total). No check was weakened.

## 9. Addendum 2026-09-10 — the static suites were rerun (engine v3.1, commit `f75a19c`)

Auditing the flat Coral curves of the 2026-09-09 family run exposed two defects. The 2026-09-09 static outputs were moved to `results/v3_prelim/{minimal,family_cbo}` (kept, never reused) and both static suites were rerun; QDCBO / QMCBO / CEO are unaffected.

### 9.1 The causal prior depended on Python's hash seed

`ccbo/adjustment.py` ordered frozenset-typed C-DAG vertices with a bare `sorted()`, which is only a partial order on sets. Which minimal adjustment set was picked among equal-size candidates, the column order of the outcome GP's inputs, and the g-computation step order therefore followed `PYTHONHASHSEED`, i.e. changed from one interpreter to the next. Measured on the Coral QCBO/CBO priors (seed 0, engine `4646697`):

| Arm (adjustment set) | HS=0 | HS=1 | HS=2 |
|---|---|---|---|
| C+D+N+O+T ({L},{S},{TE}) — mean / v | 57.2 / 132.7 | 274.8 / 127.1 | 26.4 / 0.0 |
| N+O+T ({D},{S},{TE}) — mean / v | 0.0 / 0.0 | 160.8 / 184.9 | 19.5 / 0.6 |
| QCBO C+N+O — adjustment set chosen | {CO},{L},{P},{TE} | {CO},{L},{P},{TE} | **{D,T},{L},{S},{TE}** |

Runner probes (two interpreters, `PYTHONHASHSEED` 0 vs 1): family CompleteGraph CBO and QCBO CSVs differed; ToyGraph, all 16 minimal units (PP/FD/MC × CBO/QCBO/HQCBO/HQCBOGF), QDCBO (its runner already pins the seed) and QMCBO were identical. Within one LSF job every worker inherits the parent's seed, which is why the protected-pair identities held inside the 2026-09-09 archives while the archives themselves were not reproducible by a fresh run.

Fix: canonical ordering everywhere (`_vkey` / `_sorted_v`, `lexicographical_topological_sort`, ordered ADMG expansion). After the fix the Coral priors are identical under three hash seeds, the family and minimal runner probes are byte-identical across seeds, and the minimal probe is byte-identical to its pre-fix output (the fix is neutral there). `ccbo/tests/test_hash_independence.py` fits the priors in two interpreters with different seeds (CompleteGraph fast, Coral slow); every LSF script and `reproduce_paper.sh` additionally export `PYTHONHASHSEED=0`.

### 9.2 The epistemic-free prior variance made Coral unlearnable

With the 2026-09-09 policy (`v = lik_var + spread`, GP posterior variance excluded) the Coral arms whose ranges lie outside the observational support (D ∈ [2000, 2080] vs support [3, 7]; T ∈ [2300, 2400] vs [4, 8]) receive, under canonical ordering, prior mean **0.0** and prior variance **≤ 1e-6** (C+D+N+O+T, D+N, N+O+T, N; D+T gets 4.3), while the discarded epistemic term is 1.5e4 – 6.8e7. The mean is the outcome GP's zero prior mean far from data; the variance collapses because the 8-input GP interpolates (`lik_var` at its 1e-6 floor) and every adjustment row extrapolates to the same value (no spread). A variance adjustment of ~0 makes the CausalRBF kernel identically zero, so the arm's GP cannot move away from its prior mean whatever it observes, the σ=0 branch of EI reports a certain improvement of 37 units, and the optimizer pulls the arm forever: in the preliminary run QCBO spent 390 of 400 pulls on the joint arm at y ≈ 9300–9650 and neither CBO nor QCBO improved on its initial design in any of the 10 seeds (v2 had 1–3 improvements per unit).

This is not a property of CBO: the reference implementation's `var_do = np.mean(gp.predict(rows)[1])` averages the GP's predictive variance, i.e. it keeps the epistemic and noise terms. The engine default is now `VARIANCE_POLICY = "predictive"`: `v = lik_var + spread + Σ w_i s_i²` (law of total variance under the fitted GP). `"total"` remains selectable through `CCBO_VARIANCE_POLICY` and was run once more on the family suite under canonical ordering as the comparison column; `results/v3_prelim/minimal` is the minimal suite under the retired policy (byte-neutral to the ordering fix).

### 9.3 Runs

| Suite | Policy | LSF job | Output |
|---|---|---|---|
| minimal | predictive | 29367764 | `results/v3/minimal` |
| family | predictive | 29367765 | `results/v3/family_cbo` |
| family (comparison) | total | 29367766 | `results/v3_total/family_cbo` |

Sidecars record the policy (`unit.estimator.variance_policy`) and the engine SHA; `verify_traces.py` now requires `predictive` and a single SHA per static suite, and `MINIMAL_PUB` / `FAMILY_PUB` were unfrozen until these runs complete. The numbers in §5 and §7.2 above are those of these runs (minimal 1320/1320, family 90/90, comparison 90/90; zero failures).

### 9.4 Hardware and bit-exactness

The archives are bit-exact reproductions only on the CPU model they were produced on: OpenBLAS selects micro-architecture-specific kernels, and L-BFGS amplifies ulp-level differences in the acquisition gradients into visibly different intervention values. Measured on the family suite between the retired run (XeonE5 avx2 node) and the `total`-policy comparison run (XeonGold avx512 node) — same code path for the prior-free BO arm, canonical ordering neutral on ToyGraph:

| Unit | max\|Δbest_y\| | max\|Δx\| |
|---|---|---|
| ToyGraph BO / QCBO | 1.4e-8 / 3.2e-11 | 2.2e-4 / 3.6e-7 |
| CompleteGraph BO / CBO | 7.7e-9 / 2.9e-7 | 4.4e-7 / 1.3e-5 |
| Coral BO (5-D joint arm, off-support box) | 32.8 (trajectory-level divergence; finals 9279.97 ± 3.35 vs 9278.12 ± 2.14) | 93 |

By contrast the minimal suite's two runs both landed on XeonE5 nodes and every prior-free unit (BO, BOS, CBONP, QCBONP; 840 files) is byte-identical between them. Consequences: (i) the protected-pair identities and the BO/BOS byte-identity checks compare units of the *same* job and node, so they are unaffected; (ii) `reproduce_paper.sh` regenerates everything from the archives, which is hardware-independent; (iii) re-running a suite from scratch reproduces the archive bit-for-bit on the recorded host model and to floating-point rounding elsewhere (chaotic units such as Coral BO can then differ at trajectory level while staying inside the seed-to-seed spread). Host models are now recorded per archive in `artifacts/traces/MANIFEST.md`.

| LSF job | Suite | Exec host | Model |
|---|---|---|---|
| 29365450 / 29365451 (retired) | minimal / family | n-62-21-63 / n-62-21-65 | XeonE5_2 (avx2) |
| 29367764 | minimal (record) | n-62-21-95 | XeonE5_2 (avx2) |
| 29367765 | family (record) | n-62-12-3 | XeonGold (avx512) |
| 29367766 | family, `total` comparison | n-62-11-54 | XeonGold (avx512) |
