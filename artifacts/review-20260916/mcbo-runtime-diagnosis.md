# MCBO runtime diagnosis (read-only; 17 September 2026)

## Immediate answer

The archived runs were already expensive, especially PSA: median **113.46 minutes per 100-round fine MCBO run**, versus **37.47 minutes for QMCBO**. The corrected campaign also has **200 units at only two concurrent CPU slots**, so per-run costs accumulate into days. There is evidence of some per-run slowdown, but the current five-to-eight-day estimate is a projection, not a measured completion time, and its GP-growth allowance has not been profiled. Do not explain this as a new 100,000-draw scoring requirement: the old upstream code already did that.

No algorithms, hyperparameters, jobs, or outputs were changed during this diagnosis. No new simulations were run.

## Direct historical timing evidence

Read all 80 local main-run sidecars in `work/code-audit-traces/qmcbo/trial_results_*_info.json`; each records `num_trials=100`, 20 seeds per cell. Times are recorded wall seconds converted to minutes.

| Environment / method | Historical median | Historical range | Current timing supplied by root |
|---|---:|---:|---:|
| Toy / MCBO | 22.22 min | 16.72–37.02 | full menu, 100 rounds: 42.2 min |
| Toy / QMCBO | 22.42 min | 17.68–32.56 | coarse menu, 100 rounds: 40.7 min |
| PSA / MCBO | 113.46 min | 109.24–123.18 | full menu, **20 rounds**: 45.8 min |
| PSA / QMCBO | 37.47 min | 35.67–41.37 | coarse menu, **20 rounds**: 13.7 min |

Additional new fine/coarse-menu timings: Toy 100 rounds 19.8 min; PSA 20 rounds 13.8 min. These are different algorithms/menus, not directly interchangeable historical baseline timings. Current values are root's measured reports, not independently re-read fresh HPC outputs here.

Thus measured Toy examples are approximately 1.9× and 1.8× their historical medians. Multiplying PSA's 20-round pilot times by five gives about 229 and 68.5 minutes, approximately 2.0× and 1.8× historical medians, but **that multiplication is not a measured 100-round result**. Initialization has a fixed cost, while later GP data sizes and optimizer work may grow. A single pilot also does not estimate seed variability.

## What the source actually changed

Paths below are relative to workspace. Corrected source is `work/experiment-redesign/repo`; the frozen campaign copy is `work/experiment-redesign/releases/mcbo-campaign-v3`.

1. **Action menu adds real work in Toy fine/full.** `work/mcbo-source/functions.py:313–317` gives historical Toy targets {X1}, {X0}, null (three). `ccbo/qmcbo/corrected.py:137–160` constructs all manipulable subsets and retains eligible null, producing four. `work/mcbo-source/mcbo_trial.py:309–345` optimizes separately over every target. Thus Toy full performs four target optimizations per round instead of three, plus the new joint arm's initialization. Toy coarse has joint+null (two), explaining a concrete reduction relative to full. This does not by itself explain QMCBO's slowdown.

2. **PSA full menu already had three targets.** `work/mcbo-source/functions.py:366–370` already lists A, S, A+S. Corrected PSA full therefore does not gain an extra arm. Coarse-menu fine and quotient use only A+S, cutting target optimizations from three to one. This is a meaningful contributor to the approximately 3.3× pilot full/coarse runtime difference. Fits still occur once per round, not once per target.

3. **No new per-round fit schedule.** Old `mcbo_trial.py:273–275` constructs the model before the target loop, on every round. Corrected `corrected.py:275–283` calls that same suggestion routine on every round. Upstream `work/experiment-redesign/upstream_gp_network.py:44–71,128–154` fits each node GP. There is no demonstrated change from occasional fits to every-round fits.

4. **Fine scalar GP fitting is intentionally the same class/policy.** `corrected.py:40–50,102–107` uses FixedNoiseGP, Standardize(m=1), and upstream `fit_gpytorch_model`. Historical upstream `upstream_gp_network.py:128–154` has the same policy. There is no newly introduced learned noise model for fine MCBO. In quotient singleton clusters, however, `qgp_network.py:126–151` now opts into this same fixed-noise helper; historical quotient singleton clusters used SingleTaskGP with learned noise. This changes likelihood/optimization geometry and can alter runtime; its measured contribution is unknown. Multi-member quotient clusters already used KroneckerMultiTaskGP plus full-rank task-noise likelihood; that expensive joint architecture is not new.

5. **100,000 scorer draws were already present.** Historical `mcbo_trial.py:30–39` defines `obj_mean` using `X.repeat(100000,1,1)`; lines 63–65 score initialization and lines 102–105 score each new action. Corrected `corrected.py:250–260` also scores each purchased point, then reuses stored scores for recommendations. It does not rescore every past point on each round. For deterministic zero-noise Toy, it now uses only **one** draw, so this particular change saves work. PSA remains intrinsically stochastic and uses 100,000 vectorized draws. New initial rows are scored separately rather than one batched init call; that changes initialization overhead/memory, not the factor of 100,000 itself.

6. **Acquisition behavior is corrected, not just more accurately scored.** `corrected.py:82–99,109–131` fixes physical intervention mapping, guards normalization, uses a fixed sample-average objective within numerical acquisition optimization, and broadcasts residuals across candidates. The historical fine hallucinated model re-split the original normalized X after superclass mapping (`upstream_gp_network.py:448–459`). Historical noise was freshly sampled across calls. Corrected gradients therefore optimize a different, properly mapped and stable acquisition landscape. More or fewer numerical objective/gradient evaluations are possible; the code alone cannot establish how many. Avoid claiming that old runs simply exited early without optimizer-iteration evidence.

7. **128 acquisition samples are upstream behavior.** `work/mcbo-source/mcbo_trial.py:187,201,227` already constructs SobolQMCNormalSampler(num_samples=128). Corrected runner delegates the same acquisition/optimizer routine. This is not a newly increased sample count. The local snapshot does not include the exact upstream `optimize_acqf.py`, so numerical restart/raw-sample/iteration settings should be read from the frozen vendor on HPC before asserting their values.

8. **RNG isolation adds overhead, but its share is not measured.** `corrected.py:26–37,118–121` saves/restores Python, NumPy and Torch RNG state around each acquisition posterior draw. Candidate-level deterministic residual sampling preserves the fixed objective. This can be called many times during gradients/optimization. It is a plausible overhead, not a demonstrated dominant bottleneck. Diagnostic wrappers additionally serialize model state and candidate values (`corrected.py:216–239`); no phase timers currently separate these costs.

9. **One-thread settings are not demonstrably new.** Historical `work/experiment-redesign/original-backup/ccbo/qmcbo/runner.py:21–24` already used `setdefault(...,"1")` for BLAS/OpenMP thread variables. Current `server-launch/mcbo_campaign.lsf:2–5,31` reserves one CPU per unit, caps array concurrency at two and explicitly exports one thread. Historical defaults could have been overridden by inherited environment; actual historic thread counts and scheduler concurrency are not in these timing sidecars. Therefore do not claim a measured old-many-threads/new-one-thread slowdown.

## Campaign arithmetic versus algorithm speed

The current 200-unit matrix adds a fine model restricted to coarse actions to the old fine/quotient comparisons. For each environment, main+protected totals comprise 40 fine/full, 20 fine/coarse and 40 quotient/coarse units. Using root's Toy 100-round timings and **linear 5×** PSA pilot projection gives:

- Toy: 40×42.2 + 20×19.8 + 40×40.7 = 3,712 CPU-minutes.
- PSA: 40×229 + 20×69 + 40×68.5 = 13,280 CPU-minutes.
- Total about **283.2 CPU-hours / 2 slots = 141.6 elapsed hours = 5.9 days**, excluding queue delays and variation.

This alone explains the center of a multi-day projection without assuming an extra GP-growth factor. Applying additional GP-growth inflation may be conservative but is not established by timing instrumentation. Conversely, this is not a guaranteed upper bound. Historical 100-round timings themselves imply hours/days for large seed matrices at two slots; the user's memory of a faster campaign could involve different concurrency or a smaller subset. No historical launch allocation was independently established here.

## Minimum useful profiling, without tuning the experiment

In a separate engineering replica (not by editing the running frozen release), capture per round: model construction/each GP fit time, fit optimizer iterations/retries, target-specific acquisition time and objective/gradient evaluation counts, pure environment measurement time, scoring time, RNG save/restore time, and artifact serialization time. Record training rows per node/cluster, optimizer options, library versions, CPU model, Torch intra/inter-op thread counts, and all BLAS variables. Profile the same fixed pilot seed at an early round and a saved late-round state; keep all decisions/parameters and scorer outputs unchanged. Existing unit `secs` alone cannot identify the dominant phase.

A frozen-vendor source read and existing HPC log inspection for SciPy/GP retry warnings are cheap first steps. Do not infer bottlenecks from objective rankings or select a faster seed/method. Root owns any authorization for additional runs.

## Potential equivalent optimizations (not implemented or certified)

- Precompute the deterministic acquisition residual tensors outside repeated posterior evaluations, or isolate only the RNGs actually consumed. To be equivalent, preserve exact stream/draw shape and candidate permutation/batch invariance; compare acquisition values **and gradients**, selected actions and complete paired traces. This targets repeated RNG setup, not statistical precision.
- Cache model quantities constant throughout all target optimizations for a round, e.g. a cluster's fitted noise-covariance Cholesky factor. Care: caching must preserve any gradient requirements and invalidate on each refit. Current `qgp_network.py:270–279` recomputes this small factor while sampling. Likely small matrices; magnitude unknown.
- Reuse an already stored independent score when a recommended event repeats; corrected code already does this. Do not propose it as a missing speedup.
- Warm-starting hyperparameter fits, reducing restarts/samples, fewer fitting rounds, lower scoring precision or pruning actions is **not certified mathematically equivalent** and changes the frozen study. Do not apply those silently.
- More campaign slots reduce elapsed time, not CPU cost, and require resource approval/scheduling outside this audit. Extra BLAS threads are not automatically faster for these small GP models.

Bottom-line evidence: bigger matrix/two-slot cap and Toy's added target are definite; model/acquisition corrections can change numerical workload; old 100k scoring and per-round fitting were already present. A causal breakdown of the remaining approximately twofold per-unit change requires phase-level profiling.
