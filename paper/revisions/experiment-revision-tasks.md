# Experimental corrections: exact manuscript edits and implementation tasks

This addendum supersedes the reporting-only scope of M04–M18 and E01–E04 in the earlier revision plan. The author has authorized changing experiments and rerunning them on P1 DTU HPC. The mathematical, literature, figure and provenance tasks M01–M03 and M19–M23 remain necessary. Page references refer to the supplied 20-page PDF. Quotes normalize extracted line breaks. “Implemented” below means code exists; it does not mean a scientific replication or manuscript merge is complete.

No numerical result from the corrected protocols may be inserted before the complete prespecified matrix passes its audit. Preserve the original results as a separately labeled historical archive. Do not infer success from a favorable ranking, discard unfavorable seeds, or tune benchmark equations after inspecting replication outcomes.

## E05 — Make noisy comparisons use the same information

**P0; demonstrated mismatch.** Appendix C, p.14, “Controlled suite”: “Initial and sequential outcomes are evaluated by the same closed-form population expectation.” The historical controlled learner used population responses; CEO used a noisy interface and also accessed noiseless incumbent bookkeeping. These channels do not define one matched comparison.

**Replace the controlled-suite paragraph with:**

> For each of 30 paired replicates (seeds 2000–2029), we generate 100 fresh observational rows from the true SCM. Every method and supplied-graph condition receives the same rows within a replicate. Initial and sequential interventions return one noisy observable row from the true SCM. Initial designs contain three uniformly sampled points per permitted arm, keyed by arm so common arms share values and disturbances. All measured rows, including initialization and refinement, are charged at one unit per manipulated variable. The total intervention budget is 100 for ParallelParent and FrontDoor and 120 for MediatedChain. An independent scorer evaluates population responses only for recommendations already selected from purchased measurements. Historical exact-response experiments are archived separately and are not pooled with these comparisons.

**Code:** `ccbo/matched_protocol.py` supplies named random streams, true-SCM observations and measurements, keyed initialization, and a cost ledger. `scripts/matched_ceo_runtime.py` feeds the same purchased row into native CEO bookkeeping and prohibits population-score access. `scripts/run_matched_controlled.py` writes learner events before scoring. These are opt-in protocols; retain historical modes.

**Acceptance:** actual pinned CEO integration, poisoned/disabled scorer firewall, common-row parity, analytic-mean simulation checks, and no additional uncharged observation. Real backend tests and 17 short pilots passed; nominal-budget preflight and complete replication remain required.

## E06 — Define an implementable recommendation and its metric

**P0; demonstrated interpretation error for a noisy comparison.** Figure 3 caption, p.8: “Best-so-far population objective”; Appendix C, p.14: “final values and regrets of every method.” These descriptions do not identify how a noisy learner chooses its recommendation.

**Replace the primary figure ordinate and caption wording with:**

> Population objective of the measured-data recommendation. At each budget checkpoint, the recommendation is the executed action with the smallest measured outcome, breaking ties by canonical arm and execution index. The external scorer evaluates this recommendation without affecting learning. Its population value need not improve monotonically. Oracle-best-visited population values are reported only as a separate diagnostic.

**Insert in the metric definition:**

> The primary endpoint is population recommendation gap at the declared final cost budget. The secondary endpoint is the sum of right-continuous population recommendation gaps at integer cost checkpoints 12 through the final budget. Checkpoints use only measurements already purchased. The seed is the replication unit; paired method differences use two-sided, pointwise 95% Student-t intervals. We also report all seed-level outcomes, cost, runtime and failures.

**Code:** independently reconstruct recommendations from `events.json`; validate event IDs, domains, costs, eligibility and tie rules; independently recompute closed-form population scores. Reject future interpolation, cumulative minima of oracle scores, missing seeds and incomplete budgets. `scripts/summarize_matched_campaign.py` implements this audit; complete-matrix and full-budget pilot support are being finalized.

**Acceptance:** both endpoint and cost-integrated metrics reproduced from raw events; pilots emit no inferential comparison. Final plots must display the new estimand, not recycle the historical monotone curves.

## E07 — Fit and actually use noisy GP models

**P0; demonstrated code defect and missing specification.** Appendix B, pp.13–14 describes acquisition and causal-effect priors but does not disclose that the historical scalar loop rebuilt a default GP before acquisition and discarded its just-optimized fit at the next iteration. Exact-response tiny likelihood noise is also unsuitable for the new noisy feedback channel.

**Insert after the scalar GP description:**

> In the noisy controlled protocol, the GP likelihood is optimized before the fitted model is used for acquisition. Its initial variance is max(0.01, 0.1 times the variance of the available raw target values), with bounds [10^-6, 10^6]. This is a homoscedastic observation-noise approximation. Native CEO retains its learned likelihood. We log the fitted parameters actually used at each acquisition; the historical exact-response protocol remains separate.

**Code:** the scoped matched scalar adapter unlocks and fits the likelihood before acquisition, retains that fitted model for the actual decision, and records parameters. Correct the affordable-arm mask to return the native acquisition shape, including a last-credit singleton case.

**Acceptance:** actual backend tests verify fitted parameters used before selection and forbid budget overshoot. These gates passed on HPC. Historical family optimizer comparisons using the old loop are not certified by these tests; rerun them under a separately specified corrected protocol or retain them only as historical supplementary results.

## E08 — Distinguish observational fallback from no-prior BO

**P1; demonstrated attribution ambiguity.** FrontDoor discussion, p.7: fine CBO “takes longer to do so, and its regret rises to the level …”. Similar outcomes alone do not establish that replacement of an identified prior is the only changed pathway.

**Add to the FrontDoor methods:**

> We include a forced-fallback control on the baseline graph: it retains the baseline arm menu and causal-kernel implementation but replaces both causal mean and variance functions by the same observational fallback used under B1. Complete paired traces must match B1 under this control before interpreting the B0/B1 contrast as a prior-replacement effect. A plain RBF no-prior model is a separate ablation.

**Code:** `scripts/matched_fallback.py`, method `CBO-FALLBACK`, restricted to FrontDoor/B0. Use the existing fallback factory; do not substitute the wrong graph, alter arms, or disable the causal covariance.

**Acceptance:** native fallback mean and variance agree at multiple points; complete three-decision measurements, recommendations and fitted parameters match B1. Local real-backend tests and the HPC fallback gate passed; the final 30-seed control remains separately tracked.

## E09 — Charge hierarchy expansion and remove predetermined success wording

**P0; new noisy protocol requirement.** §3.3 p.4: “When the plateau rule fires, HQCBO splits…”; Figure 3 p.8: “the dashed vertical line marks refinement at trial 7” and “reaches the fine optimum.”

**Replace the experimental refinement description with:**

> HQCBO uses the graph-free hierarchy and the fixed five-trial, 10^-3 plateau rule. A proposed split exposes new arms only after its entire three-point-per-new-arm design is affordable and purchased. These measurements count toward the shared budget and can immediately change the recommendation. Existing arm data and quotient-derived priors are retained; new arms use plain GP priors. We report the observed split time for every seed. The structural result guarantees that refinement makes the fine-optimal arm available, not that a finite noisy run attains its optimum.

**Code:** `scripts/matched_refinement.py` resumes the actual scalar learner, uses independent split streams and an atomic six-row purchase, and guards graph access after initialization. Preserve the QCBO prefix exactly. Refuse unaffordable splits without obtaining outcomes.

**Acceptance:** five real-backend tests passed on HPC. The nominal-budget pilot also passed, including an actual paid split. Report split events and include all 30 final seeds; do not retain a universal trial-7 line or historical optimum claim in the new figure.

## E10 — Correct MCBO coordinates, scalar fitting and acquisition randomness

**P0; demonstrated defects.** Appendix C p.14: “MCBO/QMCBO retain the released implementations.” Proposition 3 p.11: “QDCBO and QMCBO likewise reduce to DCBO and MCBO.” These statements conceal incompatible numerical paths.

**Replace the implementation statement with:**

> The corrected model-based comparison uses pinned upstream code with disclosed symmetric repairs. Physical intervention coordinates are mapped exactly once; epistemic controls are unchanged. Both implementations guard constant-feature normalization and share the scalar GP construction and fitting policy at singleton resolution. Acquisition optimization uses fixed common base draws so repeated values and gradients do not change with candidate ordering. Environment, initialization, acquisition and scoring use separate named random streams. Singleton posterior and gradient equivalence is checked for this corrected protocol; it is not asserted for the historical release.

**Code:** `ccbo/qmcbo/corrected.py` and scoped compatibility hooks repair upstream mapping, zero-span normalization, fixed-noise scalar fitting, candidate-batch common random numbers and logging. Upstream mapped `[-3,5,0.5]` was overwritten by normalized `[0.2,0.4,0.5]` in a direct witness. Log actual acquisition/model diagnostics; restore patches on exit.

**Acceptance:** pinned runtime tests passed, including posterior/gradient and repeated-acquisition checks; six short corrected pilots completed. Historical MCBO flatness is not an acceptance criterion. Longer runtime preflight and all affected paired replications remain necessary.

## E11 — Separate action-menu effects from model effects

**P0; demonstrated menu mismatch.** Table 1 p.8 and §5.4 pp.7–8 compare MCBO/QMCBO without identifying that upstream Toy omitted the joint action available to the quotient.

**Insert before the family comparison:**

> We compare corrected fine models with the full admissible menu, fine models restricted to the coarse menu, and quotient models on that same coarse menu. The matched-menu contrast isolates model construction; full-menu contrasts combine modeling and action-set effects. The primary model-based protocol uses 100 sequential rounds and 20 paired seeds, not an equal-total-intervention-cost claim. We report initialization counts and total intervention costs. Toy retains its selectable null action by explicit family convention; PSA observational initialization can train models but cannot be recommended because null is outside its menu.

**Code:** add the missing admissible joint Toy target; keyed initialization gives common arms identical samples; enforce recommendation eligibility. Freeze Toy/PSA × three methods × 20 seeds = 120 units, plus separately executed protected pairs if retaining a 20-seed invariance count.

**Acceptance:** common-menu initialization and target lists match exactly; all costs include initialization; no recommendation uses an inadmissible observational row. Independently audit complete manifests and traces before inference.

## E12 — Correct PSA stochasticity and scoring claims

**P0; demonstrated false generalization.** Appendix C p.14: “β=10, zero observation noise and maximization.”

**Replace with:**

> The model-based family uses β=10 and maximizes the simulator reward. Toy uses zero structural disturbance in this protocol. PSA remains stochastic because of its intrinsic age, BMI and outcome sampling; its reward is negative PSA. We evaluate each data-selected recommendation using 100,000 independent scoring draws and report Monte Carlo standard errors. These draws are unavailable to acquisition, fitting and recommendation selection. Changing scoring precision must leave the learning trajectory unchanged.

**Code:** preserve intrinsic PSA randomness, isolate scorer state and record score sample count/SE. Do not treat scalar GP regularization variance as a literal deterministic mechanism disturbance.

**Acceptance:** scorer-precision firewall and analytic/independent simulator checks. A rounded equality of means is not trajectory invariance and is not evidence of zero noise.

## E13 — Replace double-counted temporal priors

**P0; demonstrated estimator defect.** §3.2 p.4: QDCBO “otherwise retains DCBO’s acquisition and update logic.” The inherited emission/transition construction can separately predict the entire child response and add it twice.

**Replace the dynamic implementation description with:**

> The corrected temporal protocol fits each child cluster once, conditional on the union of contemporaneous and lagged parent variables. It propagates joint parent draws through the temporal model rather than adding two full-response regressions. Fine and quotient methods share this estimator and differ in partition and declared action menu. Root clusters use their empirical joint distribution. Conditional multivariate clusters use a joint GP predictive approximation with a declared residual covariance estimate. The reported predictive outcome variance is not the Monte Carlo standard error of an estimated mean.

**Insert the residual specification once validated:**

> We estimate the conditional residual covariance from leave-one-joint-row-out residuals, holding out all outputs at the same training row together, with 10% shrinkage toward its diagonal. The native diagonal likelihood is excluded before this covariance is added. This residual law includes finite-data fit error and model misspecification; it is not asserted to recover the exact structural disturbance distribution.

**Code:** `ccbo/qdcbo/coherent.py` implements union-parent conditional models and ancestral integration. A fixed-blanket Gaussian witness demonstrated inherited mean error 7.5 and variance error 5.3125; the initial coherent estimator removed this error. Independent review then identified scalar-output versus joint-row holdout and a graph-horizon dispatch bug; their fixes passed fresh gates, including actual backend dispatch.

**Acceptance:** exact linear witness, nonlinear parent integration, correlated-output block holdout, covariance ordering/no double likelihood, singleton and protected complete-trace identity, and actual pinned-backend population trajectories. These checks passed in 36 tests and 16 tiny actual-backend units. Resource and precision pilots, and the final temporal matrix, remain separate gates.

## E14 — Use an explicit dynamic population estimand

**P0; demonstrated mismatch between zero-disturbance paths and nonlinear expectations.** Table 1 p.8 labels final dynamic objective values without specifying how previous outcomes are treated; Appendix C p.14 lists three slices and ten trials per slice.

**Replace the dynamic evaluation description with:**

> The target at slice t is the population mean under the committed sequence of manipulator interventions through t. All free variables, including past outcomes, are resimulated across the complete prefix; we never replace a random past outcome by its expectation. Training evaluations use a fixed 2,048-draw common-random-number Monte Carlo response oracle. Independent post-run scoring uses 100,000 draws and records its standard error. This finite-Monte-Carlo population-response regime is distinct from the single-row noisy controlled experiments. We report each committed slice value under its own intervention prefix and their sum. Different methods can commit different prefixes; later-slice comparisons are sequential-policy outcomes.

**Code:** `ccbo/qdcbo/population.py` supplies vectorized prefix simulation and independent scoring; `runner.py` exposes opt-in `--objective-protocol population-mc`. Serialize measured-batch recommendation IDs before scoring. Exclude internal initial sentinel values from scientific metrics. Log current-arm counts and total simulated-prefix assignments separately.

**Acceptance:** E[Y_0²]=5 rather than the incorrect squared-mean value 4 in a regression witness; scalar/vectorized sampler parity; only chosen manipulators remain clamped; score precision cannot alter decisions. Determine integration precision from numerical convergence diagnostics, not rankings. Freeze the final dynamic matrix after validation.

## E15 — Replace the ill-posed nonstationary population benchmark explicitly

**P0; demonstrated integrability/graph issue.** Table 1 p.8 “nonstat”; Appendix C p.14 “nonstationary temporal SCMs”; Figure 11 p.19 graph. The inherited mechanism contains X_t/X_(t−1), where the denominator can be Gaussian or zero, and the supplied graph omits actual cross-lag parents.

**Replace the benchmark description with:**

> We introduce a separately named nonstat-regularized companion. At its change point the Z mechanism uses −X_t/sqrt(1+X_(t−1)²) + Z_(t−1) + ε_Z. The denominator is at least one; the remaining mechanisms are unchanged. The graph includes X_(t−1)→Z_t at the change point and Z_(t−1)→Y_t where the post-change equation uses that parent. These finite-horizon mechanisms have at most linear growth and finite second moments under the declared disturbances. We do not report this companion as a replication of the original nonstat SCM.

**Code:** add a separate setup name and graph constructor; preserve original historical setup; reject original nonstat in the new population mode. Test the actual setup factory, zero-valued interventions and every cross-lag parent. Redraw its graph and caption as a new benchmark.

**Acceptance:** no singular denominator or nonfinite values at valid interventions; graph matches equations; distinct filenames/configuration and separate historical/new results. Do not silently reuse historical nonstat numbers.

## E16 — Replace numerical assets only after complete audits

**P0; affected evidence.** Figures 1–3, Table 1 p.8, Tables 3–5 pp.15–17 and model-based graphs/captions must be reconsidered wherever they report a changed learner, feedback regime, action menu, objective or SCM.

**Current:** all old values, historical error bars, trial-index labels and blanket success claims in these assets.

**Required replacement:** generate new assets directly from validated frozen campaign outputs. Captions must identify protocol version, objective, minimization/maximization, resource axis, initial-design charges, number of independent seeds, uncertainty interval, and any missing/failed unit. Report old data only in a clearly separate historical section. Never fill new table cells with old results while runs are pending.

**Code/deployment:** `scripts/run_frozen_unit.py` verifies source hashes, uses a manifest-hash output namespace, captures configuration, rejects stale/incomplete resumes and hashes expected outputs. Campaign analyzers must verify the exact prespecified matrix and raw-event semantics, not only output-file existence. Every failed attempt remains archived. Fixing a demonstrated bug requires a new version and all affected paired reruns.

**Acceptance:** complete successful audit or explicit failure reporting without inferential claims; render each new figure at publication width; reconcile all text numbers against machine-readable summaries. This is still open.

## E17 — Resolve the legacy family track and finish the actual paper

**P1; open scope and author information.** CBO-family Toy/Synthetic/Coral rows use historical optimization and resource choices. The corrected noisy controlled and model-based tracks do not retroactively validate those rows.

**Required text pending a separate rerun:**

> The legacy CBO-family results are retained as historical reproductions under their archived configurations and are not included in the corrected matched-protocol conclusions.

**Remaining work:** specify an equal-resource family protocol and correct its optimizer if retaining those rows as primary evidence; freeze its seed/configuration matrix before running it. Establish Coral data/preprocessing lineage and redistribution terms before release. Apply all approved before/after edits to the current Overleaf source, rebuild the whole manuscript, resolve citations/cross-references and inspect rendered figures. The current Overleaf source and author-owned provenance records remain missing; do not restore deleted historical TeX as a substitute.

These are explicit remaining tasks, not evidence that all experiments are already correct or that the paper is ready for submission.
