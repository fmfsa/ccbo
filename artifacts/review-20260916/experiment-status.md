# Experiment and manuscript status — 17 September 2026

Snapshot: 11:01 UTC. Completion counts below are frozen-run records; only the dynamic campaign has passed its final complete-matrix audit.

| Campaign | Complete | Running | Unstarted | Scientific status |
|---|---:|---:|---:|---|
| Matched noisy static | 351 / 720 | 15 | 354 | Final inference pending |
| Corrected MCBO, main + protected | 58 / 200 | 16 | 126 | Final inference pending |
| Coherent population dynamic | 180 / 180 | 0 | 0 | Final audit passed; independently recomputed |

The user authorized increased parallelism. The cluster restricts `bmod` to administrators, so unstarted jobs were safely suspended and transferred through ordinary owner-authorized `bsub` submissions. MCBO now has 14 additional slots (job 29426675) alongside its two original active runs; static has 12 additional slots (job 29426697) alongside the original active runs. At this snapshot, **31 scientific runs are actually running: 16 MCBO and 15 static**, compared with six before the handoff. Original active runs will drain; the complementary arrays retain caps of 14 and 12.

The same 200 MCBO and 720 static scientific units remain in scope. Frozen code, seeds, budgets, scoring precision and outputs were not changed. The transferred original queue entries remain suspended to prevent duplicate execution. Dependent validators (29426676 and 29426698) verify completed-unit identities, frozen source and output hashes before resuming those original entries for hash-only reuse. This preserves the original complete-matrix final audits. No campaign STOP markers or failed array elements were present at this snapshot.

Historical work used 80 main + 80 protected MCBO runs with 16 concurrent units. The current matrix has 120 main + 80 protected runs: the additional 40 are the fine-model/coarse-menu control. Historical 100-round PSA medians were 113.5 minutes for MCBO and 37.5 minutes for QMCBO. Current full PSA runtimes remain uncertain; old code already used 100,000-draw scoring. Increased parallelism improves throughput, not individual-run duration; it does not establish an exact completion deadline.

## Dynamic evidence

All 180 units, all nine 20-seed groups, recorded per-slice values, costs, group means and paired t intervals were independently recomputed from the strict final audit with no discrepancies. Quotient/coarse minus fine/coarse differences in the sum of committed slice outcomes (lower is better): stationary−1.024 [−1.300,−0.748]; independent0 [0,0]; regularized nonstationary−0.233 [−1.051,0.585]. Intervals are pointwise 95% t intervals across seeds; Monte Carlo scoring errors are recorded separately. These are policy outcomes, not global regret or a universal superiority claim.

## Manuscript

The user supplied their actual LaTeX. The directly edited main file is `paper/qcbo_aistats.tex` in the GitHub handoff, or `revised-manuscript/qcbo_aistats.tex` in the local outputs. Existing section order, headings, macros, theorem labels and figure/table positions were preserved. Static/MCBO result floats are visibly pending. The existing family-results float/table now show the audited dynamic comparison. The stationary graph is correctly scoped and the regularized companion is a separate additional figure.

The previous `overleaf-edit-sheet.md` remains a detailed reference, not a second patch to apply to the edited file. Some old recommendations to split/remove figures were deliberately not applied because the user asked to preserve structure. `manuscript.diff` records exact direct-source edits. The reconstructed bibliography requires author confirmation of the incomplete Park 2025 venue record before submission.

## Code and campaigns

The 91 integrated reporting/scientific files and validated runtime wrappers are included in the publication branch. Frozen protocol manifests and existing controller scripts are under `experiments/review-20260916/`. The live jobs continue in `/zhome/15/b/215295/Repositories/ccbo-experiments-20260916`, separate from the publication checkout. Controller paths describe that deployment; they are not an instruction to submit duplicate arrays.

The prior code validation passed 48 analysis/provenance tests, 36 dynamic tests and 16 actual-backend smoke runs. These are distinct suites, not one aggregate test count. Full static and MCBO scientific results remain blocked until the prespecified matrices and audits finish, or an explicit documented scope amendment is agreed. The underlying code and scientific parameters have not been changed merely to obtain faster or more favorable outcomes.
