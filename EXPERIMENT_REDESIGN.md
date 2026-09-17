# Corrected experiment protocols (September 2026)

These opt-in protocols supersede the historical optimizer comparisons for new scientific claims. Historical defaults and raw outputs remain available for reproduction; their presence is not certification that the old comparisons answer the revised question.

## Implemented corrections

- `ccbo.matched_protocol`: one genuine noisy SCM row per paid intervention, fresh shared observations, arm-keyed initialization, learned scalar likelihood fitted before acquisition, affordable actions, and measured-data recommendations.
- `scripts/matched_ceo_runtime.py`: actual pinned CEO uses the same purchased row for its measurement and bookkeeping channels; population scoring is external.
- `scripts/matched_refinement.py`: graph-free HQCBO with paid atomic initialization of newly exposed arms and no graph access after initialization.
- `scripts/matched_fallback.py`: baseline FrontDoor graph with the actual nonidentified observational mean/variance fallback, retaining the original arms and causal covariance.
- `ccbo.qmcbo.corrected`: symmetric physical-coordinate mapping, constant-feature normalization, scalar fitting, deterministic acquisition base draws and isolated scoring; fine/full, fine/coarse-menu and quotient/coarse-menu comparisons.
- `ccbo.qdcbo.coherent` and `ccbo.qdcbo.population`: one union-parent conditional model per temporal cluster, joint-row held-out residual covariance, stochastic-prefix population response, manipulator-only history and independent scoring. The new nonstat_regularized setup is a separate bounded-denominator companion, not the original benchmark.

## Launch and analysis

Run experiments from immutable release snapshots. `scripts/run_frozen_unit.py` verifies manifest source hashes before execution, uses a full manifest-hash namespace, records status and output hashes, and refuses incomplete/stale resumes. Do not silently overwrite failed runs or reuse historical labels for changed objectives.

Static nominal budgets are 100 (ParallelParent/FrontDoor) and 120 (MediatedChain), with all initial/split measurements included. Engineering seeds are 1000–1004. The frozen static matrix contains 720 units over seeds 2000–2029. The strict static analyzer blocks inference for missing or invalid units and independently reconstructs recommendations and population scores. Pilot reports are engineering diagnostics only.

The corrected model-based main matrix uses Toy/PSA, 20 seeds (2000–2019), 100 sequential rounds and three model/menu configurations. Fixed rounds do not establish equal total intervention cost. Protected pairs must be separately executed. PSA remains intrinsically stochastic; score uncertainty must be reported.

The dynamic population protocol uses a finite-Monte-Carlo training oracle and independent population scoring. Its numerical integration precision and full performance matrix require the declared convergence/resource gates. Earlier smoke-test success is not a completed performance study.

The deployment target is the isolated HPC checkout `/zhome/15/b/215295/Repositories/ccbo-experiments-20260916`, with immutable releases below `releases/`. The original server checkout is preserved. HPC jobs use one CPU each and an overall concurrency ceiling of eight. Generated manifests, exact manuscript replacements, gate evidence and launch records are included in the accompanying experiment correction package.

## Manuscript obligations

Replace changed-protocol numerical figures and tables only after their full matrix passes audit. Primary noisy recommendations are selected by measured outcomes; their independently scored population quality need not be monotone. Do not relabel an oracle-best-visited curve as a learner recommendation. Report all prespecified methods, seeds, costs, failures and uncertainty, including null or reversed effects.

Legacy CBO-family Toy/Synthetic/Coral results remain historical until separately redesigned and validated. Current Overleaf source, Coral preprocessing lineage and redistribution records are still needed for final manuscript/release completion. See `paper/revisions/experiment-revision-tasks.md` for exact original-versus-replacement wording and acceptance criteria.
