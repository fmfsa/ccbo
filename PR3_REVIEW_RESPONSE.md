## Response to review — all six findings addressed; CEO rerun complete

**Finding 1 (P0, merge-blocker) — fixed and rerun.** `make_target_factory` now branches on `noisy`: the noisy path (CEO's GP targets + graph-posterior evidence) draws genuine stochastic samples from the **true latent SEM** (confounders included, then discarded) through CEO's own stateful per-seed `RandomState`; the noiseless path (trajectory scoring only) keeps the closed-form E[Y|do] oracle and consumes no randomness. `initial_interventional_data` now returns a noisy/noiseless pair at shared levels. Guarded by 11 new tests (`ccbo/tests/test_ceo_minimal_protocol.py`): noisy draws vary and differ from the oracle; noiseless equals the oracle exactly with zero RNG consumption; MC mean of noisy draws matches the oracle; alias metadata contract. **All 150 units rerun** (LSF 29090857, zero failures) and the table/JSON regenerated. Corrected numbers (n=30): CEO still survives A1 (0.77 vs committed CBO's 4.245) but its premium over QCBO is larger than the flawed run suggested (+0.71…+1.00 on PP, +0.07 on FD, all CIs excluding 0); MC unchanged (0.135 vs 4.000; HQCBO 0.040). Manuscript numbers updated throughout.

**Finding 2 — fixed.** Column renamed to $P(\text{proj})$: posterior mass on the pool member equal to the true graph's *observable projection* (caption states explicitly that the true PP/FD graphs carry bidirected edges and are in no pool); reported for multi-graph pools only ("--" for singletons); "never resolves the truth" phrasing removed (now: mass on the observable projection plateaus near 0.54). Meta field renamed `baseline_observable_cond`.

**Finding 3 — fixed.** `reproduce_paper.sh` now extracts + checksums `ceo_minimal.tar.gz`, runs the new `verify_ceo` gate section (grid 150+60, trajectory lengths 51/51/61, alias byte-identity + alias metadata, protocol constants, pool composition, recomputed finals vs published values), and runs `emit_ceo_comparison.py` with explicit workdir paths. Full gate: **54 checks, 0 failures**; end-to-end pipeline green.

**Finding 4 — fixed (the "ideal" variant).** The rerun persists per-trial exploration sets and costs in each meta sidecar; `emit_cost_analysis.py` adds CEO rows (init = 12 units, logged, total, Y at matched C*) to the MinimalBench groups of the App. E table. §5.4 and the App. F caption now state precisely what is and is not shared: shared — seeds, trial counts, three init points per arm, observational data, objective surface; not shared — total initialization cost (12 vs 6 on PP), non-cost-weighted acquisition, and stochastic-draw learning vs CRN-MC mean evaluations (asymmetry disclosed).

**Finding 5 — fixed.** `materialize_aliases` sets `meta["cond"]` to the alias with `alias_of` preserving the source; covered by a test and by the `verify_ceo` gate.

**Finding 6 — fixed.** Reproducibility statement says five archives; MANIFEST header updated to the actual current gate count (54 checks, 2026-08-11) and the CEO section carries an explicit supersession note (job 29078320 invalidated by finding 1 → corrected job 29090857); pytest baseline note updated (63/11/11 including the new protocol tests).

QMCBO work untouched, as approved. Ready for re-review.

## Second-round corrections (text-only, no rerun)

**CEO's acquisition IS cost-weighted** — pinned commit `35dd277` divides EI/CES/manual-EI by a per-variable cost (`cost_type=1`, which the runner keeps). "Non-cost-weighted" replaced in §5.4, the Table 7 caption, and the cost emitter with "a different acquisition objective, with the same per-variable cost normalization."

**`P(proj)` renamed `P(A0)`** — an observable/latent projection is an ADMG that retains bidirected edges, so no pool member equals it. The column is now defined as posterior mass on the **baseline observable DAG** (the truth's directed edge set with bidirected edges dropped); values unchanged. Emitter function/JSON key renamed accordingly (`baseline_mass`, `posterior_mass_on_directed_baseline_mean`).

Minor: `envs/README.md` page facts corrected (22 pages, refs from PDF p. 10); MANIFEST packing paragraph now states per-suite packing dates.

*(This file is PR correspondence — drop it before merge.)*
