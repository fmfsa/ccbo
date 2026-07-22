# Server run TODO — QCBO revision (branch `claude/qcbo-paper-review-528309`)

**Context.** The paper + code revision has landed on this branch: exploration
sets are the MIS of the (C-)DAG (Lee & Bareinboim rule, no size cap),
identifiability decides each arm's *prior tier* (do-calculus vs common
uninformative prior) and never its membership, PA-GAP is deleted, metrics are
final-Y/regret-primary, QMCBO is joint-mechanism only, and the base stacks
are the original authors' repos. The paper (`paper/qcbo_aistats.tex`) builds
clean; only the experiment suites and the numeric refresh of §5.5/§5.6
remain. Everything below is scripted and **deterministic (fixed seeds)** —
run from scratch; per-unit resume is automatic (existing complete CSVs are
skipped).

Reference finals from a partial local run (should reproduce):
Toy CBO −2.164±0.000, Toy QCBO −2.131±0.033 (single arm, uninformative
tier — the runner logs `[uninformative prior: XZ]`); CompleteGraph CBO
−3.139±0.190, QCBO −1.245±0.068.

## 0. Environments (once)

```bash
# main env
conda create -y -n ccbo python=3.10
conda run -n ccbo pip install -e .            # pyproject: numpy<2, GPy, emukit, ananke-causal, ...
conda run -n ccbo pip install graphviz pygraphviz   # pygraphviz needs system graphviz (apt/brew)

# era-pinned env for the MCBO authors' stack (old botorch API)
conda create -y -n mcbo python=3.9
conda run -n mcbo pip install "numpy<2" "torch==1.13.1" "gpytorch==1.9.0" \
    "botorch==0.7.2" pandas scipy matplotlib wandb

# authors' stacks (pinned SHAs inside the scripts: DCBO 85a9bdf, mcbo 0d0650e)
bash scripts/fetch_dcbo.sh
bash scripts/fetch_mcbo.sh
```

## 1. Sanity gates (fast — run before the suites)

```bash
conda run -n ccbo python -m pytest
# expect: 47 passed, 3 skipped, 3 deselected
#   (the 3 skips are QMCBO joint-net tests: they need the era botorch — fine)
PYTHONPATH=. conda run -n ccbo python -m pytest ccbo/tests/test_qdcbo.py -q
# expect after fetch_dcbo.sh: 8 passed (finest-partition identity + E2 invariance)
```

## 2. The three suites (LONG — scale jobs to cores; all resumable)

```bash
# a) CBO vs QCBO family (Toy/Complete/SimplifiedCoral, 10 seeds, 40 trials)
#    Coral CBO has 31 MIS arms — its units dominate wall-clock.
PYTHONPATH=. conda run -n ccbo python scripts/run_cbo_family.py \
    --seeds 10 --trials 40 --outdir results/family_cbo --jobs <N>

# b) DCBO vs QDCBO, 20 seeds, stat/ind/nonstat, E1+E2
#    (runner self-re-execs with PYTHONHASHSEED=0; nothing to set)
PYTHONPATH=. PY="$(conda run -n ccbo which python)" \
    bash scripts/run_qdcbo_pilot.sh results/qdcbo <N>

# c) MCBO vs QMCBO, 5 seeds, ToyGraph/PSAGraph, T=100, E1+E2  [LONGEST]
#    PSAGraph MCBO units are hours each; budget ~1-2 days at 8 jobs.
PYTHONPATH=. PY="$(conda run -n mcbo which python)" \
    bash scripts/run_qmcbo_pilot.sh results/qmcbo <N>
```

## 3. Diagnostic (cheap to launch, slow to finish; run in parallel)

```bash
# Old gate-deletion rule vs new MIS/two-tier rule, per ClusterBench10
# variant/partition — input for the release-time CB10 rerun decision.
PYTHONPATH=. conda run -n ccbo python scripts/audit_es_change.py \
    | tee results/es_audit.txt
```

## 4. Aggregate figures + tables (after step 2 completes)

```bash
PYTHONPATH=. conda run -n ccbo python scripts/plot_family_suite.py
#   -> paper/figures/family_suite.pdf   (3-row grid; needs all three suites)
PYTHONPATH=. conda run -n ccbo python scripts/emit_family_price_table.py
#   -> paper/tables/family_price.tex
PYTHONPATH=. conda run -n ccbo python scripts/emit_qmcbo_tables.py
#   -> paper/tables/qmcbo_e1.tex, qmcbo_e2.tex, results/qmcbo_pilot.json
```

## 5. Paper numeric refresh (Track B) — `paper/qcbo_aistats.tex`

- **§5.5** (`sec:exp-qmcbo`): refresh the MCBO/QMCBO finals and the E2
  identical-trajectory counts from the new runs; replace "about a third of
  the wall-clock" with the measured ratio (s/unit column of
  `qmcbo_e1.tex`); re-verify the ToyGraph optimum claim
  ($2.172\pm0.000$).
- **§5.6** (`sec:exp-family`): recount "six of the eight experiments"
  against the regenerated `family_price.tex`; verify the Toy sentence
  ("…and still matches \CBO{}") holds in the new numbers.
- **App. F** (`app:wrongpi`): verify "GAP stays within $0.06$" against
  `clusterbench10_wrongpi.tex`.
- Rebuild: `cd paper && latexmk -pdf qcbo_aistats.tex` — must exit 0 with
  **zero** LaTeX errors and **zero** undefined references.

## 6. Final verification sweep

```bash
conda run -n ccbo python -m pytest                      # green (3 QMCBO skips ok)
# zero hits allowed in paper/qcbo_aistats.tex and paper/tables/:
grep -rn "PA-GAP\|byte-for-byte\|byte-identical\|uncapped\|cluster budget\|QMCBO-ind\|QMCBOJ\|ten-dimensional\|quarter of the wall" \
    paper/qcbo_aistats.tex paper/tables/
# CausalBO_Benchmark may appear ONLY in retirement/provenance notes:
grep -rn "CausalBO_Benchmark" ccbo/ scripts/ README.md
```

## 7. Deferred — release-time (tracked in KNOWN_ISSUES.md; do NOT block)

- Port the ClusterBench10 dataset loader off the retired `DiscoveredGraph`
  (the SEM already lives in `scripts/generate_clusterbench10.py`); then
  rerun every headline ClusterBench10 condition on the released stack and
  regenerate the CB10 tables (the paper carries a provenance note until
  then, App. D).
- Add the sharpened arm-deletion variant (edge edits that change the
  fine-graph MIS while the C-DAG is untouched) — design in KNOWN_ISSUES.md.
