# Compile dependencies assembled separately

`dependencies/` contains all 11 directly referenced figure/table files, shared `figures/styles.tex`, historical `aistats2026.sty` and `fancyhdr.sty`, and reconstructed `qcbo_aistats.bib` (15 files total). `dependencies-manifest.json` records exact SHA-256 values, origins, HEAD commit and citation keys. No deleted user-repository file was restored. No manuscript source or HPC job was edited. The styles are exact git HEAD blobs; standard `abbrvnat.bst` is supplied by TeX Live and passed the BibTeX check.

The absent original `qcbo_aistats.bib` was not present in git history. The supplied PDF bibliography (pages 9–10) establishes missing works; historical `references.bib` / `updated.bib` supply available records. Previously verified canonical corrections were remapped to the existing user keys. Newly reconstructed supplemental entries are retained in `supplement.bib`; helper `assemble_dependencies.py` is reproducible.

## Required main-source substitutions (root owns main edits)

- `Anand2023` → `CausalEffectIdentification` (one AAAI 2023 cluster-DAG identification entry).
- `leeStructuralCausalBandits` → `lee2018structural` (one NeurIPS 2018 entry).

These are saved in `citekey-remap.json`. All other supplied citation keys remain. In particular `anand2026causal` remains its user key, but publication year is corrected to **2025**. After the two substitutions all 29 original distinct keys resolve to 27 distinct bibliography entries. No duplicate aliases were added. BibTeX using `abbrvnat`, with all entries requested, exits successfully without warnings (`bib-qa/check.blg`). This verifies parsing, not a complete literature audit.

The verified prior corrections update CBO/PMLR, MCBO/ICLR, Anand/AAAI, Anand/NeurIPS2025, Lee/NeurIPS2018, CAMAB/UAI2024 and AT-UCB/arXiv2025. The former references already identify those exact works.

Supplemental primary checks:

- Parviainen/Kaski journal metadata and DOI: https://www.sciencedirect.com/science/article/pii/S0888613X17303134
- Bareinboim/Forney/Pearl: https://proceedings.neurips.cc/paper_files/paper/2015/hash/795c7a7a5ec6b460ec00c5841019b9e9-Abstract.html
- Lattimore et al.: https://proceedings.neurips.cc/paper/2016/hash/b4288d9c0ec0a1841b3b3728321e7088-Abstract.html
- Nair et al.: https://proceedings.mlr.press/v130/nair21a.html
- Rubenstein et al.: https://is.mpg.de/hi/en/publications/rubensteinetal17 (author institution, linking original UAI proceedings).
- Ninad et al.: https://arxiv.org/abs/2505.10476
- Schooltink/Zennaro: https://arxiv.org/abs/2412.17080 (author is **Willem**, not William).
- Konobeev et al.: https://arxiv.org/abs/2301.11401 (preserve original cited 2023 preprint; do not silently switch to 2025 proceedings).
- Ferreira/Assaad: https://arxiv.org/abs/2504.01551 (exact current preprint title “Identifying Macro Causal Effects in a C-DMG over ADMGs”; distinct from the differently titled NeurIPS git entry, which was not substituted).

`park2025structural` uses author/title from historical source and year 2025 from supplied PDF [22], represented as `@misc`. Its venue/identifier was not independently established here and has deliberately not been invented. Remaining unchanged historical records retain their metadata; full bibliography certification is outside this bounded dependency assembly.

## Figure/table semantics and layout constraints

The copied curated `figures/dataset_dags.tex` already has verified PSA manipulator colors/caption and ParallelParent/CompleteGraph ADMG caption repairs. It retains historical Figure11 unchanged, so root must handle stationary-only scope and the separate companion figure. None of these dependency copies claims empirical completion.

The copied result PDFs/tables are **historical**. They are compile assets only, not new noisy/population/MCBO results. Root decides replacement or placeholder handling. `tables/family_effects.tex` currently has **four columns** (Task, Base, Quotient, paired Δ/CI), whereas reviewed original Table1 had three. The supplied main-source wrapper contains only `adjustbox{...}{input{...}}`, so four columns cause no TeX alignment error, but they change displayed architecture and require either a deliberate caption/header update or a derived three-column dependency. No table content was stripped here. `cost_accounting`, `minimal_ablations` and `qmcbo_e2` also remain historical, with potentially obsolete HQCBO labels/protocol interpretations.

The main source already defines TikZ node/edge styles; `styles.tex` is provided as a dependency convenience but does not need a second input if definitions conflict. Preserve the user's existing packages, wrappers, widths and labels. Newly verified companion PDF/source/caption remains separately available in `outputs/dynamic-regularized-graph.*`; it was not silently substituted for an existing filename.
