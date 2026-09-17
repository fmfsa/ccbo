# Results and caption replacements: M13/M14/M16/M17/M22

These are anchored fragments for the supplied 20-page PDF. Current Overleaf source is absent. `results-and-captions.tex` must be applied block by block; it is not a main manuscript or a claim that the displayed PDF has been revised. Methods/protocol passages M05–M12, M15, M18 and detailed M22 provenance belong to the methods package and are not duplicated here.

## Exact application anchors

| Task | Old passage/location | Replacement action |
|---|---|---|
| M13 | End of §3.3 p.4 | Append HQCBO archive-identifier paragraph. |
| M13 | Figure 3 p.8: “the dashed vertical line marks refinement at trial 7.” | Replace with all-seed observed-trigger paragraph. |
| M13 | Figure 3 and Figure 5 p.16 graphic files/legends show both HQCBO and HQCBO-GF | Use regenerated `paper/figures/minimal_refine.pdf` and `paper/figures/cost_indexed.pdf`; displayed HQCBO maps only to `HQCBOGF`. Remove residual caption discussion of a main-paper graph-informed curve. Preserve the historical archive identifier in provenance. |
| M14 | Figure 5 caption ending “Curves stop at the last budget reached by every seed.” | Replace with all-seed support/initialization paragraph; replace image with regenerated cost PDF. |
| M14 | Table 4 p.17 caption starting “Cost accounting for the controlled suite…” | Replace entire caption with M14 caption block. Replace entire table body with `paper/tables/cost_accounting.tex`. |
| M14 | Discussion following Table 4 | Append Coral initialization/common-budget caveat. |
| M16/M17 | Table 3 p.15 caption promising “every method and condition”; column headed “Arm” | Replace caption with M16/M17 block and **entire** body with `paper/tables/minimal_ablations.tex`, whose header is already “Method” and which includes PP A2. |
| M16 | Appendix C p.14 controlled-results paragraph | Append method–condition inventory paragraph. |
| M17 | §5.2 “Figure 2(a,b)…”; §5.3 “Figure 2(c,d)…”; §3.1 duplicate Let | Apply `theory-edits.tex` blocks, not a second competing replacement. |
| M17 | Table 4 historical labels BOS, CBONP, QCBONP | Regenerated body provides BO-S, CBO−np, QCBO−np; do not hand-edit generated cells. |
| M22 | End of Appendix C | Add verified provenance paragraph; availability sentence remains blocked on actual public release fields as explained below. |
| M22 | End of §7 limitations | Append scope paragraph. |

Use the existing Overleaf `\input` path convention for the regenerated table bodies; do not nest their tabular environments inside another tabular. These fragments have no captions, so preserve surrounding table environments. Preserve existing figure/table labels or apply the explicit label mapping in `theory-edits.md`. The expanded Table 3 needs a final layout check after adding A2; passing standalone table builds is not a substitute for inspecting the assembled page.

## Verified numerical integration checkpoints

Evidence: `work/revision-run/reports/results.md`, regenerated summaries and assets. No manual numeric retyping is needed for table bodies.

- Graph-free HQCBO regret is 37.74380980782844±1.4122298068669712; mean total cost is 97.76666666666667, displayed 98. All 30 seeds split at trial 7; min=max=mean=7. The graph-informed comparator's 35.52±1.05 must not replace this result.
- PP A2 includes all six primary methods. A2 CBO final mean 0.0003389815005678897 and regret mean 3.660383402068064 differ from A0; copying A0 is incorrect.
- With the verified inventory, C* is 62, 50, 53, 63, 57, 95, 217 for PP A0, PP A1, FD B0, MC C0, Toy, Synthetic, Coral. Determining methods are respectively CEO; CBO/CBO−np; CBO/CBO−np; CBO/CBO−np; CBO; QCBO; QCBO. These values depend on included methods and seeds.
- Superseded PP A0 C*=73 becomes 62; Synthetic C*=100 becomes 95. Synthetic BO Y@C* becomes −0.570±0.020 and CBO becomes −1.248±0.288. Regenerate all associated cells together.
- Coral CBO initialization costs 800, so its Y@217 entry is n/a. This does not imply a failed run.
- Primary inventory has 43 method–condition groups / 1290 runs; historical validation additionally includes graph-informed HQCBO, making 44 / 1320. All applicable primary cells have 30 seeds. This inventory distinction belongs in reproduction instructions, not a claim that a fresh primary run recreates the historical comparator.

## Table 1 family-final source caveat

The supplied Table 1 has three columns (task, base final, quotient final). **No `family_finals.tex` exists in this snapshot.** The available generated `paper/tables/family_effects.tex` has a fourth paired finite-budget difference/95% CI column. It is not a drop-in structural replacement for Table 1.

Keep the existing three-column Table 1 unless explicitly choosing the four-column presentation. If choosing the latter, use the optional caption block and change the `scripts/emit_paired_effects.py` header from theoretical-looking `Δ` to “Paired difference [95% CI]”, then regenerate (this agent did not edit that separately owned emitter). The finite-budget difference is quotient-minus-base for minimization and base-minus-quotient for maximization. It includes optimization, initialization, action-space and stochastic effects and must not be identified with the population action-family price Δ(Π). A CI around this empirical difference is not a CI for theoretical Δ. Verify the emitter's CI convention when adding explanatory detail beyond the caption. Do not claim that a model-based score is an exact population expectation; its protocol is specified by the methods package.

## M22 release gate, without fabricated availability

The provenance fragment states verified engine revisions and archive-versus-rerun scope. It deliberately does **not** assert that a public release already exists and contains no bracketed release placeholders in compilable prose. The **public release URL and immutable release ID are still unresolved**. Once the release exists and its contents are verified, add a first sentence linking that real URL and naming its real immutable ID, then describe only the materials actually included. The historical engine hashes are not the reporting revision and a dirty baseline HEAD is not a final release identifier. A zip created during this local revision is not itself evidence of a published immutable release.

The methods package owns pinned dependencies, per-run protocol metadata, Coral origin/license and benchmark-only PSA interpretation; avoid duplicating or weakening those caveats. Preserve the distinction between running regression/smoke tests now and freshly rerunning full benchmark optimizers.

## Verification status

The captions and claims were checked against the results-agent report and actual regenerated table headers. Table 1's fourth-column mismatch was checked directly in `family_effects.tex`. The theory patch's bibliography metadata retrieval blocker is now closed with official BibTeX. Final caption/asset placement, rendered widths and actual labels still require current Overleaf source. No optimizer behavior or archived numerical outcome was changed by this package.
