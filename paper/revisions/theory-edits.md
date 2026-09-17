# A01/A08: formal and editorial manuscript patch

Status: apply-ready replacement package, independently checked against the supplied 20-page PDF. The current main manuscript and bibliography are maintained in Overleaf and are absent from this snapshot. No deleted historical manuscripts were restored. These files do **not** mean the assembled manuscript has been edited or its references resolved. Apply each corresponding block from `theory-edits.tex` at the anchors below; preserve the manuscript's existing theorem environments and labels (the paragraph headings here mark their replacement content).

## Exact replacement anchors

| Task | Supplied PDF anchor (line breaks normalized) | Operation |
|---|---|---|
| M01 | §2.1 p.2: “where D(X) is the intervention domain.” | Replace that sentence fragment with M01 block. |
| M02 | §4.2 p.5: “The MIS-reduction result in Appendix A connects this action-family value to the quotient exploration set.” | Replace with first M02 paragraph. |
| M02 | Appendix A.3 pp.11–12, all of Lemma 1, its proof, and the immediately following “Consequently, when the null intervention is handled consistently…” paragraph | Replace with full M02 lemma/proof/quotient/counterexample block. Preserve the following chain example. |
| M03 | §3.2 p.4, display P(V_C | V_pa(C)) through “within a cluster.” | Replace with structural-kernel block. Retain subsequent GP implementation paragraph. |
| M03 | Appendix A.1 p.11, all of Proposition 4 and its proof | Replace with M03 proposition/proof block. |
| M04 | §3.1 pp.3–4, “The singleton partition recovers…” through “Appendix A.” | Replace with first M04 block. |
| M04 | §4.1 p.4, “At the singleton partition, they recover their base methods (Proposition 3).” | Replace with second M04 block. |
| M04 | Appendix A.1 p.11, Proposition 3 and its proof | Replace with construction-level statement/proof plus actual singleton-noise distinction. |
| M17 | §5.2 p.7, “Figure 2(a,b) shows the consequence.” | Replace with first M17 block. |
| M17 | §5.3 p.7, “Figure 2(c,d) shows a different failure.” | Replace with second M17 block. |
| M17 | §3.1 p.3, “extension here. Let / Let Π_X” | Delete only the first duplicated “Let”. |
| M21 | §6 p.8, from “A setting related to ours is causally abstracted MABs (CAMABs)” through the final “unknown.” | Replace with CAMAB block. |
| M21 | §6 model-based discussion | Insert vector-valued mechanism paragraph. |
| M21 | §4.4 p.6 immediately after “These conditions are sufficient but not necessary.” | Insert original-CBO domain paragraph. |

Add `\label{fig:controlled-misspecification}` immediately after Figure 2's caption and `\label{tab:controlled-complete}` immediately after Table 3's caption, unless equivalent labels already exist, in which case substitute those actual keys in both replacement sentences. Figure 2 has only panels (a)/(b); Figure 3 has its own (a)/(b). Do not apply the Figure 2 edit to Figure 3. Table publication labels BO-S, CBO−np, QCBO−np belong to the result-emitter work package; no emitter was edited here.

## Independent mathematical checks

1. **Projection closure versus completion support.** Product domains imply admissibility of restrictions used in MIS reduction. They do not imply that natural values of added variables belong to their intervention domains. In particular the counterexample deliberately puts the typical natural A near zero while D_A=[1,2]. No domains or experiment protocols were changed.
2. **MIS reduction.** The last-removed-node argument proves absence of any causal route from removed nodes after holding the retained nodes fixed. It requires the graph to represent the SCM used to evaluate effects. Bidirected edges do not create directed causal routes; latent confounding does not invalidate this reduction. The quotient proof only needs true projected paths and valid cluster order, and is not claiming arbitrary vector SCM causal sufficiency.
3. **Null and nonempty values.** The set inclusion gives V≤W. Together with min(μ_empty,W)=min(μ_empty,V), W≤μ_empty forces equality, including the equality boundary W=μ_empty. If W>μ_empty the identity permits V<W. The explicit independent-root example proves V=.01 and W=1. Its joint arm {A,Z} also has value 1; its only retained arm is {A}.
4. **Structural kernels.** The replacement explicitly assumes an acyclic Markovian SCM/independent node disturbances. A valid quotient permits composing internal kernels in cluster order. Observational conditionals identify the kernels only almost surely in parent values; “on support” without this qualifier is too strong for arbitrary versions of continuous conditionals. The structural identity is not an off-support observational identification theorem.
5. **Proposition 5 remains valid.** Take c=1 and independent standard normal ε₂, ε₃, ε_Y. The first model X₂=ε₂, X₃=X₂+ε₃ has covariance [[1,1],[1,2]]. The reverse model X₃=√2 η₃, X₂=X₃/2+η₂/√2 has the same joint root-cluster law. In both Y=X₂+ε_Y. The full structural root-cluster kernel and the Y kernel agree at every parent value, and all whole-cluster laws agree. Under do(X₃=x), the Y means are respectively 0 and x/2. This validates the proposition with SCM-defined kernels, not merely observed conditionals.
6. **Downstream results.** Proposition 2 uses nested nonempty action families and attained minima; unchanged. Theorem 1(i) explicitly limits eligible incumbents to executed MIS interventions; its ParallelParent witness remains valid. Part (ii) is the algebraic decomposition plus quotient invariance, not an assertion of convergence or nonempty MIS optimality. Corollary 1 follows from independent exogenous groups, integration and completion support; it concerns U(Π), not MIS optimality. Corollary 2 requires an exposed optimal arm and consistency; its Cesàro argument remains valid. Corollary 3 remains a same-input/same-randomness argument. No further assumptions are silently added to any benchmark.
7. **Executed incumbents.** `ccbo/cbo/cbo.py` observation branch appends the preceding incumbent and cost and logs arm/x as None; observations do not introduce a null recommendation. `compute_MIS` remains unchanged.
8. **Singleton distinction.** Graphs, action construction and factorization reduce at singleton resolution. They do not force matching noise models or numerical trajectories. The snapshot's `JointQuotientGPNetwork._fit_cluster` uses learned-noise SingleTaskGP. The parent task checked pinned upstream `mcbo/models/gp_network.py` and found FixedNoiseGP supplied with additive disturbance variances. The replacement explicitly disclaims executable singleton equivalence; no model change or archived-result rerun is prescribed by this wording correction. The parent task's one-fit numerical probe (`server/validation/singleton_mcbo_probe.json`, 40 training points, data seed 7, fit seed 123) also found maximum posterior mean difference 0.0356192016 and maximum variance difference 0.0059380402 across three nodes. This is a counterexample to exact executable identity, not a universal bound or benchmark effect estimate.

The search of the complete extracted PDF for `recovers`, `MIS reduction`, `same optimum`, `reduce to`, and `singleton` identified the three M04 sites and the M02 sites above. Section 4.5, Discussion and Conclusion already qualify refinement recovery. The original source-wide search and final assembled-PDF check remain pending receipt of the Overleaf source.

## Bibliography merge and primary-source evidence

`references-corrections.bib` contains proposed canonical keys. Merge into the actual bibliography; map **all old citation keys**, not just the visible sites, before deleting duplicates. Numbered references below refer only to the supplied PDF and will change after BibTeX runs.

| Existing PDF entries | Canonical key | Action and primary evidence |
|---|---|---|
| [3] | `anand2025clusterdiscovery` | Correct year to 2025, volume 38; official [NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7d998ff3e6ecce821986c9dd876be9bc-Abstract-Conference.html) verifies authors/title/year/DOI. |
| [4] and [5] | `anand2023clusterdags` | Merge as one work, correcting [4]'s title/volume: [AAAI official record](https://ojs.aaai.org/index.php/AAAI/article/view/26435) gives 37(10), 12172–12179, 2023 and DOI 10.1609/aaai.v37i10.26435. |
| [13] and [14] | `lee2018structuralbandits` | Merge as one 2018 work with volume 31; [official NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2018/hash/c0a271bc0ecb776a094786474322cb82-Abstract.html). |
| [9] | `dyer2025atucb` | [Author manuscript](https://arxiv.org/html/2509.04296v1), §3 Algorithm 1/Proposition 1, verifies abstract action filtering and optimal-reward relation. Entry uses verified arXiv version; workshop metadata was not independently checked and is not fabricated. |
| [29] | `zennaro2024camab` | [Official PMLR record](https://proceedings.mlr.press/v244/zennaro24a.html) verifies UAI 2024, volume 244, 4109–4139. |
| [27] | `sussex2023mcbo` | [Author manuscript](https://arxiv.org/html/2211.10257), §2 and Appendix A.1, explicitly supports vector output, dependent output kernels and component-index GP modeling. Existing ICLR 2023 citation retained; official OpenReview landing page presented a browser verification challenge during this audit. |
| [1] | `aglietti2020cbo` | [PMLR record](https://proceedings.mlr.press/v108/aglietti20a.html) validates metadata; [paper §4.2](https://proceedings.mlr.press/v108/aglietti20a/aglietti20a.pdf) explicitly discusses mediator values outside intervention domains. |

The parent task subsequently retrieved the official NeurIPS BibTeX into `work/revision-run/server/neurips2025.bib` and `neurips2018.bib`. Both records are now merged verbatim except for canonical citation keys and omission of the empty 2018 pages field. The 2025 record specifies pages 86792–86819 and volume `38, Main Conference`; its editor/publisher metadata are retained. The 2018 official record supplies no page range, so none is guessed from conflicting secondary sources. The retrieval blocker is closed. No DOI is invented where the official record does not supply one. Unrelated references, including legitimate 2026 works, are untouched. Verification is scoped to changed/cited records, not a fresh audit of all 29 references.

## Validation and remaining assembly work

A focused regression test protects the nonempty action contract; it uses the real `compute_MIS` and exact analytical moments. Command from the revision snapshot root:

```sh
PYTHONPATH=. /Users/fmfsa/Repositories/ccbo/.venv/bin/python -m pytest ccbo/tests/test_nonempty_mis_contract.py -q
```

Result: 1 passed, no skips. This finite test documents the counterexample; the proof above carries the universal result. A standalone LaTeX smoke build checks the replacement fragments and all seven BibTeX records. Its dummy Figure/Table labels test syntax only, not actual Overleaf cross-reference resolution. A final source merge, bibliography-key remapping, publication-layout render and updated PDF audit remain necessary. Numerical experimental results have not changed. No optimizer rerun is needed for these corrections.

Final patch-text audit after bibliography merge: all replacement anchors checked against the supplied PDF; corrected Figure 2 references use only (a)/(b); all seven canonical bibliography keys remain unique; the four keys cited by the replacement text resolve. Claims of null-augmented optimality retain graph/domain assumptions, refinement remains conditional, and the singleton claim remains construction-only. Current Overleaf source is still unavailable.
