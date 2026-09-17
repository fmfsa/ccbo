# Overleaf edit sheet: exact replacements and asset swaps

Prepared 16 September 2026 against **Coarsenings_in_Causal_Bayesian_Optimization (11).pdf**, 20 pages. This is the consolidated manual handoff; no Overleaf access or source upload is needed to apply it. Page numbers and quoted anchors refer to that PDF. Typography, line-break hyphenation and extraction spacing in quotations are normalized. Long edits give an exact start/end span; replace the entire span, not just its first sentence.

This sheet supersedes conflicting wording in the earlier `revision-plan.md`, `theory-edits.*`, `methods-and-protocol.*` and `results-and-captions.*`. It follows the corrected frozen experimental protocols. It does **not** certify that the new campaign has completed or that old result files are valid for it.

## How to apply

1. Apply text edits T01–T30 below. Each provides **Before**, an operation boundary, and a LaTeX **After** block. Insertions explicitly say when no old text exists. Keep the manuscript's real theorem labels and citation keys, remapping to the supplied canonical keys as needed.
2. Apply the complete figure/table map after the text edits. Upload the ready files from the ZIP; do not input the entire reference gallery. Let LaTeX renumber all later figures/tables.
3. **Pending result assets are not supplied.** Their exact future filenames, estimands, schemas and captions are specified. Old values must not accompany the new methods. The result sections supplied now state valid analytic consequences and evaluation design; add empirical effect paragraphs only after the complete relevant audit passes.
4. Resolve the two code-release fields and compile the assembled paper. The included fragments have a syntax check, but the complete Overleaf layout, label resolution and venue length have not been checked in your source.

## What is ready and what remains

| Category | Status |
|---|---|
| Product domains; MIS/null counterexample and proof; structural kernels; qualified singleton identity | Demonstrated logical/definition repairs; ready to apply. |
| Acquisition, noise, scoring, initialization, paid refinement, menus and coherent dynamics | Implementation/protocol alignment; ready to apply as methods, with numerical results still pending. |
| Empirical superiority, FrontDoor prior-only effect size, refinement success, complete-trace equality counts | Unresolved empirical quantities; no numerical conclusion is supplied or assumed. |
| PSA colors; stationary graph scope; regularized companion graph; structural Table2 | Ready assets in the ZIP. |
| Historical CBO-family/Coral evidence | Excluded from the corrected primary comparison. The archive remains intact; reintroduction requires its own validated protocol/provenance. |
| Public release URL and immutable release ID | Missing information; publication completion items, not invented values. |

The three specialist follow-ups were reconciled into this single sheet. Overlapping §3.2 drafts are merged into one structural-kernel-plus-implementation replacement. The older scalar-noise mismatch is described as historical; the corrected comparison matches the scalar fitting policy. The mathematical regret decomposition remains valid even when noisy recommendation quality worsens. Conditional refinement recovery remains unproved for that particular recommendation rule.

## Text edit index

- [ ] **T01:** Abstract: replace the complete abstract (PDF p.1)
- [ ] **T02:** Introduction: replace the contribution list (PDF p.2)
- [ ] **T03:** Product domains and null convention (Section 2.1, PDF p.2)
- [ ] **T04:** Scope of the implementation construction (Section 3 opening, PDF p.3)
- [ ] **T05:** Singleton scope: two main-text replacements (PDF pp.3–4)
- [ ] **T06:** Dynamic and model-based variants: replace all of §3.2 (p.4)
- [ ] **T07:** Adaptive refinement: replace the final paragraph of §3.3 (p.4)
- [ ] **T08:** Temporal invariance scope: local replacement in §4.1 (p.4)
- [ ] **T09:** MIS/nonempty value connection (Section 4.2, PDF p.5)
- [ ] **T10:** Executed recommendations and Eqs.(10)–(11) (Section 4.3, PDF p.5)
- [ ] **T11:** Keep refinement recovery explicitly conditional (Section 4.5, PDF p.6)
- [ ] **T12:** Evaluation design: replace all of §5.1 (p.6)
- [ ] **T13:** ParallelParent: replace the empirical paragraph in §5.2 (p.7)
- [ ] **T14:** FrontDoor: replace the empirical paragraph and add domains in §5.3 (p.7)
- [ ] **T15:** MediatedChain: complete its SCM and replace the floor sentence in §5.4 (p.7)
- [ ] **T16:** Replace §5.5 and its heading (pp.7–8)
- [ ] **T17:** Related work: three local edits (§4.4 p.6; §6 p.8)
- [ ] **T18:** Discussion and limitations: replace the complete §7 body (p.8)
- [ ] **T19:** Conclusion: replace the complete paragraph (Section 8, PDF p.8)
- [ ] **T20:** Full singleton proposition and proof (Appendix A.1, PDF p.11)
- [ ] **T21:** Full structural-kernel proposition and proof (Appendix A.1, PDF p.11)
- [ ] **T22:** Full MIS lemma, proof, quotient consequence and counterexample (Appendix A.3, PDF pp.11–12)
- [ ] **T23:** Appendix B.1 — replace the complete subsection (pp.13–14)
- [ ] **T24:** Appendix B.2 — replace the complete subsection (p.14)
- [ ] **T25:** Insert new Appendix B.3 before Appendix C
- [ ] **T26:** Appendix C — replace “Controlled suite.” paragraph (PDF p.14)
- [ ] **T27:** Appendix C — CEO paragraph (insert after matched static protocol)
- [ ] **T28:** Appendix C — replace the complete old “Family suite.” paragraph, part 1 (p.14)
- [ ] **T29:** Appendix C — replace the complete old “Family suite.” paragraph, part 2 (p.14)
- [ ] **T30:** Appendix C: add exact domain, configuration and reporting details (p.14)

## T01 — Abstract: replace the complete abstract (PDF p.1)

**Before — exact opening and closing anchors:** starts “Choosing effective interventions in complex systems often requires costly experiments.” Ends “Our results show how coarse causal knowledge can support intervention optimization while making explicit the tradeoff between robustness to graph errors and flexibility in choosing interventions.” Replace everything between and including these sentences, inside the existing abstract environment.

**After:**

```latex
Choosing effective interventions in complex systems often requires costly
experiments. Causal Bayesian optimization uses a supplied causal graph,
but errors in that graph can alter causal priors or exclude useful
interventions. We study optimizers that instead use a supplied partition
and its cluster graph, with interventions restricted to complete clusters.
When every graph-dependent operation uses only this quotient, coupled runs
are identical under graph changes that preserve it. This invariance does
not guarantee accurate predictions or good decisions. We characterize the
population loss from restricting the action family, show that it cannot
increase under refinement, and give sufficient structural conditions for
zero loss. We also distinguish the value of all nonempty cluster-union
actions from the value of retained minimal intervention sets, and separate
structural singleton identity from numerical equivalence of implementations.
These results provide a framework for comparing graph robustness, action
restrictions and finite-budget learning under explicitly matched feedback
and recommendation protocols.
```

Do not reinsert “our results show” or claim superiority across three families until the corresponding immutable final matrices pass their analyzers. A later validated result sentence can be added without changing these theoretical claims.

## T02 — Introduction: replace the contribution list (PDF p.2)

**Before:** from “We make three contributions:” through item 3, “We use controlled experiments to separate errors that remove useful interventions from errors that alter causal priors, and evaluate the construction across optimizer families.” Leave the following “Proofs, complete dataset graphs, and additional results are provided in the appendices.” sentence in place.

**After:**

```latex
We make three contributions:
\begin{enumerate}
\item We construct quotient-based causal optimizers that use a supplied
cluster graph and intervene on complete clusters. We state the common
input and implementation conditions needed for pathwise invariance.
\item We characterize the population loss from restricting the nonempty
action family and give sufficient conditions for zero loss. We make
explicit the null-intervention condition needed to connect this value to
minimal-intervention-set reduction, and distinguish structural kernels
from their observational estimates.
\item We provide a controlled evaluation design that separates loss of
available actions, changes in causal priors, and model construction.
Matched feedback, explicit initialization and intervention costs, and
independent scoring of implementable recommendations distinguish these
comparisons from idealized exact-response examples.
\end{enumerate}
```

The preceding paragraph ending “therefore changes the C-DAG consumed by QCBO” is already appropriately limited to quotient-preserving errors; retain it.

## T03 — Product domains and null convention (Section 2.1, PDF p.2)

**Before:** “where $D(X)$ is the intervention domain.” This follows the definition of a **nonempty** intervention set. Replace that sentence only; preserve Eq.(1).

**After:**

```latex
where $D(X)$ is the intervention domain. We use fixed nonempty per-variable
intervention domains $D_v$ and set $D(X)=\prod_{v\in X}D_v$.
Thus the restriction of an admissible assignment to any subset of its
intervened variables remains admissible. These domains need not contain
all naturally attainable values; the stronger completion-support condition
is stated separately in Definition~1.
```

This projection convention makes the restriction used in Lemma 1 admissible. It does **not** assume natural completion values lie in intervention domains; Definition 1 supplies that separate, stronger condition. Keep the nonempty conventions in Eqs.(2), (6), (8) and (9).

## T04 — Scope of the implementation construction (Section 3 opening, PDF p.3)

**Before:** “In each case, only the graph-dependent components of the base optimizer are replaced; its acquisition and update rules are otherwise retained.”

**After:**

```latex
Within each declared comparison, the fine and quotient methods share the
specified acquisition and update policy. Numerical corrections shared by
both comparators, and the coherent dynamic conditional model, are stated
separately in the experimental protocol. These corrected comparisons are
not claims of numerical identity with every released implementation of
the original algorithms.
```

Also remove the duplicate “Let” between “we do not implement that extension here.” and “Let $\Pi_{\mathcal X}$” in Section 3.1, p.3.

## T05 — Singleton scope: two main-text replacements (PDF pp.3–4)

**5a Before:** Section 3.1, starting p.3 and ending p.4: “The singleton partition recovers the fine graph, the CBO exploration set, and the CBO prior construction arm by arm. A formal statement and proof appear in Appendix A.”

**After:**

```latex
At the singleton partition, the graph and intervention construction reduce
to the fine graph and its nonempty MIS. QCBO's prior construction agrees
arm by arm with the matched CBO implementation when it uses the same
estimators, observation policy and numerical settings. This is not an
identity claim for every original CBO MIS/POMIS configuration.
Appendix~A distinguishes construction identity from executable identity.
```

**5b Before:** Section 4.1, p.4: “At the singleton partition, they recover their base methods (Proposition 3).” Replace only this sentence, after the quotient-contract discussion.

**After:**

```latex
At the singleton partition, the quotient graph and action construction
reduce to their fine counterparts. Proposition~3 separates this structural
identity from the matching implementation conditions needed for numerical
baseline equivalence.
```

## T06 — Dynamic and model-based variants: replace all of §3.2 (p.4)

**Before — complete section boundary:** from the heading “Dynamic and model-based quotient variants” and “Quotient Dynamic CBO (QDCBO) replaces every time-slice and its temporal transitions by their quotients” through “Appendix A states and proves both facts.” Stop before §3.3. This single block replaces both old method paragraphs and the soundness paragraph.

**After:**

```latex
\subsection{Dynamic and model-based quotient variants}
Quotient model-based CBO (QMCBO) replaces MCBO node mechanisms~\cite{sussex2023mcbo} by structural
cluster kernels
\[
 K_C(dv_C\mid v_{\operatorname{pa}(C)}),
\]
and composes them along the C-DAG. Here $v_C$ is the vector of member
variables and $\operatorname{pa}(C)$ is the set of parent clusters.
The structural kernel $K_C$ is induced by composing the member-level SCM
mechanisms within $C$ at the specified parent-cluster values. Under causal
sufficiency and a valid partition, it is a version of the observational
conditional, agreeing almost surely in the observational parent distribution.
At other parent values the notation refers to the SCM-defined kernel;
observational data alone need not determine its extension, even at
individual zero-probability values in the support. GP extrapolation beyond
observed support is a modeling assumption. Joint cluster mechanisms preserve
within-cluster dependence.
We approximate these kernels using multi-output GP models for multivariate
clusters and the matched scalar GP policy for singleton clusters. The
factorization result concerns the SCM-defined kernels; it does not assert
that a fitted GP recovers them, identifies off-support effects, or exactly
represents every within-cluster predictive distribution.

Our corrected MCBO comparison shares a scalar GP construction between fine
nodes and singleton clusters: a fixed-noise GP with the supplied additive-noise
variance, standardized scalar outputs, and marginal-likelihood fitting.
Multi-variable clusters use a Kronecker multi-task GP with a full-rank task-noise
likelihood. The optimistic mechanism simulator includes this joint residual
noise once. Fine and quotient implementations share the corrected intervention
map, input normalization, and acquisition random-number protocol. We separately
compare fine MCBO with its full action menu, fine MCBO restricted to the quotient
menu, and QMCBO on that same restricted menu.

For dynamic models, the coherent DCBO and QDCBO adapters fit each conditional
mechanism to the union of its contemporaneous and lagged external parents.
Singleton mechanisms share the same scalar implementation. Nonroot
multi-variable mechanisms combine a multi-output GP mean and latent covariance
with an explicitly estimated joint residual covariance, described in
Appendix~B. Dynamic population responses integrate the complete intervention
history: earlier selected manipulator actions remain fixed, whereas all free
past variables, including past outcomes, are re-simulated. This coherent
population protocol is a declared adaptation of the released DCBO software.

Under an acyclic Markovian SCM with mutually independent node disturbances
and a valid partition, composition of the structural cluster kernels is
sound for whole-cluster interventions. These kernels do not in general
determine interventions on strict subsets of a cluster: two fine SCMs can
agree on all cluster kernels and whole-cluster laws while disagreeing on a
sub-cluster effect. Propositions~3--5 distinguish structural construction,
whole-cluster soundness, and this limitation.
```

Retain the existing DCBO citation [2] at its first mention in the dynamic paragraph. The corrected scalar noise policy supersedes the earlier edit package's historical SingleTaskGP comparison. The structural kernel is SCM-defined; the fitted Gaussian approximation does not identify its off-support values.

## T07 — Adaptive refinement: replace the final paragraph of §3.3 (p.4)

**Before:** “In our experiments, refinement is triggered when the incumbent objective improves by less than 10−3 over five trials. The hierarchy and trigger are heuristic; Section 4.5 states the conditional recovery and invariance guarantees.”

**After:**

```latex
In the noisy controlled protocol, refinement is triggered when the best
measured response improves by less than $10^{-3}$ over five trials. A split
adds its new arms only after purchasing their complete initial designs,
with three measurements per new arm. The MediatedChain split adds two
singleton arms and therefore costs six units. If the full design is
unaffordable, the split is refused. Existing arms retain their data and
initial-quotient priors; new arms receive plain GP priors. Split times are
recorded separately for each seed. The hierarchy and trigger are heuristic;
Section~4.5 gives conditional recovery and invariance statements, not a
fixed split time or a finite-budget success guarantee.
```

**Reason/status:** protocol update; the old universal trial-7 interpretation and free refinement data cannot describe the corrected runs. Use publication label **HQCBO** for the implemented graph-free hierarchy (`HQCBOGF` internally); do not plot a second purported method named HQCBO-GF.

## T08 — Temporal invariance scope: local replacement in §4.1 (p.4)

**Before:** “QCBO and QMCBO satisfy this contract, as does QDCBO within each time slice.”

**After:**

```latex
QCBO and QMCBO satisfy this contract under the declared implementations.
For the dynamic construction, the contract must hold for the full temporal
input, including lagged parent relations, history handling and transitions;
equality of within-slice quotients alone is insufficient.
```

**Reason/status:** scope clarification. This does not alter Proposition 1; it makes explicit which dynamic inputs must match.

## T09 — MIS/nonempty value connection (Section 4.2, PDF p.5)

**Before:** “The MIS-reduction result in Appendix A connects this action-family value to the quotient exploration set.” Keep Proposition 2 and Eqs.(8)–(9) unchanged.

**After:**

```latex
Appendix~A shows that MIS reduction preserves the population optimum when
the null intervention is included. The action families defined in this section and the nonempty-arm
controlled protocol exclude that intervention. Equality between $V(\Pi)$
and the best retained nonempty quotient-MIS value therefore needs an
additional condition, such as a retained action with population value at
most $\mathbb E_M[Y]$. This conclusion requires a graph compatible with the
interventional distributions being evaluated; it does not apply to an
arbitrary misspecified supplied graph. Proposition~2 concerns $\mathcal U(\Pi)$
directly and does not require MIS reduction.
```

## T10 — Executed recommendations and Eqs.(10)–(11) (Section 4.3, PDF p.5)

**Before — first paragraph:** “Let $\hat z_t=(\hat X_t,\hat x_t)$ be the incumbent intervention after trial $t$, and define its population value by” followed by the display defining $\mu(\hat z_t)$.

**After** (replace that paragraph and display; keep the existing Eq.(10) display):

```latex
Let $\hat z_t=(\hat X_t,\hat x_t)$ be an eligible recommendation selected
among executed interventions using the learner's available information
and its declared recommendation rule. Here $t$ indexes recommendation
epochs after at least one eligible intervention has been executed; an
observation-only initialization does not create an executed incumbent.
Define its population value by
\[
\mu(\hat z_t):=\mathbb E_M[Y\mid
\operatorname{do}(\hat X_t=\hat x_t)].
\]
This population value is an evaluation quantity, not information that a
noisy-feedback learner may use unless a population-response oracle is
explicitly part of its protocol.
```

**Before — sentence immediately after Eq.(10):** “This metric sums the gaps of the incumbent recommendations.”

**After:**

```latex
This metric sums population gaps of the learner's recommendations. It
neither assumes nor implies that $\mu(\hat z_t)$ decreases with $t$.
In the noisy controlled protocol, the recommendation is chosen from
executed points by the smallest measured response, with ties broken by
canonical arm and execution index. A new noisy measurement can therefore
select a recommendation with worse population value. Independent scoring
evaluates that serialized recommendation without changing it. The best
population value among all visited points is a separate oracle diagnostic,
not the learner's recommendation.
```

**After Eq.(11), insert** before “A semi-Markovian SCM has mutually independent exogenous variables…”:

```latex
Equation~(11) is a pointwise algebraic decomposition for any eligible
cluster-union recommendation. Its second term is nonnegative because
$V(\Pi)$ is the minimum over the full nonempty cluster-union family; no
monotonicity, consistency, or MIS-optimality assumption is needed for this
inequality. A retained MIS family can have an additional restriction gap,
as made explicit in Appendix~A.
```

**Do not change** Theorem 1(i)/(ii) or their proof on p.12: both already quantify over eligible executed actions; the arm-loss floor and pathwise invariance hold under noisy observations as well. For the theorem, shared randomness means coupled optimizer and environment streams, not merely the same scalar seed. Its Part(ii) does not promise small optimization error or convergence.

**Why this preserves the theorem:** a noisy rule still chooses an executed action in the admissible family. Therefore its true population value cannot fall below that family's minimum, even when it occasionally increases. The proof never takes a running minimum of externally scored population values. The null counterexample changes an attempted identification of $V$ with retained-MIS value, not the definition of $V$ or this decomposition.

## T11 — Keep refinement recovery explicitly conditional (Section 4.5, PDF p.6)

**Before:** the final sentence “Recovery is conditional on refinement exposing a fine-optimal intervention and on subsequent incumbent consistency.” Replace that sentence only. Keep Definition 3, Corollary 2 and its proof (p.13), and Corollary 3 unchanged.

**After:**

```latex
Recovery is conditional on refinement exposing a fine-optimal intervention
and on subsequent incumbent consistency. Definition~3 is an additional
assumption on the complete optimizer and recommendation rule. We do not
establish it for the minimum-measured-response rule under noisy feedback,
for the finite-budget refinement heuristic, or for the model-based and
dynamic variants. Exposing an optimal arm does not by itself show that the
implemented recommendation reaches its optimum.
```

This prevents interpreting the valid conditional Cesaro argument as a new consistency theorem. Do not replace “if” with an unconditional recovery claim, or state that the noisy refinement method reaches the fine optimum from its arm exposure alone.

## T12 — Evaluation design: replace all of §5.1 (p.6)

**Before — complete section boundary:** from “We use ParallelParent, FrontDoor, and MediatedChain to isolate arm loss, prior corruption, and the price of coarsening, respectively.” through “CBO−np and QCBO−np keep their graph-derived arms and observation schedule but drop the causal prior.”

**After:**

```latex
\subsection{Evaluation design}
ParallelParent, FrontDoor, and MediatedChain provide constructed tests of
arm removal, prior replacement, and action restriction. Their closed-form
population objectives score recommendations selected using purchased noisy
measurements. The primary endpoint is population recommendation regret at
total intervention cost $B$, with $B=100$ for ParallelParent and FrontDoor
and $B=120$ for MediatedChain. The secondary endpoint is the sum of these
gaps at integer cost checkpoints $12,\ldots,B$. Initialization and
refinement purchases count toward the budget. These cost-checkpoint sums
are distinct from the trial-indexed quantity in Equation~(10).

Within each of 30 paired seeds, methods share 100 fresh observational rows,
initial values and measurements for common arms, and separately keyed
environment and algorithm random streams. Graph changes affect the supplied
model, not the true SCM. Recommendations minimize measured outcomes among
executed points; their population quality may increase or decrease. The
population scorer does not provide feedback to the learner. Protected-pair
audits compare complete actions, values, measurements, recommendations,
costs and refinement events, rather than only rounded final values.

BO uses the full joint intervention arm; BO-S uses every nonempty subset
with plain GP priors. CBO-NP retains CBO's graph-derived arms but uses plain
GP priors. CBO-FALLBACK retains the correct FrontDoor graph and its arm
menu while replacing both causal prior mean and variance by the same
observational fallback used under the confounded hypothesis. This control
distinguishes fallback replacement from simply deleting a prior. A
single-graph CEO comparator receives the same noisy purchases and budget;
its model-class limitations are stated in Appendix~C.

Separate MCBO and dynamic comparisons include fine/full-menu,
fine/coarse-menu and quotient/coarse-menu configurations, with 20 paired
seeds. The matched-menu contrast examines model construction; the full-menu
comparison also changes available actions. These suites use fixed rounds,
not equal total intervention costs. Dynamic endpoints are committed-policy
population outcomes under each method's intervention history, not regret
relative to a known global optimum. We report paired seed-level differences
with pointwise two-sided 95\% Student-$t$ intervals and disclose costs,
runtime, failures and Monte Carlo scoring uncertainty. Appendix~C specifies
the frozen protocols and complete configuration matrices.
```

**Reason/status:** required protocol replacement. This is valid methods prose before result completion; it asserts no empirical ordering. No new QCBO-NP campaign is claimed. Do not retain the old “eight benchmark tasks” sentence.

## T13 — ParallelParent: replace the empirical paragraph in §5.2 (p.7)

**Before — entire paragraph:** starts “Figure 2(a,b) shows the consequence.” Ends “Final values and regrets for every method appear in Table 3.” This includes “CBO stops at 4.25”, the trajectory-coincidence claim, and the A3 observed-deviation claim.

**After:**

```latex
The deletion restricts every eligible fine-CBO recommendation to population
value at least $4.25$, regardless of measurement noise or optimization
accuracy. This is an action-space lower bound, not a claim that a finite
run attains it. For $\Pi=\{\{X_1,X_2\},\{Y\}\}$ the deletion leaves the
quotient edge intact through $X_2\to Y$. Proposition~1 therefore predicts
identical coupled QCBO trajectories under the two supplied graphs when
the quotient contract and shared-input conditions hold. The matched noisy
comparison estimates finite-budget recommendation quality and audits this
pathwise prediction. Quotient-visible edits are outside the guarantee;
they are engineering diagnostics rather than additional 30-seed primary
comparisons in the frozen matrix.
```

**Also insert immediately after Eq.(12):**

```latex
The disturbances are independent standard Gaussians, and
$D(X_1)=D(X_2)=[-3,3]$.
```

**After audited results arrive:** add the generated estimate and interval from the PP A1 QCBO-minus-CBO row, reference `fig:controlled-misspecification` panel (a) and `tab:controlled-complete`, and state the actual complete-trace audit outcome. This is an unresolved evidence dependency, not a license to restore the old numbers.

## T14 — FrontDoor: replace the empirical paragraph and add domains in §5.3 (p.7)

**Before — entire paragraph:** from “Figure 2(c,d) shows a different failure.” through “Thus QCBO is invariant to the tested error but already lacks the informative prior available to correctly specified CBO.”

**After:**

```latex
The perturbation is internal to $\{X_1,M\}$ and preserves the quotient,
so Proposition~1 predicts coupled QCBO invariance. The quotient has both
$\{X_1,M\}\to Y$ and $\{X_1,M\}\leftrightarrow Y$; its cluster effect
is not identified, and the implementation uses observational mean and
variance fallback. Fine CBO's B0/B1 comparison holds its arm menu fixed
while replacing the identified prior. We additionally compare B1 against
CBO-FALLBACK on B0, which deliberately applies the same fallback while
retaining the correct graph. This control is required before attributing
an observed difference solely to prior replacement. Neither the direction
nor the size of that finite-budget effect follows from nonidentification.
```

**Insert immediately after Eq.(13):**

```latex
The disturbances are independent standard Gaussians, and
$D(X_1)=D(M)=[-3,3]$.
```

**After audited results arrive:** report the prespecified paired effects and the actual fallback audit; use Figure 2 **(b)**, never “(c,d)”. “Non-identifiable” describes the causal functional; it does not prove a harmful performance effect.

## T15 — MediatedChain: complete its SCM and replace the floor sentence in §5.4 (p.7)

**Before, displayed equations:** `$X_2=2X_1+0.2\varepsilon_2,\quad Y=(X_2-4)^2+0.1\varepsilon_Y$`.

**After:**

```latex
\[
X_1=0.5\varepsilon_1,\qquad
X_2=2X_1+0.2\varepsilon_2,\qquad
Y=(X_2-4)^2+0.1\varepsilon_Y.
\]
```

Retain the existing independent-Gaussian statement, domains, and analytical values $0.04$, $4$ and $3.96$.

**Before:** “Equation (11) gives fixed QCBO the permanent floor $R_T\geq3.96T$, consistent with Figure 3(b).”

**After:**

```latex
For every fixed-coarse recommendation, Equation~(11) gives a population
gap of at least $3.96$ and hence $R_T\geq3.96T$ on the trial index of
Equation~(10). On an integer cost grid beginning at 12, the corresponding
lower bound for a sum through cost $b$ is $3.96(b-11)$, for $b\geq12$.
HQCBO can expose the useful singleton arm through a paid split. The
noisy experiment measures whether and when refinement improves the
recommendation within budget; arm exposure alone does not imply attainment
of the fine optimum.
```

**Reason/status:** missing root equation plus metric alignment. The cost-grid bound counts one eligible recommendation at each integer checkpoint, with no invented values before initialization.

## T16 — Replace §5.5 and its heading (pp.7–8)

**Before heading:** “Portability across optimizer families”. **Before body:** the two paragraphs from “Figure 4 compares each quotient construction with its base optimizer on the eight family tasks” through “they do not directly estimate the nonnegative coarsening price $\Delta(\Pi)$.”

**After:**

```latex
\subsection{Separating action menus and model construction}
The corrected MCBO comparison uses ToyGraph and PSAGraph with three
configurations: fine/full-menu, fine/coarse-menu and quotient/coarse-menu.
We distinguish expected reward of the measured-data recommendation from
oracle-best-visited reward. Protected graph edits are checked using complete
paired traces, not only final equality. The dynamic comparison uses the
same three-way design on stationary, independent and explicitly regularized
nonstationary SCMs. It scores each committed slice intervention under the
method's full preceding intervention history and reports both per-slice
outcomes and their sum.

The quotient/coarse versus fine/coarse contrast holds the action menu
fixed. Full-menu contrasts additionally include action restrictions.
Finite-budget performance differences therefore do not directly estimate
the nonnegative population price $\Delta(\Pi)$. The original Toy,
Synthetic and Coral CBO-family runs are historical experiments outside
these corrected comparisons and are excluded from the primary results.
```

**After audited results arrive:** insert separate MCBO and dynamic effect paragraphs linked to their new figures/tables. Report negative, null or positive paired effects as observed. Do not use the original Table 1 cells or Figure 4 panels to fill these paragraphs.

## T17 — Related work: three local edits (§4.4 p.6; §6 p.8)

**Insert in §4.4 immediately after “These conditions are sufficient but not necessary.”**

```latex
Original CBO also discusses natural mediator responses outside their
intervention domains \cite{aglietti2020cbo}. Our action-family value
$\Delta(\Pi)$ formalizes the restriction imposed by a chosen partition,
while the completion result supplies sufficient conditions for that
restriction to be lossless.
```

**Before, §6 final span:** from “A setting related to ours is causally abstracted MABs (CAMABs) [9, 29]” through the final “unknown.”

**After:**

```latex
CAMAB methods relate decision problems through causal abstractions
\cite{zennaro2024camab,dyer2025atucb}. In particular, AT-UCB uses an
abstract model to eliminate candidate actions before learning in the base
model and bounds the relation between optimal rewards across abstraction
levels \cite{dyer2025atucb}. Our setting fixes a supplied quotient and
restricts interventions to whole clusters. Our contribution is optimizer
invariance to quotient-preserving graph changes and the associated
action-restriction and completion analysis. Abstraction-based action
elimination is not claimed as new.
```

**Insert after the first paragraph of §6, ending “operate over, or trigger refinement of, a quotient.”**

```latex
MCBO already accommodates vector-valued mechanisms
\cite{sussex2023mcbo}. QMCBO replaces fine mechanisms by mechanisms on a
supplied quotient and enforces whole-cluster intervention targets;
vector-output modeling itself is not our novelty.
```

**Evidence:** [original CBO §4.2](https://proceedings.mlr.press/v108/aglietti20a/aglietti20a.pdf), [MCBO §2 and Appendix A.1](https://arxiv.org/html/2211.10257), [CAMAB](https://proceedings.mlr.press/v244/zennaro24a.html), [AT-UCB §3](https://arxiv.org/html/2509.04296v1). Use the supplied canonical BibTeX keys or remap them consistently.

## T18 — Discussion and limitations: replace the complete §7 body (p.8)

**Before:** from “Exact invariance holds between supplied graphs with the same quotient.” through “Finally, QMCBO inherits the causal-sufficiency assumption of its model-based factorization.”

**After:**

```latex
Exact invariance holds between supplied graphs with the same quotient
when the optimizer satisfies the quotient contract and all other inputs
and random streams are coupled. Protection relative to a true-graph run
requires that shared quotient to match the true quotient. Invariance does
not imply an informative prior, low regret or recovery of the fine optimum.
The sufficient zero-price conditions concern the true SCM and intervention
domains; exogenous separation cannot generally be verified from observational
data alone. Connecting the nonempty action-family optimum to retained MIS
actions also needs the null-intervention qualification in Appendix~A.

Refinement is heuristic. Recovery requires exposure of a fine-optimal
intervention and subsequent incumbent consistency; we do not prove that
consistency for the minimum-measured-response rule under noisy feedback.
Partitions change arm counts, initialization costs and intervention costs.
The static experiments match total intervention cost, whereas the MCBO
and dynamic experiments match rounds and report their differing costs.
The static homoscedastic GP likelihood is an approximation to potentially
heteroskedastic intervention responses. The single-DAG CEO comparator
does not exercise graph-uncertainty learning and cannot represent the
latent confounding of ParallelParent and FrontDoor.

The model-based structural factorization assumes causal sufficiency and
SCM-defined kernels; finite-data GP estimates do not identify off-support
effects automatically. Dynamic joint residual covariance includes model
and estimation error, and finite-particle integration remains approximate.
The dynamic population protocol re-simulates free past variables and is
distinct from both path-conditioned control and the single-row noisy static
protocol. Its regularized nonstationary SCM is a new companion, not a
renaming of results from the original reciprocal-normal benchmark.

The controlled examples were constructed to display specific mechanisms,
and their settings were selected after exploratory work. Independent
replication seeds do not establish average benefit across a representative
population of real problems. The PSA study is a synthetic simulator
benchmark and provides no clinical validation or treatment recommendation.
Historical Coral runs are excluded from the corrected primary evidence;
their data and preprocessing lineage would need verification before use
as new empirical evidence. Reported intervals quantify between-seed
variation under these fixed designs, not uncertainty over graph elicitation,
benchmark selection or untested deployment settings.
```

**Reason/status:** scope disclosure, not a claim that every possible concern is a demonstrated empirical failure. Do not invent patient recruitment, ethical approval or permissions.

## T19 — Conclusion: replace the complete paragraph (Section 8, PDF p.8)

**Before:** “Quotient causal optimizers are invariant to graph errors that leave the supplied C-DAG unchanged. Restricting interventions to unions of clusters can incur a price, which cannot increase under refinement and vanishes under the stated structural conditions. Hierarchical refinement offers a conditional route to recovering the fine optimum.”

**After:**

```latex
Quotient causal optimizers satisfying the quotient contract have identical
coupled trajectories under graph changes that preserve their supplied
cluster graph. This robustness concerns the optimizer's inputs; it does
not guarantee accurate causal estimates or good finite-budget decisions.
Restricting the nonempty action family to unions of clusters can incur a
population price that cannot increase under refinement and vanishes under
the stated structural conditions. Connecting that action-family value to
retained MIS actions requires the null-intervention qualification proved
in Appendix~A. Refinement provides a conditional route to recovery, not a
consistency guarantee for the implemented noisy recommendation rule.
```

## T20 — Full singleton proposition and proof (Appendix A.1, PDF p.11)

**Before:** replace the complete “Proposition 3 (Identity recovery)” statement and its proof, ending “also reduce to their node-level counterparts.” Preserve the following Proposition 4's location/counter and the existing Proposition 3 label.

**After:**

```latex
\begin{proposition}[Singleton construction identity]
At $\Pi=\Pi_{\mathrm{fine}}$, quotienting preserves the graph and the QCBO
exploration set is $\operatorname{MIS}(\widehat G)$. With the same
identification and estimation routines, its prior construction agrees
arm by arm with matched nonempty-MIS CBO. The dynamic quotient graphs and
the structural cluster factorization likewise reduce to their fine
counterparts. Equality of instantiated optimizers additionally requires
matching estimators, GP parameterization, acquisition, initialization,
observation policy, numerical settings and random-number streams.

\end{proposition}

\begin{proof}
At singleton resolution $\chi$ is one-to-one, no graph edge is changed,
and unions of manipulable singleton clusters are exactly the subsets of
$\mathcal X$. Equation~(6) therefore becomes Equation~(2).
Identification queries coincide, and matched numerical estimators give
matched priors. The same argument applies to dynamic slice and transition
graphs, while a singleton structural kernel is just its node kernel.
These facts prove construction identity. They imply equality of
optimizer trajectories only when the additional implementation inputs
and routines match, in which case induction over updates applies.
\end{proof}
```

**Add immediately after the proof:**

```latex
The historical QMCBO implementation and its upstream comparator did not
match their singleton noise policies, so those archived implementations
do not satisfy an automatic numerical identity claim. The corrected
comparisons explicitly match the scalar fitting policy and test posterior,
gradient and trajectory agreement under their declared inputs. Such tests
verify the implemented configurations; the graph construction alone does
not imply them.
```

This supersedes the old patch paragraph beginning “Our current QMCBO implementation learns the observation-noise parameter…”: that description applied to the archived implementation, not the corrected matched scalar policy. Do not claim unmodified upstream MCBO and QMCBO are numerically identical.

## T21 — Full structural-kernel proposition and proof (Appendix A.1, PDF p.11)

**Before:** replace “Proposition 4 (Soundness of joint cluster mechanisms)” and its proof, ending “Validity orders the resulting mechanisms along the C-DAG.” Keep Proposition 5 and its proof in place.

**After:**

```latex
\begin{proposition}[Soundness of structural cluster kernels]
Assume an acyclic causally sufficient SCM with mutually independent
node disturbances and a valid partition $\Pi$. For
$\varnothing\ne S\subseteq\Pi_{\mathcal X}$ and $X=\bigcup_{C\in S}C$,
for every $x\in D(X)$,
\[
 P_M(dv_{\mathcal V\setminus X}\mid\operatorname{do}(X=x))
 =\left.\prod_{C\in\Pi\setminus S}
 K_C(dv_C\mid v_{\operatorname{pa}(C)})\right|_{X=x}.
\]
The product is a composition of probability kernels in quotient
topological order; intervened coordinates are fixed constants.

\end{proposition}

\begin{proof}
Each node equation and its independent disturbance define a structural
kernel at every parent assignment. For each cluster, compose its member
kernels in their internal topological order, holding parent-cluster values
fixed, to obtain $K_C$. A whole-cluster intervention deletes every member
kernel of each intervened cluster and fixes its coordinates. Grouping the
remaining kernels gives the displayed law. Quotient validity permits an
acyclic cluster order. Since cluster disturbances are independent of their
parent clusters, $K_C$ is a version of the observational conditional.
The structural construction, rather than a choice of observational
conditional on null events, defines its values elsewhere.
\end{proof}
```

The observable conditional equals a structural kernel only almost surely in parent values under the stated independent-disturbance assumptions. “Every point in the support” would still overclaim identification at zero-probability points. Whole-cluster soundness is not a confounded-DAG factorization theorem, not GP consistency, and not identification of partial-cluster interventions. Proposition 5's reverse linear-Gaussian witness remains valid with this SCM-defined kernel interpretation.

## T22 — Full MIS lemma, proof, quotient consequence and counterexample (Appendix A.3, PDF pp.11–12)

**Before — exact replacement span:** from “Lemma 1 (MIS reduction). Let $G$ be an ADMG with manipulable variables $\mathcal X$ and target $Y$.” through the end of its proof and the immediately following sentence “Consequently, when the null intervention is handled consistently, the best value over quotient MIS expanded to fine variables equals the best value over the corresponding cluster-union action family.” Stop **before** “For $X_1\to X_2\to Y$, intervening on both variables gives $X'=\{X_2\}$; fixing $X_2$ already blocks the effect of $X_1$.” Retain that chain example and the following Proposition 2 and Theorem 1 proofs.

**After:**

```latex
\begin{lemma}[MIS reduction, with the null intervention]
Let $G$ be the latent-projection ADMG of the acyclic SCM $M$ whose
interventional distributions are evaluated, with manipulable variables
$\mathcal X$ and target $Y$. Assume the product-domain convention of
Section~2.1 and assume additionally that $\mathbb E_M[|Y|]<\infty$
under the null intervention. For $X\subseteq\mathcal X$, put
\[
 X'=X\cap\operatorname{An}_{G_{\overline X}}(Y).
\]
Then (i) for every $x\in D(X)$,
\[
 P_M(Y\mid\operatorname{do}(X=x))
 =P_M(Y\mid\operatorname{do}(X'=x|_{X'}));
\]
(ii) if $X'\ne\varnothing$, then $X'\in\operatorname{MIS}(G)$;
and (iii), defining $D(\varnothing)=\{\varnothing\}$ and
$f_{\varnothing}(\varnothing)=\mu_{\varnothing}:=\mathbb E_M[Y]$,
\[
 \min_{X\in\{\varnothing\}\cup\operatorname{MIS}(G)}\mu^\star(X)
 =\min_{X\subseteq\mathcal X}\mu^\star(X).
\]
Here MIS retains the manuscript's nonempty-set convention, and the
attainment assumptions of Section~4.2 apply.

\end{lemma}

\begin{proof}
Write $R=X\setminus X'$. If some $r\in R$ had a directed path to $Y$
in $G_{\overline {X'}}$, that path could not enter $X'$. Let $r^\star$
be the last member of $R$ on the path. The remaining path from $r^\star$
to $Y$ enters no member of $X$, so it also exists in $G_{\overline X}$,
contradicting the definition of $R$. Hence restoring the equations for
$R$ while holding $X'$ fixed cannot affect $Y$. Coupling the two
interventions with the same exogenous disturbances proves (i), including
when $X'$ is empty. Domain projection makes $x|_{X'}$ admissible.
For each $v\in X'$, its directed path to $Y$ in $G_{\overline X}$ also
exists in $G_{\overline {X'}}$; thus (ii) follows from the MIS criterion.
Every candidate value on the right side of (iii) is therefore represented
on the left, and the reverse inequality follows by action-family inclusion.

\end{proof}

For a valid quotient of this true graph, let
$\mathcal E_\Pi=\operatorname{ES}_\Pi(G_\Pi)$,
$W(\Pi)=\min_{X\in\mathcal E_\Pi}\mu^\star(X)$, and set
$W(\Pi)=+\infty$ if $\mathcal E_\Pi$ is empty. All clusters eligible for
intervention are fully manipulable. Apply the same last-removed-ancestor
argument at cluster level: after the retained clusters are fixed, a
fine directed path from any removed cluster to $Y$ would project to a
directed quotient path, a contradiction. Thus removing those whole-cluster
interventions preserves the response, even with latent confounding, and
\[
 \min\{\mu_{\varnothing},W(\Pi)\}
 =\min\{\mu_{\varnothing},V(\Pi)\}.
\]
Since $\mathcal E_\Pi\subseteq\mathcal U(\Pi)$, we also have
$V(\Pi)\le W(\Pi)$. If $W(\Pi)\le\mu_{\varnothing}$, the displayed
equality forces $V(\Pi)=W(\Pi)$: a strictly smaller $V(\Pi)$ would make
the right side strictly smaller than the left. This sufficient condition
is not assumed for arbitrary benchmarks or wrong supplied graphs.

For a counterexample without this condition, take independent standard
normal $U,U_Z$ and set $A=0.1U$, $Z=U_Z$, $Y=A^2$, with manipulable
variables $\{A,Z\}$, $D_A=[1,2]$ and $D_Z=[-1,1]$.
The graph has only the edge $A\to Y$. Its only nonempty MIS is $\{A\}$,
whose optimal value is $1$, whereas intervening on $Z$ alone gives
$\mathbb E[A^2]=0.01$. Hence at singleton resolution
$V=0.01<W=1$. Reducing $\{Z\}$ gives the null intervention and
$\min\{0.01,1\}=\min\{0.01,0.01\}$, as asserted by the corrected lemma.
In the nonempty-action protocol, observation-only rounds do not make the
null intervention eligible as a recommendation. Any experiment that permits
a null recommendation must declare that different action-family convention
separately.
```

The paragraph after `\end{proof}` is the quotient consequence and counterexample, not a new numbered theorem. Empty retained families use $W=+\infty$ explicitly. The result applies to the graph compatible with the SCM being evaluated; applying it to a wrong supplied graph would reintroduce the original error.

## T23 — Appendix B.1 — replace the complete subsection (pp.13–14)

**Operation:** replace the existing B.1 heading and all text through the sentence “where beta_t > 0 controls exploration”, stopping before B.2. The replacement includes its own subsection heading.

**Before — original anchors:** PDF p.13 “After global trial t”; PDF p.14 “Model-based experiments retain their base method's lower confidence bound”. The single generic $\sqrt{\beta_t}$ formula does not describe the MCBO implementation's optimistic mechanism parameterization.

```latex
\subsection{Acquisition and recommendations}
For an arm $X$, let $\mu_{X,t}(x)$ and $\sigma_{X,t}(x)$ denote the
posterior mean and standard deviation conditional on the learner's purchased
data. The static arm-based methods maximize their plug-in expected improvement
per intervention cost,
\[
 \operatorname{EI}_{X,t}(x)/c(X),\qquad
 \operatorname{EI}_{X,t}(x)=
 \mathbb E[(y_t^{\rm best}-f_X(x))_+\mid D_t],
 \qquad c(X)=|X|.
\]
Here $y_t^{\rm best}$ is the smallest measured outcome. This is the declared
plug-in noisy-data acquisition, not a claim to implement a noise-integrated
expected-improvement criterion. Unaffordable arms are masked before a purchase;
initialization and refinement measurements also consume the budget.

MCBO uses its released optimistic mechanism acquisition, with the
implementation's multiplier $\beta=10$ in the mechanism-level expression
$\mu+\beta\sigma\eta$. We do not reinterpret this setting as a
$\sqrt{10}$ multiplier or tune it using replication outcomes. Residual base
draws are fixed during each acquisition optimization and broadcast across
candidate batches, so repeated or permuted candidate evaluations use the same
Monte Carlo objective. The executed optimistic-network acquisition uses \texttt{qSimpleRegret}
with $q=1$ and 128 Sobol samples. Its mechanism confidence parameters use
a coordinatewise box, not a calibrated joint confidence ellipsoid.
Algorithm, environment-measurement, and scoring streams
are separately keyed.

Performance is evaluated at a recommendation selected using measured data.
For static minimization this is the purchased point with the smallest measured
outcome, with ties resolved by canonical arm order and then event index.
For MCBO maximization it is the eligible purchased point with the largest
measured outcome, with the first event resolving ties. Population evaluations
score this selected point and cannot change the recommendation. The best
population value among visited points is logged only as an oracle diagnostic.
Consequently, the scored recommendation curve need not be monotone.
```

Do not apply the static recommendation rule or MCBO acquisition formula to the dynamic adapter by implication. Its selected slice intervention is scored under the full prefix specified below.

## T24 — Appendix B.2 — replace the complete subsection (p.14)

**Original anchor, PDF p.14:** “The complete identification check is performed on the fine ADMG for CBO or the supplied C-DAG for QCBO.” Replace the B.2 heading and entire subsection, stopping before Appendix C. The replacement below already includes the unchanged backdoor integral; do not duplicate it. It distinguishes identification, estimator support, and GP approximation.

```latex
\subsection{Causal-effect and mechanism estimation}
Identification is evaluated using the graph supplied to the method: a fine
ADMG for CBO or the supplied C-DAG for QCBO. Identification of a functional
and availability of its numerical implementation are separate checks.
Supported backdoor, frontdoor, and g-computation functionals are instantiated
with observational plug-in estimators; an unsupported or nonidentified query
uses the declared observational fallback. For example, a supported backdoor
adjustment set $Z$ gives
\[
 \widehat{\mathbb E}[Y\mid\operatorname{do}(X=x)]
 =\int \widehat{\mathbb E}[Y\mid X=x,Z=z]\,d\widehat P(z).
\]
In the matched static experiments, purchased outcomes are noisy SCM samples.
The arm GP's Gaussian likelihood variance is learned on the raw target scale,
initialized at $\max(0.01,0.1\operatorname{Var}(Y))$, and constrained to
$[10^{-6},10^6]$. Rebuilt models are optimized before acquisition evaluation;
freeing a noise parameter without subsequently fitting it is insufficient.
The exact population functional is available only to the separate evaluator.

Dynamic multivariate conditionals use an intrinsic-coregionalization GP with
joint predictive task covariance. For each original observational row, all
its task outputs are held out together when forming the residual approximation.
Writing $\alpha=K^{-1}y$ and $I$ for those output indices, the held-row residual
is $([K^{-1}]_{II})^{-1}\alpha_I$. The centered sample covariance $S$ of these
residual vectors is regularized as
\[
 \Sigma_{\rm res}=0.9S+0.1\operatorname{diag}(S)
 +10^{-8}\max\{\operatorname{tr}(S)/m,1\}I_m.
\]
Prediction adds $\Sigma_{\rm res}$ to the latent GP task covariance, with the
stock likelihood excluded to avoid adding observation noise twice. This
residual estimate also reflects finite-data mean-estimation error and model
misspecification; it is not asserted to equal the true structural-noise
covariance. Unconditional multivariate roots use their joint empirical law.
The integration retains task covariance at each sampled input, but does not
sample a globally correlated posterior function across all particles.
```

## T25 — Insert new Appendix B.3 before Appendix C

**Location:** PDF p.14, immediately before “C Experimental details”. There is no B.3 in the supplied PDF.

```latex
\subsection{Frozen numerical settings}
\begin{center}
\begin{tabular}{p{0.23\linewidth}p{0.69\linewidth}}
\hline
Protocol & Settings \\
\hline
Matched static & 100 observational rows; 3 initial points per arm;
30 paired seeds (2000--2029); total variable-intervention cost
$B=100$ for ParallelParent/FrontDoor and $B=120$ for MediatedChain. \\
Static likelihood & Learned Gaussian variance; initialization
$\max(0.01,0.1\operatorname{Var}(Y))$; bounds $[10^{-6},10^6]$. \\
CEO comparator & One committed observable DAG; 35 acquisition anchors;
the same noisy purchases and cost budget; no extra observational rows. \\
MCBO comparison & 20 paired seeds (2000--2019); 100 sequential rounds;
5 initial observational rows and 2 initial rows per nonnull allowed target;
optimism multiplier $\beta=10$. \\
MCBO scoring & Toy: exact population response at zero additive noise.
PSA: independent scoring stream with 100,000 samples per action;
intrinsic simulator randomness remains present. \\
Dynamic comparison & 20 paired seeds (2000--2019); 3 slices;
10 trials per slice; 100 observational trajectories; 100 acquisition anchors. \\
Dynamic integration & 1,024 predictive particles; 2,048 feedback particles;
100,000 independent scoring particles; 10\% residual diagonal shrinkage. \\
\hline
\end{tabular}
\end{center}
Engineering and precision pilots use separate seeds and do not enter the
replication estimates. Source files, upstream modules, settings, and runtime
versions are frozen in manifests. Strict analyses require the full declared
matrix and reject failed, missing, or inconsistent units. Reported paired
intervals are pointwise two-sided Student-$t$ intervals across seeds; they are
not simultaneous intervals or automatic superiority decisions.
```

## T26 — Appendix C — replace “Controlled suite.” paragraph (PDF p.14)

**Obsolete span:** especially “T=50 trials, or 60” and “Initial and sequential outcomes are evaluated by the same closed-form population expectation”.

```latex
\paragraph{Matched static suite.}
ParallelParent, FrontDoor, and MediatedChain use a shared observational sample
of 100 rows per seed and three uniformly sampled initial points per allowed
arm. Initialization is keyed by arm, so overlapping arms share their initial
values and measurements across methods. Each purchase returns one noisy SCM
row; the learner never receives its population expectation. Exogenous Gaussian
disturbances, including shared disturbances inducing the specified bidirected
edges, follow the declared SCM. The total budget counts one unit per intervened
variable, including initial and refinement data. It is 100 for ParallelParent
and FrontDoor and 120 for MediatedChain. Runs stop when no permitted purchase
is affordable. HQCBO purchases the complete three-point initial design for each
new arm before that arm enters sequential selection; such designs are atomic
budgeted purchases.

We evaluate the measured-data recommendation using the closed-form population
objective after purchasing data. The primary endpoint is recommendation regret
at the declared budget. The cost-indexed summary sums this regret at integer
costs 12 through the budget, carrying forward the latest affordable recorded
recommendation. These are not oracle best-visited regrets. The frozen matrix
contains 24 method--SCM--condition configurations and 30 paired seeds,
2000--2029. The primary paired contrast is QCBO minus CBO within the specified
SCM and condition. Full trajectories, costs, initial designs, and recommendation
indices are retained for audit.
```

## T27 — Appendix C — CEO paragraph (insert after matched static protocol)

```latex
\paragraph{Committed-graph CEO comparator.}
The pinned CEO implementation is run with a single observable candidate DAG
and posterior mass one. This is a committed-graph comparator, not evaluation
of CEO's graph-uncertainty learning. The observable DAG does not represent
the latent confounding in ParallelParent or FrontDoor; MediatedChain's true
graph is representable. CEO receives the same 100 observational rows, allowed
fine intervention arms, arm-keyed initial measurements, and total cost budget.
It uses its learned Gaussian likelihood and 35 acquisition anchors. The adapter
masks unaffordable arms and performs no additional observational acquisition.
Its second callback, called ``noiseless'' upstream, replays the already purchased
noisy row; it neither queries an oracle nor purchases another measurement.
The measured row is used for fitting and internal bookkeeping, and population
scoring is performed separately after the run. These modifications and the
single-graph restriction are part of the declared comparator.
```

## T28 — Appendix C — replace the complete old “Family suite.” paragraph, part 1 (p.14)

**Replacement boundary:** delete the entire old “Family suite.” paragraph, from “The CBO family uses ToyGraph” through “Table 5 reports paired invariance counts for the model-based family.” Paste this MCBO paragraph and the dynamic paragraph below in its place. Do not retain the old CBO/Coral results protocol.

**Old anchors:** “The MCBO family uses ToyGraph and PSAGraph … zero observation noise”; “DCBO/QDCBO and MCBO/QMCBO retain the released implementations”. Delete both unqualified claims.

```latex
\paragraph{Corrected MCBO comparison.}
ToyGraph and PSAGraph compare fine MCBO with its full intervention menu,
fine MCBO restricted to the quotient menu, and QMCBO with that restricted menu.
The main matrix has 120 runs: two SCMs, three method/menu configurations,
and 20 paired seeds, with 100 sequential rounds per run. All methods share
the corrected physical intervention map and guarded parent normalization;
a naturally generated mediator is not clipped to its intervention domain.
ToyGraph uses zero additive disturbance. PSAGraph remains intrinsically
stochastic even though its interface receives \texttt{noise\_scale=0}.
ToyGraph's observational action is eligible when present in its menu;
PSAGraph's initial observational rows train mechanisms but are not eligible
intervention recommendations.

The primary comparison holds the action menu fixed and contrasts QMCBO with
restricted fine MCBO. Full-menu MCBO is a separate comparison. We report the
final recommendation's expected reward and the sum of recommendation expected
rewards over equal sequential rounds. Initialization counts and intervention
costs are recorded; equal rounds do not imply equal total cost across menus.
An additional frozen 80-run matrix applies the protected within-cluster edits
(ToyGraph: delete $X\to Z$; PSAGraph: add aspirin$\to$statin) and compares each
run with its unedited counterpart. Quotient invariance is checked on complete
executed action, measurement, scoring, cost, and recommendation traces, rather
than equality of the final score alone. No invariance count or performance
estimate is inferred from engineering pilots.
```

## T29 — Appendix C — replace the complete old “Family suite.” paragraph, part 2 (p.14)

**Insert immediately after the preceding corrected MCBO paragraph, as the second part of the same full-paragraph replacement.**

```latex
\paragraph{Coherent dynamic population protocol.}
The stationary and independent SCMs are accompanied by an explicitly named
regularized nonstationary SCM. In the latter, there is one change point at $t=1$, where
$Z_1=-X_1/\sqrt{1+X_0^2}+Z_0+\varepsilon_{Z,1}$ replaces the
reciprocal-denominator mechanism. All other mechanisms are unchanged.
The equation-parent graph includes $X_0\to Z_1$ at the change point and
$Z_1\to Y_2$ in the subsequent regime; its quotient is constructed
from this complete temporal graph. This is a new integrable companion, not the original nonstationary
benchmark. The original reciprocal-normal population objective is excluded
from primary expected-value comparisons.

At slice $t$, a candidate is evaluated as
$\mathbb E[Y_t\mid\operatorname{do}(a_0,\ldots,a_{t-1},a_t)]$.
Only earlier selected manipulator interventions are fixed. Past outcomes and
other free variables are re-simulated, not clamped to past estimated means.
The learner receives a fixed-common-noise 2,048-particle response estimate;
a separate post-run stream uses 100,000 particles for population scoring.
This is a controlled finite-Monte-Carlo response-oracle experiment, distinct
from the single-row noisy static protocol. The coherent fine/full-menu,
fine/restricted-menu, and quotient/restricted-menu configurations share the
numerical settings in Appendix~B.3. The 1,024-particle predictive integration
was selected through a separate fixed-grid sensitivity check against 4,096
particles; this check does not establish exact integration. Comparisons use
fixed rounds and make no equal physical-cost or global-optimum regret claim.
```

## T30 — Appendix C: add exact domain, configuration and reporting details (p.14)

**Before:** missing. Insert after the new matched-static paragraph.

**After:**

```latex
\paragraph{Domains and configuration inventory.}
ParallelParent uses $D(X_1)=D(X_2)=[-3,3]$;
FrontDoor uses $D(X_1)=D(M)=[-3,3]$;
MediatedChain uses $D(X_1)=[-3,3]$ and $D(X_2)=[-2,2]$.
Natural mediator responses are not clipped to intervention domains.
The 24 static configurations comprise CBO and QCBO on PP A0/A1,
FD B0/B1 and MC C0; CEO on PP A0, FD B0 and MC C0; HQCBO on MC C0;
CBO-NP, BO-S and BO on PP A0; CBO-NP on PP A1;
CBO-NP, BO-S, BO and CBO-FALLBACK on FD B0;
and BO-S and BO on MC C0. Each configuration has 30 seeds.
A2 and A3 are separately labelled engineering diagnostics.
The main MCBO matrix has 120 runs, with 80 additional edited runs for
protected-pair checks. The dynamic matrix has 180 runs: three SCMs,
three configurations and 20 seeds. In the dynamic scheduler, trial zero
is observational bookkeeping; nine purchased interventions per slice
give 27 purchased interventions across the three slices.

\paragraph{Inference and reproducibility.}
The replication unit is the seed. For each prespecified paired contrast,
we report the mean paired difference and its two-sided 95\% Student-$t$
interval, with 29 degrees of freedom for static comparisons and 19 for
MCBO and dynamic comparisons. Intervals are pointwise, not simultaneous.
The primary static contrast is QCBO minus CBO within each declared
condition; the primary model-based and dynamic contrast is quotient/coarse
minus fine/coarse. Other method summaries are descriptive unless a
contrast is explicitly prespecified. Monte Carlo scoring standard errors
are reported separately from between-seed variation. Dynamic slices share
scoring randomness, so independence of their scoring errors is not assumed.
Complete configuration matrices, seed-level records, event ledgers, source
and dependency manifests, and figure-generation commands are required for
each reported result. Missing or failed units block the affected primary
analysis; they are not silently omitted or replaced by successful seeds.
```

**Availability statement — release-dependent, do not paste unfilled fields:**

```latex
\paragraph{Code availability.}
The reproducibility materials are available at \url{RELEASE_URL}.
The immutable release identifier is \texttt{IMMUTABLE\_\allowbreak RELEASE\_\allowbreak ID}.
The release contains code, frozen configurations, environment manifests,
complete run records and figure-generation commands.
Historical exact-response results and corrected replication results are
stored separately and identified by their source and reporting versions.
```

`RELEASE_URL` and `IMMUTABLE_RELEASE_ID` remain required release information. Local ZIP files and a dirty Git HEAD do not establish a public immutable release. No availability claim is ready until those fields are real.

## Figure swaps: all 13 original figures

Numbers below refer to the supplied PDF. Let LaTeX renumber the revised paper. The filenames in **READY** rows are included in the companion ZIP under `figures/`. Filenames in **PENDING** rows are exact proposed deliverable names for the result-generation agent; those files do **not** yet exist. Replacing a caption does not validate an old plot.

| Original figure | Action | Replace with | State |
|---|---|---|---|
| 1, p.1: ParallelParent motivation | Retain the existing three-panel diagram and caption. | Existing asset, unchanged. | RETAIN |
| 2, p.7: controlled misspecification | Replace the entire two-panel historical result PDF, `minimal_misspec.pdf`. | `controlled-noisy-misspecification.pdf`: (a) PP A0/A1, (b) FD B0/B1, data-selected recommendation population values versus total paid cost. CBO and QCBO are the primary curves; distinguish conditions. Use all 30 seeds. | PENDING static720 audit |
| 3, p.8: refinement | Replace `minimal_refine.pdf` completely, including the old legend and trial-7 marker. | `mediatedchain-noisy-refinement.pdf`: (a) CBO/QCBO/HQCBO population recommendation values versus cost; (b) cost-grid cumulative gaps. Report observed split costs separately, not one fixed split time. | PENDING static720 audit |
| 4, p.9: eight-task family suite | Remove the entire old `family_suite.pdf`. Replace with **two separately numbered figures**. Remove historical panels (a–c). | `dynamic-population-comparison.pdf`: stat/ind/nonstat_regularized outcomes under committed prefixes, all three menu/model configurations. `mcbo-menu-comparison.pdf`: ToyGraph/PSAGraph expected recommendation reward, all three configurations. | PENDING dynamic180 / MCBO120 audits |
| 5, p.16: cost-indexed historical curves | Replace `cost_indexed.pdf`; do not repeat the old achieved-budget truncation. | `controlled-noisy-attribution.pdf`: (a) PP A0, (b) FD B0/B1 with CBO-FALLBACK, (c) MC C0, including the prespecified available controls. Paid-cost axis; gap of measured-data recommendations. | PENDING static720 audit |
| 6, p.18: PP graph | Keep topology; replace full float with provided caption-corrected TeX, or edit caption only. | `fig06-parallelparent.tex`. “Fine DAG” becomes “Fine ADMG”. | READY |
| 7, p.18: FD/Toy graph | Retain topology. Use FD for the new static suite; clarify that “ToyGraph” here is the historical arm-based example, not the distinct MCBO ToyGraph. | Existing figure; caption clarification below. | RETAIN + CAPTION |
| 8, p.18: MC / MCBO Toy graph | Retain topology and action-domain explanation. | Existing asset, unchanged; do not imply identical equations merely from common topology. | RETAIN |
| 9, p.19: CompleteGraph | Remove from the revised primary benchmark gallery because the Synthetic legacy family track is excluded. Archive unchanged outside the primary results. | No replacement. If retained solely as an explicitly historical illustration, “Fine DAG” must become “Fine ADMG”. | REMOVE FROM PRIMARY |
| 10, p.19: Coral | Remove from the revised primary benchmark gallery; historical Coral is outside the corrected comparison. Preserve its original archive. | No replacement. Reintroduction as empirical evidence requires a separate protocol and verified data/preprocessing lineage. | REMOVE FROM PRIMARY |
| 11, p.19: “DCBO stat / nonstat” graph | Replace its title/caption with stationary-only scope; keep stationary drawing. Insert the **new companion figure immediately after it**. | `fig11-stationary.tex`, then `dynamic-regularized-figure.tex`, which includes `dynamic-regularized-graph.pdf`. | READY |
| 12, p.20: independent dynamic graph | Retain. | Existing asset, unchanged. | RETAIN |
| 13, p.20: PSA | Replace both TikZ panels and caption. | `fig13-psa.tex`: orange only Aspirin and Statin and their cluster; Age, bmi and Cancer remain white; target blue. | READY |

The choice to exclude the old CBO-family panels and their two unique dataset diagrams resolves the otherwise unvalidated legacy track without pretending it was rerun. It does not delete archived evidence or assert the historical runs never happened.

### Exact captions for the pending performance figures

Use these only with the corresponding audited new asset. Curves below specify **mean with pointwise 95% Student-t intervals across seeds**; the generator must calculate those intervals. Paired effect intervals belong in the tables. Neither shaded curve overlap nor non-overlap is a paired test.

**Figure 2 — Before:** “Best-so-far population objective, mean ± standard error over 30 paired seeds. (a) ParallelParent: deleting X1→Y removes fine arms containing X1. (b) FrontDoor: adding X1↔M removes identified causal priors without changing the fine arms. Correct- and misspecified-graph QCBO trajectories coincide in both experiments and are shown by a single curve.”

**After:**

```latex
Population value of the measured-data recommendation versus total paid
intervention cost, mean with pointwise 95\% Student-$t$ intervals across
30 paired seeds. (a) ParallelParent A0/A1: deleting $X_1\to Y$ removes
fine arms containing $X_1$, imposing a population floor of $4.25$ on A1
fine-CBO recommendations. (b) FrontDoor B0/B1: adding $X_1\leftrightarrow M$
preserves fine arms but replaces identified priors by observational fallback.
Each measurement is a noisy SCM response; population scoring is external.
Recommendation quality need not be monotone. Initialization is charged.
Protected curves may be collapsed only when their complete-trace audit
passes; paired contrasts and audit outcomes are reported separately.
\label{fig:controlled-misspecification}
```

**Figure 3 — Before:** the caption beginning “MediatedChain, mean ± standard error over 30 seeds”, including the trial-7 line, trial-indexed floor, and “reaches the fine optimum.”

**After:**

```latex
MediatedChain under noisy feedback and a total intervention-cost budget
of 120, with 30 paired seeds. (a) Population value of the measured-data
recommendation; horizontal references are the fixed-coarse optimum $4$
and fine optimum $0.04$. (b) Sum of recommendation gaps at integer costs
12 through the displayed cost $b$; the fixed-coarse lower bound is
$3.96(b-11)$. Curves show means with pointwise 95\% Student-$t$ intervals.
HQCBO uses only its initial quotient and pays for all new-arm measurements.
Split costs and refused splits are reported across seeds; no universal
split trial or finite-budget attainment claim is assumed.
\label{fig:controlled-refinement}
```

**Figure 4 — Before:** “Quotient methods (orange, dashed) against their base methods (blue) under the released family protocols…” followed by eight panels.

**After, new dynamic figure:**

```latex
Coherent dynamic population comparison for the stationary, independent
and nonstat\_\allowbreak regularized SCMs. Fine/full-menu, fine/coarse-menu and
quotient/coarse-menu methods are shown separately. Each slice recommendation
is scored under that method's committed intervention prefix, with free
past variables re-simulated. Values are population outcomes to minimize,
not regret against a known global optimum. Means and pointwise 95\%
Student-$t$ intervals use 20 paired seeds; independent 100,000-particle
scoring uncertainty is reported separately. Slice boundaries are marked;
no running minimum is carried across slices. Fixed rounds do not imply
equal intervention cost. The regularized SCM is a separately defined
companion to the historical nonstationary benchmark.
\label{fig:dynamic-population}
```

**After, new MCBO figure:**

```latex
Corrected MCBO comparison on ToyGraph and PSAGraph. Fine/full-menu,
fine/coarse-menu and quotient/coarse-menu methods are compared over
100 sequential rounds, with 20 paired seeds. Curves show expected reward
of the eligible measured-data recommendation, with pointwise 95\%
Student-$t$ intervals across seeds; larger is better. ToyGraph is
deterministic at zero additive noise. PSAGraph remains stochastic and
uses an independent 100,000-draw scorer; its reward is negative PSA.
Population-best-visited values are separate diagnostics, not recommendations.
Initialization and total costs differ across menus and are reported.
\label{fig:mcbo-menu}
```

**Figure 5 — Before:** historical “cost-indexed” comparisons from the same old runs.

**After:**

```latex
Attribution controls under the matched noisy static protocol. Panels show
ParallelParent, FrontDoor and MediatedChain recommendation gaps at paid
cost checkpoints, using all prespecified configurations shown in the legend.
Means and pointwise 95\% Student-$t$ intervals use 30 paired seeds.
BO uses the joint arm, BO-S all nonempty subsets, and CBO-NP the native
CBO arms with plain priors. FrontDoor additionally includes CBO-FALLBACK,
which replaces both prior mean and variance while retaining the correct
graph and arm menu. Initialization and refinement costs are charged;
budgets are 100 for ParallelParent/FrontDoor and 120 for MediatedChain.
\label{fig:controlled-attribution}
```

### Exact structural caption operations

**Figure 6:** replace “Fine DAG with coarse partition (left)” by “Fine ADMG with coarse partition (left)”. The supplied file applies this edit.

**Figure 7:** retain its caption and append:

```latex
The ToyGraph name here refers to the historical arm-based CBO example;
the distinct model-based ToyGraph has the chain topology shown in the
MediatedChain/ToyGraph-M figure.
```

**Figure 11:** replace “DCBO stat / nonstat (QDCBO family)” with “DCBO stat (QDCBO family)”. Replace “Slices t=0,1,2; the nonstat setup shares this topology with a change point in the SEM.” with:

```latex
Slices $t=0,1,2$. The separately defined nonstat\_\allowbreak regularized companion,
including its additional cross-lag parents, is shown in
Figure~\ref{fig:dag-dynamic-regularized}.
```

The supplied stationary file applies both changes. The new companion has **14 fine edges and eight quotient edges**, one change point at t=1, and the additional parents X0→Z1 and Z1→Y2. Its source, PDF and full caption are included. This corrects the equation-parent mismatch; it does not overwrite the historical benchmark.

**Figure 13 — replace full caption with:**

```latex
\textbf{PSAGraph (QMCBO family).} Fine DAG with the
\{Aspirin, Statin\} partition (left) and quotient C-DAG (right).
Only Aspirin and Statin are manipulable; Age, bmi and Cancer remain
nonintervened variables. The simulator defines PSA, while the optimizer
maximizes reward $-\mathrm{PSA}$. Dashed boxes mark manipulable clusters.
```

Changing the caption alone is insufficient: use the supplied repaired TikZ panels as well.

## Table swaps: all five original tables

| Original table | Action and exact replacement | State |
|---|---|---|
| 1, p.8: eight-task Base/Quotient final values | Remove every old numeric cell. Remove historical CBO Toy/Synthetic/Coral rows. Replace by **two tables**: `dynamic-population-summary.tex` and `mcbo-menu-summary.tex`, with distinct estimands and all three configurations. | PENDING audited 180 /120 |
| 2, p.14: graph edits plus numeric sup deviations | Replace with `controlled-perturbations-structural.tex`: retain the four structural edit definitions, remove all measured-deviation columns, mark A2/A3 as diagnostics only. | READY |
| 3, p.15: controlled results | Replace `minimal_ablations.tex` with `controlled-noisy-summary.tex`, covering exactly the new 24 configurations. | PENDING audited 720 |
| 4, p.17: mixed historical cost accounting | Replace `cost_accounting.tex` with `controlled-noisy-costs.tex`: paid-cost accounting for the same 24 configurations, including initialization and refinement. | PENDING audited 720 |
| 5, p.17: identical-final MCBO counts | Replace `qmcbo_e2.tex` with `mcbo-protected-traces.tex`, generated from all 80 edited runs paired with their unedited counterparts. | PENDING audited MCBO 200 |

The new numerical-settings table in Appendix B.3 is an additional table of **design constants**, ready now. It is not a replacement for a result table. Keep it unnumbered as supplied or give it a distinct label and allow later tables to renumber.

### Exact schema and caption for each pending table

**New dynamic Table 1 component** (`tab:dynamic-population`). Rows: stat, ind, nonstat_regularized, each with fine/full, fine/coarse, quotient/coarse. Columns: setup; configuration; n; committed population value at slices0/1/2; their sum; total intervention cost; scoring-SE summary. Provide a separate paired-contrast panel for quotient/coarse minus fine/coarse, per slice and sum, with mean and95%t interval. Use an upper bound, not an independence calculation, for the SE of a sum when only marginal scoring SEs are retained.

```latex
Committed-policy population outcomes in the coherent dynamic comparison,
20 paired seeds per setup and configuration; lower is better. Slice values
are evaluated under each method's own intervention history. The paired
contrast is quotient/coarse minus fine/coarse, with pointwise 95\%
Student-$t$ intervals. The slice sum is a sequential-policy outcome,
not regret against a known optimum. Scoring uncertainty and intervention
cost are reported separately. The nonstat\_\allowbreak regularized row is a new SCM
and does not reuse historical nonstationary results.
```

**New MCBO Table 1 component** (`tab:mcbo-menu`). Rows: ToyGraph/PSAGraph × three configurations. Columns: n; final expected recommendation reward; sum of expected recommendation rewards across 100 rounds; initial measured-row count; intervention cost; scoring SE. Separate primary paired quotient/coarse-minus-fine/coarse effect panel for the final reward and sum, mean and95%t interval. Fine/full summaries remain separate.

```latex
Corrected MCBO recommendation outcomes after 100 sequential rounds,
20 paired seeds per configuration; larger reward is better. The primary
paired contrast holds the coarse menu fixed. Intervals are pointwise 95\%
Student-$t$ intervals across seed-level differences. Initial measurements,
total intervention cost and independent scoring uncertainty are reported;
fixed rounds do not imply equal cost. PSA reward is negative PSA.
```

**Table 2** supplied structural table has columns: condition; edit; quotient changed; fine MIS changed; fine prior changed; evidence role. Its caption says exactly:

```latex
Controlled graph perturbations and their structural consequences.
A1 and B1 occur in the final matched noisy matrix; A2 and A3 are
engineering diagnostics. An unchanged quotient predicts pathwise
invariance only under the shared-input quotient contract. This table
reports graph properties, not measured deviations or replication counts.
```

**Table 3** (`tab:controlled-complete`). Exactly 24 rows per the AppendixC inventory. Columns: SCM; condition; method; n; mean population recommendation value atB; mean gap atB; mean sum of gaps at integer costs12:B. Add95%t intervals for the five prespecified QCBO-minus-CBO condition contrasts in a separate panel. Keep method summaries and paired differences distinguishable. Report failures outside numeric summaries; do not compute an incomplete-matrix estimate and label it n=30.

```latex
Matched noisy controlled results at total cost budgets 100
(ParallelParent/FrontDoor) and 120 (MediatedChain), with 30 paired seeds.
Recommendations are chosen by measured responses and scored separately.
The primary endpoint is population recommendation gap at budget; the
secondary sum uses integer cost checkpoints 12 through the budget.
Paired QCBO-minus-CBO differences have pointwise 95\% Student-$t$
intervals. Other method summaries are descriptive unless identified
as a prespecified contrast. All initialization and split purchases count.
```

**Table 4** (`tab:controlled-costs`). Same24 rows. Columns: budgetB; native arm count; initial cost; sequential cost; split cost; total spent; unspent; number of purchased rows; runtime. HQCBO additionally: fraction split; distribution of first split cost; refused-split count. Report seed mean and range where cost varies; retain exact budget cap. Do not carry forward old common-C* values such as62.

```latex
Resource accounting for the matched noisy static matrix. Every initial,
sequential and refinement measurement is charged once, at one unit per
intervened variable. Means and ranges summarize30 seeds where counts
vary. Total spend never exceeds the declared budget; unspent budget and
unaffordable refused splits remain visible. Runtime is reported separately
from experimental cost. Fixed-budget outcomes appear in the result table.
```

**Table 5** (`tab:mcbo-protected-traces`). Four rows: Toy/fine-full, Toy/quotient-coarse, PSA/fine-full, PSA/quotient-coarse. Each row has 20 paired edited/unedited runs. Columns: edit; quotient preserved; n; exact eligible-action matches; coordinate deviations; measurement deviations; recommendation-index matches; cost matches; scorer deviations; final-score matches. The frozen acceptance gate requires **exact decoded trace equality, with no tolerance**, excluding the separately logged `acquisition_diagnostics` fields. This is equality of decoded JSON fields, not bitwise IEEE floating-point identity. Deviations are descriptive diagnostics and must not weaken the acceptance rule. Fine/full is an unprotected comparator: a change is allowed and not a failed quotient theorem. Do not collapse these checks into the old 20/20 final-equality statistic.

```latex
Complete-trace comparison for protected graph perturbations in the
corrected MCBO protocol: delete $X\to Z$ in ToyGraph and add
aspirin$\to$statin in PSAGraph. Each edited run is paired with its
unedited counterpart at the same seed and settings. Both edits preserve
the chosen quotient; invariance is predicted for the quotient method,
not for the fine comparator. Action, measurement, recommendation, cost
and score checks are reported separately. Protected acceptance requires
exact equality of decoded trace fields, excluding acquisition diagnostics;
no numerical tolerance is used. Reported deviations are descriptive.
Final-score equality alone is not a trajectory-invariance check.
```

**Result-generation completion gate:** the output agent must emit the new files from the complete frozen matrices, the precise generating command, source/configuration/output hashes, counts and audit result. Then replace each PENDING row with its actual file and fill the empirical paragraphs. Historical `minimal_*.pdf`, `family_suite.pdf`, `family_effects.tex`, `ceo_minimal.tex` and `qmcbo_e*.tex` are not fallbacks for missing corrected results.


## Bibliography: exact reference operations

The ZIP contains `references-corrections.bib`. Merge by work identity; do not append duplicates. The original numbers below will change after BibTeX. Remap **all** citations of a deleted duplicate to its canonical key before removing it.

| Original PDF reference | What is written / problem | What should be written / operation |
|---|---|---|
| [3], p.9 | NeurIPS volume38 paper dated2026 | `anand2025clusterdiscovery`:2025, volume38 (Main Conference), pp.86792–86819. Official metadata is in the supplied BibTeX. |
| [4] and[5], p.9 | Two records for the cluster-DAG identification work; [4] has an inconsistent title/volume82 | Merge as `anand2023clusterdags`: *Causal Effect Identification in Cluster DAGs*, AAAI37(10),12172–12179(2023), DOI10.1609/aaai.v37i10.26435. |
| [13] and[14], p.10 | Duplicate/incomplete records of *Structural Causal Bandits: Where to Intervene?* | Merge as `lee2018structuralbandits`, NeurIPS31(2018). Do not invent a page range absent from the official BibTeX. |
| [9], pp.9–10 | AT-UCB work | Use `dyer2025atucb`, arXiv2509.04296; do not add unverified workshop metadata. |
| [29], p.10 | CAMAB work | Use `zennaro2024camab`, UAI2024/PMLR244,4109–4139. |
| [27], p.10 | MCBO | Retain verified ICLR2023 work as `sussex2023mcbo`; cite its existing vector-output scope. |
| [1], p.8 | Original CBO | Retain verified PMLR108,3155–3164(2020) as `aglietti2020cbo`; cite its discussion of natural mediator values outside intervention domains. |

Primary records: [NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7d998ff3e6ecce821986c9dd876be9bc-Abstract-Conference.html), [AAAI cluster identification](https://ojs.aaai.org/index.php/AAAI/article/view/26435), [NeurIPS2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/c0a271bc0ecb776a094786474322cb82-Abstract.html), [AT-UCB author manuscript](https://arxiv.org/html/2509.04296v1), [CAMAB official record](https://proceedings.mlr.press/v244/zennaro24a.html), [MCBO author manuscript](https://arxiv.org/html/2211.10257), [original CBO official record](https://proceedings.mlr.press/v108/aglietti20a.html). Verification applies to these changed/cited records; it is not a fresh verification of every unrelated bibliography entry.

## Final application checklist

- [ ] Each T-entry is applied once. The old §3.2 and both old AppendixC protocol paragraphs are fully replaced; no conflicting historical noise or response-oracle sentences remain.
- [ ] Proposition3, Proposition4 and Lemma1 use the replacements, preserving original labels. Theorem1, Proposition2, valid zero-price/refinement results and Proposition5 remain intact.
- [ ] Remove the duplicated “Let” in §3.1; use consistent publication names CBO-NP, BO-S, CBO-FALLBACK and HQCBO. Native code identifiers can appear in the release manifest.
- [ ] Resolve Figure2 references as panels(a)/(b); no nonexistent(c)/(d). Remove the eight-task claim, old trial7 line, old unconditional optimum-attainment wording and “zero observation noise” for PSA.
- [ ] Do not globally replace every occurrence of “best-so-far”: mathematical definitions, measured incumbents and explicitly named oracle diagnostics have distinct meanings. Correct each plotted metric according to this sheet.
- [ ] New dynamic captions use nonstat_regularized, one change point at t1 and its true lagged parents. No old nonstat cell is relabeled.
- [ ] Every pending numerical asset comes from the complete matching matrix and passes its strict audit. No old20/20 count, final mean, standard error or cost total is copied over.
- [ ] All interval definitions, objective directions, sample sizes, paid costs and row counts agree between text, plots and tables. PSA maximizes negative PSA; static and dynamic objectives minimize.
- [ ] Replace real citation keys consistently, compile BibTeX, and remove duplicates without changing unrelated valid records.
- [ ] Supply real release URL/immutable identifier; no literal `RELEASE_URL` or `IMMUTABLE_RELEASE_ID` remains.
- [ ] Compile and inspect the assembled Overleaf PDF at publication size: theorem/figure/table references, float order, graph colors, axes, legends, uncertainty, table widths and venue page limits. This final assembly check remains necessary even though the isolated replacement fragments compile.

## Package contents and limits

`figures/fig06-parallelparent.tex`, `figures/fig11-stationary.tex`, `figures/fig13-psa.tex` are full replacement figure environments. They share `figures/styles.tex` and the existing TikZ/adjustbox dependencies. Preserve or remap labels already used by your main source. `figures/dynamic-regularized-figure.tex` inserts the new PDF with its caption. `tables/controlled-perturbations-structural.tex` is the replacement structural Table2. `replacement-blocks.tex` is a copy/paste catalog with edit/block comments, **not** a file to input wholesale into the manuscript. All expected future numeric result filenames in this sheet are specifications, not supplied result assets.
