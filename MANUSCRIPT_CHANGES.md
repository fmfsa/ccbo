# Manuscript change-list — engine v3 (post-review)

**Source of truth:** the authors' pasted `aistats2026` source ("Coarsening Confers
Robustness: Causal Bayesian Optimization under Graph Misspecification"), *not*
`paper/qcbo_aistats.tex`. Section numbers below follow that source.
**Number provenance:** `results/v3/` (engine v3, branch `fix/engine-v3`); all
numbers below are final (runs LSF 29365450/51/82/83, gate 0 failures). **Status tags:** `KNOWN-NOW` (text is final) /
`PENDING-RESULTS` (structure final, numbers pending).

The reviewer's replacement blocks (their §§1–17) are reused where they were
right, and corrected where the engine changed: the GP noise is now *fixed*
(not free), the prior variance is the *predictive* `Var[Y|do]` under the
fitted plug-in model (fitted noise + between-row spread + the outcome GP's
posterior variance — the epistemic-free variant was run on 2026-09-09 and
retired, see `RERUN_ENGINE_V3_REPORT.md` §9), front-door / g-computation are
*integrated* (so the
"mean-substitution approximation" caveat must NOT be pasted), the refinement
has two variants, and the cost axis charges the initial design.

---

## 0. Preamble — `KNOWN-NOW`

**Add** (Table 1 and the new tables use them):
```tex
\newcommand{\BO}{\textsc{BO}}
\newcommand{\BOS}{\textsc{BO-S}}
\newcommand{\CBONP}{\textsc{CBO}$_{\text{-np}}$}
\newcommand{\QCBONP}{\textsc{QCBO}$_{\text{-np}}$}
\newcommand{\HQCBOGF}{\textsc{HQCBO-GF}}
```

## 1. Introduction — `KNOWN-NOW`

**Written (contribution 4):** "Experiments isolate exploration set and prior corruption, measure the action-space price, and test portability across optimizer families."

**Should read:** "Controlled experiments separate arm exclusion from prior corruption, quantify the population price of a fixed partition and its recovery by graph-informed and graph-free refinement, and isolate the contribution of the causal prior with matched no-prior and structure-free baselines; archived family experiments assess the quotient constructions under their base methods' protocols."

**Written (contribution 1):** "…together with an adaptive variant." → **Should read:** "…together with two adaptive variants that refine the partition with or without consulting the fine hypothesis graph."

## 2.1 Causal models — `KNOWN-NOW`

Move here (from §4.3, before Theorem 1) the sentence defining a **semi-Markovian SCM** ("mutually independent exogenous variables, each a parent of at most two endogenous variables; shared exogenous parents are the bidirected edges of the latent projection"), and the NPSEM sentence now embedded in Definition 2. §4.3 then just says "For a semi-Markovian SCM (§2.1)…".

## 2.2 Causal Bayesian optimization — `KNOWN-NOW`

**Replace the whole subsection** (keep the CBO / MIS citations; the teal CMAB paragraph moves to §6, see §6 below):

```tex
\subsection{Causal Bayesian optimization}
Causal Bayesian optimization (\CBO{}) \citep{agliettiCausalBayesianOptimization2020b}
searches jointly over \emph{which} variables to intervene on and the
\emph{values} to assign them. A retained intervention set $A$ is an
\emph{arm}; each arm has a response surface $f_A(a)=\E[Y\mid\doo(A=a)]$ over
its own domain $D(A)$, so choosing an arm and choosing its values are separate
decisions. One run proceeds as follows.

\textbf{(1) Arms.} The supplied graph $\widehat{\Gg}$ determines which arms are
retained: the minimal intervention sets (MIS)
\citep{leeStructuralCausalBandits,agliettiCausalBayesianOptimization2020b}.
Writing $\widehat{\Gg}_{\bar A}$ for the graph with incoming edges to $A$
removed,
\begin{equation}
  \ES(\widehat{\Gg})=\mathrm{MIS}(\widehat{\Gg})
  :=\{\emptyset\neq A\subseteq\mathbf X:\ A\subseteq\An_{\widehat{\Gg}_{\bar A}}(Y)\}.
  \label{eq:mis}
\end{equation}
Every variable of a retained arm can still affect $Y$ once the arm is
intervened on. In $X_1\to X_2\to Y$, for instance, $\doo(X_1=x_1,X_2=x_2)$
blocks $X_1$ by fixing $X_2$, so the joint arm is redundant and only the
singletons are retained.

\textbf{(2) Causal priors.} For each arm the graph decides whether the
observational data identify the effect. If it does, the observational
estimate of the identified functional supplies a Gaussian-process prior
\emph{before any intervention on that arm is performed}:
\begin{equation}
  f_A\sim\mathcal{GP}\!\big(m_A^{\widehat{\Gg}},\,k_A^{\widehat{\Gg}}\big),
  \qquad
  k_A^{\widehat{\Gg}}(a,a')=\sigma_f^2\exp\!\Big(-\tfrac{\|a-a'\|^2}{2\ell^2}\Big)
  +\sigma_A^{\widehat{\Gg}}(a)\,\sigma_A^{\widehat{\Gg}}(a'),
  \label{eq:cbo-gp}
\end{equation}
with $m_A^{\widehat{\Gg}}(a)=\widehat{\E}_{\widehat{\Gg}}[Y\mid\doo(A=a)]$ and
$\sigma_A^{\widehat{\Gg}}(a)^2=\widehat{\mathrm{Var}}_{\widehat{\Gg}}[Y\mid\doo(A=a)]$
the plug-in estimates of the identified mean and variance functionals
(Appendix~\ref{app:estimators} gives the estimators). Otherwise the arm
starts from the \emph{constant observational fallback} $m_A=\bar Y_{\rm obs}$,
$\sigma_A^2=s^2_{\rm obs}(Y)$.

\textbf{(3) Surrogates.} Each prior is conditioned on the arm's initial
interventional design. Interventional outcomes in our suites are exact
population expectations, so the likelihood variance is fixed at $10^{-10}$;
$(\sigma_f^2,\ell)$ are re-fitted by marginal likelihood after every update.

\textbf{(4) Observe or intervene.} With probability $\epsilon_t$
(§\ref{sec:exp-design}) the optimizer takes an \emph{observation} action:
it reveals a fresh batch of observational rows, re-estimates every identified
functional and rebuilds every arm's GP. Otherwise it intervenes.

\textbf{(5) Intervene.} It picks the arm and value maximizing expected
improvement per unit cost,
$\alpha_{A,t}(a)=\E[(y^{\rm best}_t-f_A(a))_+\mid\mathcal D_t]/c(A,a)$,
executes $\doo(A=a)$ in the true system, and updates that arm's GP and the
incumbent (the best executed intervention so far). Steps (4)–(5) repeat for
$T$ trials.

The supplied graph therefore enters through the arm set and the complete
prior,
\[
\widehat{\Gg}\longmapsto
\Big(\ES(\widehat{\Gg}),\ \{(m_A^{\widehat{\Gg}},\sigma_A^{\widehat{\Gg}})\}_{A\in\ES(\widehat{\Gg})}\Big),
\]
and a graph error can remove a useful arm, corrupt its prior, or both. The
fine and quotient methods share one implementation, so their comparison
isolates the graph resolution.
```

**Appendix `app:estimators` — replace "Causal-effect estimation"** (`KNOWN-NOW`):

```tex
\subsection{Causal-effect estimation and the complete prior}
Identification and numerical evaluation are distinct. A complete
identification check (Ananke's ID algorithm) is run on the supplied C-DAG for
every arm; an identified query is then evaluated by the first applicable
estimator of the cascade backdoor $\to$ frontdoor $\to$ $g$-computation on the
latent-expanded DAG. An identified query whose functional none of the three
estimators covers, a non-identified query, and an identification-algorithm
error (recorded separately) all use the constant observational fallback; no
arm is ever removed on identification grounds.

\emph{Backdoor} ($Z$ a valid adjustment set): a GP $Y\mid A,Z$ is fitted on
the observational rows and
$\widehat{\E}[Y\mid\doo(A=a)]=\tfrac1n\sum_j\widehat{\E}[Y\mid a,z_j]$.
\emph{Frontdoor} ($M$ the mediator set): GPs $M_j\mid A,M_{<j}$ and
$Y\mid M,A$ are fitted; the mediator law is the plug-in Gaussian
$\mathcal N(\widehat\mu_M(a),\widehat\sigma_M^2)$ and the outer integral
$\int p(m\mid a)\,\tfrac1n\sum_j\widehat{\E}[Y\mid m,a_j]\,dm$ is evaluated by
Gauss--Hermite quadrature (20 nodes per mediator, product grid up to two
mediators, seeded draws beyond). \emph{$g$-computation}: every propagated
variable receives a plug-in conditional GP given its parents and the earlier
members of its own cluster (chain rule), and the functional is evaluated by
seeded common-random-number draws through the DAG. Intermediate conditionals
are never replaced by their means.

The prior variance is the predictive variance of $Y$ under $\doo(A=a)$
in the fitted plug-in model, obtained from the same identified functional:
$\sigma_A(a)^2=\widehat{\mathrm{Var}}[Y\mid\doo(A=a)]
=\widehat\sigma^2_{\rm noise}+\sum_i w_i(\mu_i-m_A(a))^2+\sum_i w_i\,s_i^2$,
where $\mu_i$ and $s_i^2$ are the outcome GP's posterior mean and variance at
the $i$-th conditioning row and $w_i$ the mixture weights (adjustment rows,
quadrature nodes or propagation draws). The three terms are the fitted noise,
the between-row spread of the conditional mean, and the regression's own
posterior uncertainty; the last one is what the reference CBO implementation
averages (`np.mean(gp.predict(rows)[1])`) and what keeps the prior wide where
the outcome regression is unsupported (e.g.\ Coral's $D,T$ ranges lie three
orders of magnitude outside the observational support; without this term the
prior variance there collapses to the noise floor, the rank-one kernel term
vanishes and the arm's surrogate can no longer learn from its own pulls). The rank-one term
$\sigma_A(a)\sigma_A(a')$ is positive semidefinite and its input derivative
enters the acquisition gradient (central differences of $\sigma_A^2$ and of
$m_A$, whose estimators are deterministic and continuous in $a$ by
construction). The interventional likelihood variance is fixed at $10^{-10}$.
```

## 3.1 Arm-based QCBO — `KNOWN-NOW`

**Written:** "For each arm … a complete causal-effect identification step determines whether the observational distribution identifies its prior mean. The resulting prior hence given by:"

**Should read:** "For each retained arm, a cluster-level identification query decides whether the observational law identifies its effect under the supplied C-DAG; identification and numerical evaluation are distinct (Appendix~\ref{app:estimators}). The prior is"
and Eq. (two-tier) becomes the pair $(m_A^\Pi,\sigma_A^\Pi)$: identified-and-supported $\Rightarrow$ $(\widehat{\E}[Y\mid\doo(A=a)],\widehat{\mathrm{Var}}[Y\mid\doo(A=a)])$; otherwise $(\bar Y_{\rm obs}, s^2_{\rm obs}(Y))$. Replace "uninformative" everywhere by "constant observational fallback".

**Written:** "We take every nonmanipulable variable, to be a singleton cluster. Hence a cluster that contains a manipulable variable contains only manipulable variables."

**Should read:** "Our implementation keeps nonmanipulable variables, including the target, as singleton clusters and groups only manipulable variables. This is an implementation restriction, not a requirement of the construction: purely nonmanipulable clusters can be treated as vector-valued observational nodes that are never intervened on (Appendix~\ref{app:nonmanip-clusters})."

**Insert appendix `app:nonmanip-clusters`** — use the reviewer's block verbatim (regrouping $\mathbf N$ leaves $\mathcal U(\Pi)$, $V(\Pi)$, $\Delta(\Pi)$ unchanged; can lose identification and add potential ancestry, e.g. $X\to Z_1$, $Z_2\to Y$ merged into $X\to\{Z_1,Z_2\}\to Y$; acyclicity must be re-checked; current wrapper rejects it; mixed clusters need a different intervention semantics).

## 3.2 Price of restricting interventions — `KNOWN-NOW`

**Insert before $\mu^\star(A)$:** "For these population statements each $D(A)$ is nonempty and compact and $f_A$ is finite and continuous on $D(A)$; the domain of an intervention set does not depend on the partition representing it. Without attainment, replace minima by infima."

**Written:** "The MIS-reduction result in Appendix~\ref{app:proofs} connects this action-family value to the quotient exploration set."

**Should read:** "The MIS-reduction lemma preserves the optimum over $\{\varnothing\}\cup\mathrm{MIS}$; our experiments search nonempty interventions, so equating the all-actions optimum with the nonempty-MIS optimum additionally requires that no null-equivalent action improves on the best retained arm — true of the controlled SCMs, whose optimal intervention is a retained nonempty arm."

**Appendix, after Lemma MIS-reduction:** replace the "Consequently, when the null intervention is handled consistently…" sentence with the reviewer's §5 block (the reduction is over $\{\varnothing\}\cup\mathrm{MIS}(\Gg^\Pi)$; MIS may be empty; the experimental convention is specified separately). Add the 3-line ParallelParent instance the red box asks about (fine arms $\{X_1\},\{X_2\},\{X_1,X_2\}$; $A=\{X_1,X_2\}$ has $A'=A$; under A1, $A'=\{X_2\}$) — it is short and helps.

## 3.4 When coarsening is lossless — `KNOWN-NOW`

- Rename Definition 2 → **"Exogenous separation under cluster completion"** (keep label `def:ccc`); Corollary → **"Sufficient structural condition for zero price"**.
- **Definition 1 restated** (the current wording is violated by every Gaussian-noise SCM, since natural supports are unbounded): "A partition has full intervention support when, for every fine intervention $\doo(A=a)$, $\inf_{b\in D_B(a)}\E[Y(a,b)]\le\inf_{b\in\operatorname{supp}B(a)}\E[Y(a,b)]$ — in particular whenever every attainable natural value of $A^\Pi\setminus A$ lies in the completed domain."
- **Prop. `free`:** premise "$B(a)$ independent of the whole process $\{Y(a,b):b\in D_B(a)\}$, jointly measurable and integrable" (the pairwise version admits the discontinuous counterexample the reviewer gives); proof as in the reviewer's §6 block. Lemma `ccc` proof: last sentence → process-level independence from disjoint independent exogenous sets.
- **Explanatory paragraph after the corollary** → reviewer's §6 block ("stronger than bidirected edges staying inside clusters; an observed nonmanipulable common cause can violate it; stated on the augmented structural graph; both conditions sufficient, not necessary").
- **Insert at the end of §5.3 (FrontDoor):** "This example is also lossless at the population level ($\doo(M=1)$ and every completion $\doo(X_1=x,M=1)$ attain $0$), yet it violates the separation premise ($U$ is an exogenous ancestor of both $X_1$ and $Y$) and its Gaussian natural values exceed any bounded box: the sufficient conditions are not necessary. Appendix~\ref{app:lossless-examples} isolates the two conditions analytically." *(Do not use Fig. 2(b) as the full-support exemplar.)*
- **Insert appendix `app:lossless-examples`** — the reviewer's §8 analytic constructions (bounded-noise MediatedChain price curve $\Delta_h=[(4-h)_+^2-0.04]_+$; sign-noise crossing example $\Delta_\alpha=(2\alpha-1)_+$; contained-latent control). These are analytic, independent of the reruns.

## 3.5 Adaptive refinement — `KNOWN-NOW`

**Replace the HQCBO paragraph with:**

```tex
Hierarchical quotient CBO (\HQCBO{}) begins from a coarse partition and a
declared split hierarchy; when a plateau rule ($k{=}5$ trials, threshold
$10^{-3}$) detects stalled progress it splits the clusters that meet the
incumbent arm. We evaluate two variants that differ in what the split is
allowed to know. \emph{Graph-informed} \HQCBO{} reads the refined quotient
from the fine hypothesis graph: the new arms are the refined MIS with C-DAG
priors, a cyclic refined quotient is refused, and the phase is therefore
protected only against errors invisible at \emph{every} visited partition.
\emph{Graph-free} \HQCBOGF{} consults no graph after the split: the
hierarchy exposes every nonempty union of the split cluster's sub-clusters
as a new arm with a plain GP (constant mean, RBF kernel), historical arms and
their data are retained, and no refined-graph pruning, prior or validity
check is run — its protected class is that of the initial quotient. Both
variants initialize each newly exposed arm with three exact evaluations,
charged to the cost axis at the trigger trial, and begin the refined phase
with one observation action when observational budget remains. The
hierarchy and plateau rule are heuristic inputs; refinement is not
guaranteed to expose a useful intervention.
```

**Corollary `adaptive-invariance` restated:** "Fix the hierarchy, initial data, numerical settings and coupled randomness. If every graph-dependent decision — candidate splits, acceptance tests, rejected proposals, arm construction and priors — uses only quotient inputs that agree between the coupled runs, the complete adaptive trajectories agree. For \HQCBOGF{} these inputs are the initial quotient alone; for \HQCBO{} they include the refined quotients."

## 4. Dynamic and model-based quotient variants — `KNOWN-NOW`

- **MCBO acquisition:** "It then optimizes interventions using a GP upper confidence bound (GP-UCB) acquisition rule." → reviewer's §11 block (mechanism-level optimistic perturbations $m_C+\beta\,\eta_C\odot s_C$, $\eta_C\in[-1,1]$, composed through the graph; not a single target-level confidence bound).
- **Joint mechanisms:** reviewer's §11 wording (multi-output GP with task covariance and Gaussian residual covariance; a statistical approximation; soundness concerns the exact conditionals).
- **Factorisation display:** `P(\mathbf X_{\Pi\setminus S}\mid\doo(\mathbf X_S=\mathbf x_S)) = \prod_{C\notin S} P(\mathbf X_C\mid\mathbf X_{\operatorname{pa}(C)})|_{\mathbf X_S=\mathbf x_S}`.
- **Singleton recovery (Cor. portability + Prop. identity):** "At the singleton partition the quotient graph, intervention sets and mechanism parent sets recover their fine counterparts; numerical identity of complete runs additionally requires matching implementations, normalisation, fitting order and random streams. It holds by test for QDCBO against the (corrected) stock DCBO; for QMCBO it is structural."
- **QDCBO contract:** "QDCBO satisfies it within every time slice." → "QDCBO obeys the contract when the within-slice and the temporal-transition quotient inputs agree, together with the intervention specification and all numerical inputs." **Add** (the corrected-semantics statement, engine v3): "The stock DCBO implementation fits transition mechanisms on the wrong time slice and drops zero-valued interventions (truthiness clamping). Both defects are corrected in our baseline and quotient implementations alike; a historical mode reproducing the stock behaviour is retained only for the finest-partition identity test. All dynamic results in this paper use the corrected semantics." *(Do NOT paste the reviewer's "archived dynamic implementation preserves two baseline behaviors" paragraph — that described the pre-v3 archive.)*

## 5.1 Evaluation design — `KNOWN-NOW`

**Replace** "Each SCM provides one shared observational dataset; the optimizer receives its first $n_{\mathrm{obs}}=100$ observations." **with:**

```tex
\textbf{Observation--intervention protocol.} Following the released CBO
loop, every trial is either an observation or an intervention. Each SCM
provides one fixed observational pool; a run starts from its first
$n_{\mathrm{obs}}=100$ rows and may acquire at most $50$ more in batches of
$20$ previously unseen rows (last batch truncated), so $N_{\max}=150$. At
trial $t$ the optimizer observes with probability
$\epsilon_t=\operatorname{clip}\!\big[\tfrac{\mathrm{Vol}\,C(\mathcal D^O_t)/\mathrm{Vol}\,D(\mathbf X)}{N_t/N_{\max}},0,1\big]$,
with $C(\mathcal D^O_t)$ the convex hull of the observed manipulable values
and $N_t$ the current number of rows;\footnote{The released implementation
divides by $N_t/N_{\max}$ whereas the CBO paper prints a product; we keep
the released schedule. On \textsc{ParallelParent} at $N_t=100$ the two give
$\epsilon_t\approx0.11$ and $0.05$.} the first trial of a run is a forced
observation and the second a forced intervention, and once $N_t=N_{\max}$
the optimizer always intervenes. Every observation re-estimates the
identified functionals and rebuilds every arm's GP. Observation trials count
against $T$ but cost nothing.

\textbf{Cost axis.} Under unit per-variable costs, \texttt{cum\_cost} charges
the initial interventional design (three points per arm on this suite, ten
on the family suite) at trial $0$ and the refinement arms' split design at
their trigger trial; the trial axis is unchanged.

\textbf{Baselines and ablations.} Besides \CBO{} and \QCBO{} we run plain
\BO{} (one joint arm over all manipulable variables, plain GP, no observation
actions), \BOS{} (every nonempty subset of the manipulables as an arm, plain
GPs, no observation actions: the fine arm menu without any graph), and the
no-prior ablations \CBONP{} and \QCBONP{} (the same arms as \CBO{} / \QCBO{}
with plain GPs and observation actions on, isolating the causal prior). \BO{}
and \BOS{} read no graph, so their trajectories are identical across the
misspecification conditions of an SCM (Table~\ref{tab:minimal-taxonomy}).
```

**Also in §5.1:** "We use $30$ paired seeds…" — add: "The $30$ seeds share one observational dataset per SCM; the reported standard errors therefore describe optimizer and initial-design variation conditional on that dataset."

**MediatedChain SCM (§5.4):** `X_2=2X_1+\varepsilon_2, Y=(X_2-4)^2+\varepsilon_Y` → `X_1=0.5\varepsilon_1,\ X_2=2X_1+0.2\varepsilon_2,\ Y=(X_2-4)^2+0.1\varepsilon_Y` (standard-normal $\varepsilon$; the $0.04$ follows from $0.2^2$).

**Invariance measurement language (§5.1 + App. exp. details):** "Protected pairs are compared on the persisted full-precision decision logs — the observe/intervene draws, selected arms, intervention values, outcomes and incumbents — and are identical; in the dynamic and model-based suites the logged decisions (intervention sets and levels; chosen $X$ per iteration) and values agree on every seed."

**Reproducibility statement (App. exp. details):** add: "Every per-seed trace is archived with its full-precision decision log and per-file checksums; the verification gate (`scripts/verify_traces.py`) recomputes every displayed number and re-tests every invariance claim from the archives, and `scripts/reproduce_paper.sh` regenerates all figures and tables from them. Re-executing a suite reproduces its archive bit-for-bit on the CPU model recorded in the archive manifest and to floating-point rounding on other models (BLAS kernels differ across micro-architectures; the acquisition optimizer amplifies such differences into different intervention values without changing the aggregates). Identification order is independent of the interpreter's hash seed, which is nevertheless pinned in every run script."

**Random stream $\omega$ (§4.1 definition of $\xi$):** "…all coupled randomness: optimization, observational sampling, numerical causal-effect estimation and experimental outcomes."

## 5.2–5.5 Numbers — `FINAL (v3)`

| Location | Written | Should read | Emitter |
|---|---|---|---|
| §5.2 correct-graph CBO final | `0.0016±0.0002` | **`0.0004±0.0001`** (raw 3.93e-04) | `artifacts/summaries/minimal_exact.json` → `groups.ParallelParent_A0_CBO` |
| §5.2 misspecified CBO final | `4.2500±0.0000` | **`4.2500±0.0000`** (unchanged) | `…ParallelParent_A1_CBO` |
| §5.2 regrets | `4.21±0.37` and `216.23±1.12` | **`3.65±0.35` and `216.15±1.11`** | same groups, `cumulative_regret_*` |
| §5.2 A3 deviation | `21.45` | **`21.43`** (Table 1 prints 21.4) | `paired.ParallelParent_A0_vs_A3_QCBO` |
| §5.3 paired ΔR50 | `5.91±1.04` | **`6.31±1.04`** (95 % CI [4.19, 8.44]) | `paired.FrontDoor_B1_minus_B0_CBO_cumulative_regret` |
| §5.3 finals | `0.0003±0.0003` / `0.0035±0.0022` | **`0.00006±0.00004` / `0.00393±0.00223`** (5 dp; B0 is $6\times10^{-5}$) | `groups.FrontDoor_B0_CBO`, `…B1_CBO` |
| §5.4 finals | `0.0400, 4.0000, 0.0400` | **`0.0400±0.0000`, `4.0000±0.0000`, `0.0400±0.0000`**; \HQCBOGF{} **`0.0400±0.0000`** | `groups.MediatedChain_C0_*` |
| §5.4 regrets | `5.46±1.61, 243.23±0.90, 36.97±1.18` | **`5.51±1.64`, `243.15±0.90`, `35.52±1.05`**; \HQCBOGF{} **`37.74±1.41`** | same |
| §5.4 trigger | `7.4±0.1` | **`7.0±0.0`** (all 30 seeds trigger at trial 7; all splits accepted) | `refinement.HQCBO` |
| §5.4 new sentence | — | "Graph-free \HQCBOGF{} finishes at the same `0.0400±0.0000` with $R_{60}=37.74±1.41$; paired against \HQCBO{}: $\Delta$final $+1.3\times10^{-7}$ (indistinguishable), $\Delta R_{60}=+2.23$ $[+0.86,\,+3.59]$ — the causal priors after the split buy about two regret units, not the optimum." | `refinement.HQCBOGF_minus_HQCBO` |
| §5.2 ablation sentence | — | "On the correct graph the no-prior \CBONP{} reaches `0.0004±0.0001` ($R_{50}=23.03±2.61$ vs `3.65±0.35` with priors), \QCBONP{} `0.0006±0.0002` (`19.09±2.17`), the structure-free \BOS{} `0.0005±0.0002` (`17.49±2.08`) and \BO{} `0.0037±0.0035` (`5.83±1.23`); under A1 \CBONP{} is confined to the same `4.2500±0.0000` floor as \CBO{}, while \BOS{} and \BO{} are unaffected by construction." | `paper/tables/minimal_ablations.tex` |
| §5.3 ablation sentence | — | "Without priors the B1 corruption cannot act: \CBONP{} is identical under B0 and B1 ($\Delta R_{50}=0$) at `0.0011±0.0003` ($R_{50}=8.17±0.92$), so the 6.3-unit paired increase is entirely the price of a corrupted prior; \QCBONP{} `0.0009±0.0003` (`15.63±1.89`) vs \QCBO{} `0.0015±0.0006` (`11.18±1.37`); \BOS{} `0.0013±0.0004` (`12.58±1.53`); \BO{} `0.40±0.15` (`30.62±6.59`)." | same |
| §5.4 ablation sentence | — | "\CBONP{} also reaches $y^\star$ ($R_{60}=8.44±1.96$, paired $+2.92$ [1.76, 4.08] over \CBO{}); \QCBONP{} sits on the same floor as \QCBO{} (`245.32±1.11`); \BOS{}, which owns the $\{X_1\}$ arm, reaches $y^\star$ with $R_{60}=12.00±2.76$; \BO{}'s joint arm shares the fixed-partition floor (`4.008±0.008`)." | same |
| Table 3 CBO rows | `−2.16/−2.13`, `−3.14±0.19/−1.25±0.07`, `36.08/36.42` | **Toy `$-2.16{\scriptstyle\,\pm\,}0.00$` / `$-2.17{\scriptstyle\,\pm\,}0.01$`, paired `$-0.005$ $[-0.016,\,+0.007]$`; Synthetic `$-3.57{\scriptstyle\,\pm\,}0.00$` / `$-1.31{\scriptstyle\,\pm\,}0.00$`, paired `$+2.25$ $[+2.25,\,+2.26]$`; Coral `$36.04{\scriptstyle\,\pm\,}0.01$` / `$36.40{\scriptstyle\,\pm\,}0.01$`, paired `$+0.36$ $[+0.32,\,+0.39]$`** (plain \BO{}: Toy $-2.17$, Synthetic $-0.63$, Coral $9278$) | `paper/tables/family_effects.tex` (switch to `\input`) |
| Table 3 DCBO rows | `−6.14/−6.43`, `−3.12/−5.57`, `8.03/6.33` | **stat `$-6.12{\scriptstyle\,\pm\,}0.05$` / `$-6.40{\scriptstyle\,\pm\,}0.03$`, paired `$-0.27$ $[-0.39,\,-0.15]$`; ind `$-3.13{\scriptstyle\,\pm\,}0.06$` / `$-5.55{\scriptstyle\,\pm\,}0.13$`, paired `$-2.43$ $[-2.69,\,-2.17]$`; nonstat `$7.01{\scriptstyle\,\pm\,}0.83$` / `$5.78{\scriptstyle\,\pm\,}0.02$`, paired `$-1.23$ $[-2.98,\,+0.52]$`** (both baseline and quotient rerun with the corrected stock semantics) | same |
| Table 3 MCBO rows | `1.39±0.13/2.16±0.00`, `−5.15/−5.15` | **ToyGraph `$1.39{\scriptstyle\,\pm\,}0.13$` / `$2.16{\scriptstyle\,\pm\,}0.00$`, paired `$-0.77$ $[-1.04,\,-0.50]$`; PSAGraph `$-5.15{\scriptstyle\,\pm\,}0.00$` / `$-5.15{\scriptstyle\,\pm\,}0.00$`, paired `$+0.0002$ $[-0.0003,\,+0.0008]$`** (unchanged at displayed precision) | same |
| §5.5 "60 paired dynamic runs / 40 model-based" | counts | unchanged counts; write "…invariant in all 60 paired dynamic runs and in all 40 paired model-based runs, at the level of the logged decisions (intervention sets and levels; chosen $X$ per iteration) as well as of the values" | `verify_traces.py` |
| Table 1 | `\input{tables/minimal_taxonomy}` | regenerated: A1 0, A2 0, **A3 21.4**, B1 0; BO column 0 | `emit_minimal_taxonomy.py` |
| §5.2 new invariance sentence | — | "The no-prior \QCBONP{} is invariant under A1, A2 \emph{and} the quotient-visible A3: on this SCM the A3 edit changes the quotient prior only, so an arm without priors cannot see it." | `verify_traces.py` (QCBONP A3 check) |
| Cost table (if included) | — | regenerated: e.g. Coral \CBO{} init 800 vs \QCBO{} 100 (|ES| 31 vs 3); \HQCBO{} / \HQCBOGF{} init 6+6 | `paper/tables/cost_accounting.tex` |

## Captions — `KNOWN-NOW`

- **Fig. 2:** use "(a)/(b)" throughout (text currently mixes "(a)", "the right panel", "Top/Bottom"); append "Grey dotted: plain \BO{}, identical under every condition."
- **Fig. 3:** append "Dashed cyan: graph-free \HQCBOGF{}. Grey dotted: \BO{} on the joint arm, which shares fixed \QCBO{}'s floor. Vertical line: mean trigger."
- **Fig. 4:** "Top: CBO minimization with plain \BO{} (grey dotted) on Toy and Synthetic; on Coral \BO{}'s value ($\approx9.28\times10^{3}$) is annotated because the benchmark's intervention ranges force its joint arm far from the natural regime. Middle: … Bottom: …" — remove the red box once §5.5 numbers are refreshed from the v3 (corrected-semantics) reruns.
- **New Fig. (ablations, `figures/minimal_ablations.pdf`):** "(a) ParallelParent: fine \CBO{} with and without causal priors under A0 and A1, \BOS{} and \BO{}; (b) FrontDoor: prior corruption (B1) against the no-prior references; (c) MediatedChain: graph-informed vs graph-free refinement. Mean ± s.e. over 30 seeds."
- **New Fig. (cost, `figures/cost_indexed.pdf`, supplementary):** "Best-so-far population objective against initialization-inclusive cumulative cost (initial design at trial 0, split design at the trigger); step interpolation, mean ± s.e. over seeds that reached each budget."
- **Table 1 caption:** append "The two $\sup|\Delta|$ columns are the paired incumbent deviation of \QCBO{} and of the graph-free \BO{} baseline (zero by construction)."
- **Table 3:** switch to `\input{tables/family_effects}` (adds the paired-difference column the §5.5 text already describes).
- **New Table (`tables/minimal_ablations.tex`):** "Final value and cumulative incumbent regret of every arm per condition; last row: paired \HQCBOGF{}$-$\HQCBO{} contrast with 95\% CI."

## 6. Related work — `KNOWN-NOW`

Move the teal CMAB paragraph here and replace it with the reviewer's §12 two-paragraph block (arm = intervention set with a continuous surface vs bandit action $(A,a)$; cumulative *incumbent* regret vs executed-action regret; CAMAB transfer vs commitment to a coarse input). Keep the citations; check `leeStructuralCausalBandits` vs `lee2018structural` for duplication.

## 7. Discussion / Conclusion — `KNOWN-NOW`

Replace "Theoretical and empirical results show that using quotients can prevent catastrophic graph errors…" with the reviewer's §15 block, and add to Limitations: exact population responses and hand-selected partitions; one observational dataset per SCM; the structural zero-price conditions are not certified by the coarse graph; graph-informed refinement receives new graph information (graph-free does not, at the price of no post-split prior); the joint-cluster GP is a statistical approximation.

## Editorial — `KNOWN-NOW`

| Written | Should read |
|---|---|
| "The resulting prior hence given by" | "The resulting prior is" |
| "With $\mathbf X_C$ as the vector …" (fragment) | "Here $\mathbf X_C$ denotes the vector …" |
| "every nonmanipulable variable, to be" | "every nonmanipulable variable to be" |
| "a setting related to our" | "a related setting" |
| "QCBO aims solve an sequential" | (replaced with §6 block) |
| "like in \ref{sec:exp-a}" | "as in Section~\ref{sec:exp-a}" |
| "The MediatedChain, in Appendix …, graph has" | "The MediatedChain graph (Appendix …) has" |
| "The right panel of Figure~\ref{fig:minimal-misspec}" | "Panel (b) of Figure~\ref{fig:minimal-misspec}" |
| "Arrow up for maximization and down for minimization objective give the objective direction." | "Arrows indicate maximization ($\uparrow$) or minimization ($\downarrow$)." |
| "uninformative prior" (all occurrences) | "constant observational fallback" |
| grey duplicate intro paragraphs, red/teal markup, "MCBO HAS A PROBLEM" box | delete after integrating |

## Not to change

Protocol constants ($n_{\rm obs}=100$, batch 20, $N_{\max}=150$, seeds, trials, plateau $k=5$/$\delta=10^{-3}$, unit costs), the analytic values ($4.25$, $3.96$, $237.6$, $0.04$, $V(\Pi)=4$), the ε$_t$ schedule (division form), and every theorem statement not listed above.
