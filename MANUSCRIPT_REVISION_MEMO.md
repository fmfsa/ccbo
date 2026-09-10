# QCBO manuscript — post-rerun revision memo

Date 2026-09-07. Ground truth for the paper is the source you pasted (the `aistats2026` file, not `paper/qcbo_aistats.tex` in the repo). Ground truth for every number is `results/v2` on the DTU cluster, archived on branch `rerun/online-obs-v2` (PR #4). Nothing in the manuscript has been edited; this memo lists what to change and gives the replacement text.

## 0. TL;DR

- The online-observation protocol in the vendored CBO loop was broken in three places (observe trials re-revealed the same 20 rows, never refreshed the prior, and hit stale caches). It was fixed at commit `d23022e` and everything it touched was rerun: 660 MinimalBench units and 90 CBO-family units. QDCBO, QMCBO and CEO use separate engines and were carried over unchanged from verified archives.
- **The paper's central claim survives.** Under A1, A2 and B1 the paired QCBO runs are exactly identical on all 30 seeds (sup|Δ| = 0, same arms, same values); the quotient-visible control A3 diverges (21.56). The claim is now non-vacuous: observations genuinely refresh the prior and invariance still holds.
- Headline numbers moved slightly (Table below). Every number in §5.2–§5.4 and the three CBO rows of Table 3 are stale. Nothing changes sign except the Toy paired effect, which stays a statistical tie.
- A plain-BO baseline arm was added (runbook §4). It is not yet mentioned anywhere in the manuscript, and the regenerated Table 1 needs `\newcommand{\BO}{\textsc{BO}}` or it will not compile.
- The manuscript does not describe the observe/intervene protocol at all (only "receives its first 100 observations"). After the fix this is the one protocol paragraph the paper cannot ship without; draft text is in §3.3.
- Figures 2, 3 and 4 are regenerated (BO added to all three; Coral annotated; trigger annotation fixed). The MCBO red box remains open — see §3.8.

## 1. What we did

1. **Diagnosis** (in `RERUN_ONLINE_OBS.md`): `observe()` sliced a constant window so every observe returned rows 100–120; `CoarsenedGraph.refit_models` never wrote the new frame back nor cleared the memoized do-estimators; the point-wise prior caches were keyed by `str(x)` and never invalidated. Net effect: observe trials consumed a trial slot and moved ε_t but were no-ops on the prior. HQCBO additionally lost phase-1 observations at the refinement boundary.
2. **Fix + tests** (`d23022e`): cursor-based pool with budget cap, prior refresh on every observe, cache clearing, resume/phase-boundary persistence. New tests: `test_observation_pool`, `test_observation_policy`, `test_prior_refresh_on_observe`, `test_resume_equivalence`. The scientific guard `test_minimal_invariance` is in the slow suite.
3. **Rerun** on the DTU HPC (LSF `hpc`, 16 cores per job, BLAS pinned to 1 thread): minimal 660/660 OK (job 29163301), family 90/90 OK (job 29163302); zero unit failures; fresh `results/v2/` directories so no stale CSV could be silently reused.
4. **Regeneration**: Table 1, Table 3 rows, Figs 2–4, CEO table, cost table, all summary JSONs; archives repacked deterministically with per-file SHA-256 lists; `verify_traces.py` updated to the 660/90 grids with BO checks added (none weakened).
5. **Review round** (your Codex pass): caught that `plot_family_suite.py` never drew BO. Fixed; Coral BO analysed (§4 of `RERUN_ONLINE_OBS_REPORT.md`): BO's joint arm must clamp T∈[2300,2400] and D∈[2000,2080] and 9274 is the population floor of that box, which BO attains.
6. **Today**: full re-verification (below) and BO added to Figs 2–3 as well.

## 2. Verification pass — is everything correct and exact?

| Claim | How checked today | Result |
|---|---|---|
| Fast test suite on branch head | `pytest -q` | **106 passed, 11 skipped, 14 deselected** (identical to 2026-08-21) |
| Slow suite incl. `test_minimal_invariance` | run 2026-08-21 | **14 passed** |
| Grid completeness | dry-run vs. CSV count | 660/660 minimal, 90/90 family, no `FAIL` lines |
| Protected pairs A1, A2, B1 | paired arm/x/incumbent comparison, 30 seeds | sup|Δ| = 0.0, decisions equal, all three |
| Negative control A3 | same | sup|Δ| = 21.5646, decisions differ |
| BO harness leak | byte-identity of BO CSVs across A0–A3 and B0–B1 per seed | none — identical |
| Observe budget | observe rows per unit | max 3 everywhere (⌈50/20⌉); BO always 0; 95–97 % of causal minimal units exhaust the budget |
| Archives | sha256 of every file + fresh-extraction re-verify | all pass; `verify_traces.py`: **61 checks, 0 failures** (re-run today) |
| ε_t footnote values | recomputed on ParallelParent at N_t = 100 | coverage ratio 0.0760 → division form **0.114**, product form **0.051** |
| Family protocol as written (10 init pts, 40 trials, n_obs = 100) | runner defaults | matches |
| "QDCBO invariant in all 60 paired runs; QMCBO in all 40" | verify gate on archives | 60/60 and 20/20 + 20/20 — holds |
| Coral BO value | MC evaluation of the SEM over BO's joint box | box floor ≈ 9273.9 (corner C=0.3, D=2000, N=−2, O=3, T=2300); BO min 9274.2, mean 9279.5 — BO is optimal within its feasible set |

Protocol facts the new text relies on, read from `ccbo/cbo/cbo.py` and `CBO_functions.py` today: pool of 5 000 rows per SCM; optimizer starts on the first 100; batch 20; `N_max = 150` (runners pass `initial + 50`); ε_t = clip{[Vol C(D_t^O)/Vol D(X)] / (N_t/N_max), 0, 1}; the first trial of a fresh run/phase is a forced observation and the second a forced intervention (vendored convention); an exhausted pool always intervenes; every observation refits the identified functionals and rebuilds every arm GP; HQCBO's refined phase enters with one forced observation when budget remains.

Causal prior as implemented (`BO_functions.update_BO_models`, `causal_kernels.CausalRBF`): `f_A ~ GP(m_A, k_A)` with `k_A(a,a') = σ_f² exp(−‖a−a'‖²/2ℓ²) + σ_A(a) σ_A(a')`, `σ_A(a)² = V̂ar[Y | do(A=a)]` from the same do-calculus estimator as the mean; `σ_f² = ℓ = 1` initially, noise 10⁻¹⁰, hyperparameters re-optimised by marginal likelihood after each update; the uninformative tier uses `m = Ȳ_obs` and `σ_A² = s²_obs(Y)`.

**Two inconsistencies found in the manuscript itself** (not in the code): (i) Fig 2 is referenced as "(a)" in §5.2, "the right panel" in §5.3 and "Top/Bottom" in its caption — the PDF is stacked with panel titles "(a)"/"(b)"; (ii) §5.5 says "the paired difference is quotient minus base …" but Table 3 has no paired-difference column.

**One theory flag** (for the "not ready for review" parts): Definition 1 (full intervention support) asks that every attainable natural value of `A^Π∖A` lie in `D_B(a)`. With Gaussian disturbances the natural support is unbounded, so no bounded domain satisfies it literally — ParallelParent included. The proof of Prop. `free` only uses `inf_{b∈D_B(a)} E[Y(a,b)] ≤ inf_{b∈supp B(a)} E[Y(a,b)]`; restating Definition 1 in that form (or "with probability one" plus truncated disturbances) makes the corollary apply to the suite as run.

## 3. Text changes — written vs. should read

### 3.1 Numbers (mechanical, all exact from `results/v2`)

| Location | Written | Should read |
|---|---|---|
| §5.2 ¶2 | `correct-graph CBO reaches $0.0016\pm0.0002$` | `correct-graph CBO reaches $0.0014\pm0.0003$` |
| §5.2 ¶2 | `cumulative incumbent regrets are $4.21\pm0.37$ and $216.23\pm1.12$` | `… are $4.17\pm0.37$ and $216.16\pm1.12$` |
| §5.2 ¶3 | `maximum paired incumbent deviation $21.45$` | `… deviation $21.56$` |
| §5.3 ¶2 | `The paired increase in cumulative regret is $5.91\pm1.04$` | `… is $6.18\pm1.06$` |
| §5.3 ¶2 | `$0.0003\pm0.0003$ under the correct graph and $0.0035\pm0.0022$ under the corrupted graph` | `$0.0001\pm0.0001$ … and $0.0019\pm0.0007$ …` |
| §5.4 ¶4 | `cumulative regrets $5.46\pm1.61$, $243.23\pm0.90$, and $36.97\pm1.18$` | `… $5.98\pm1.88$, $243.19\pm0.90$, and $36.76\pm1.23$` |
| §5.4 ¶4 | `HQCBO refines at trial $7.4\pm0.1$` | `… at trial $7.3\pm0.1$` |
| Table 3, Toy | `$-2.16 \pm 0.00$ & $-2.13 \pm 0.03$` | `$-2.16 \pm 0.00$ & $-2.17 \pm 0.01$` |
| Table 3, Synthetic | `$-3.14 \pm 0.19$ & $-1.25 \pm 0.07$` | `$-3.47 \pm 0.04$ & $-1.28 \pm 0.03$` |
| Table 3, Coral | `$36.08 \pm 0.02$ & $36.42 \pm 0.01$` | `$36.07 \pm 0.02$ & $36.45 \pm 0.02$` |

Unchanged and correct as written: `$4.2500\pm0.0000$` (misspecified CBO), the MediatedChain finals `$0.0400$, $4.0000$, $0.0400$`, "all $30$ declared splits are accepted", the floor comparison `60×3.96=237.6`, all DCBO/MCBO rows, and the `60`/`40` paired-run counts in §5.5.

### 3.2 Compile blocker

Add to the preamble: `\newcommand{\BO}{\textsc{BO}}`. The regenerated `tables/minimal_taxonomy.tex` uses it in its header.

### 3.3 §5.1 — the protocol paragraph the paper is missing

Written (end of ¶2): `Each SCM provides one shared observational dataset; the optimizer receives its first $n_{\mathrm{obs}}=100$ observations.`

Should read (replace that sentence with a short paragraph):

> **Observation–intervention protocol.** Following the released CBO loop, every trial is either an observation or an intervention. Each SCM provides one fixed observational pool; the optimizer starts from its first $n_{\mathrm{obs}}=100$ rows and may acquire at most $50$ more, in batches of $20$ previously unseen rows (the final batch is truncated), so $N_{\max}=150$. At trial $t$ it observes with probability
> $\epsilon_t=\operatorname{clip}\!\big[\tfrac{\mathrm{Vol}\,C(\mathcal D^O_t)/\mathrm{Vol}\,D(\mathbf X)}{N_t/N_{\max}},\,0,\,1\big]$,
> where $C(\mathcal D^O_t)$ is the convex hull of the observed manipulable values and $N_t$ the current number of observational rows;\footnote{The released implementation divides by $N_t/N_{\max}$, whereas the CBO paper prints a product. We keep the released schedule; on \textsc{ParallelParent} at $N_t=100$ the two give $\epsilon_t\approx0.11$ and $0.05$.} the first trial of a run is a forced observation and the second a forced intervention, and once $N_t=N_{\max}$ the optimizer always intervenes. Every observation refits the identified functionals on the enlarged dataset and rebuilds every arm's GP. Observation trials count against $T$ but incur no intervention cost. The same protocol governs the CBO family; QDCBO and QMCBO follow their base methods' loops.

### 3.4 §5.1 — the BO baseline paragraph (new)

Add after the protocol paragraph:

> **Non-causal reference.** On Experiments A–C and the CBO family we also run plain BO: one GP over the joint intervention on all manipulable variables, an uninformative prior, the same cost-normalised EI, and no observation actions. BO reads no graph, so its trajectory is identical across the misspecification conditions of an SCM — the \BO{} column of Table~\ref{tab:minimal-taxonomy} is zero by construction and doubles as a harness check — and under unit costs it pays $|\mathbf X|$ per trial. It ends at $0.0017\pm0.0017$ on \textsc{ParallelParent} (the joint arm is the optimal arm), $0.60\pm0.17$ on \textsc{FrontDoor} (no prior to locate the narrow well) and $4.003\pm0.003$ on \textsc{MediatedChain} (the joint arm carries the same $\Delta(\Pi)=3.96$ floor as fixed \QCBO{}); on the family suite $-2.17\pm0.00$ (Toy), $-0.63\pm0.00$ (Synthetic) and $9279\pm2$ (Coral). The Coral value is not an optimisation failure: the benchmark's declared ranges put $T\in[2300,2400]$ and $D\in[2000,2080]$ far from their natural regime ($T\approx4$–$8$, $D\approx3$–$7$), a causal arm can intervene on $N$ alone and leave them untouched, whereas BO's joint arm must clamp both; $9274$ is the population floor of that box and BO reaches it. We annotate it in Figure~\ref{fig:family-suite} rather than plot it.

### 3.5 Figure and table captions

| Item | Written | Should read |
|---|---|---|
| Fig 2 refs | §5.2 "Figure~\ref{fig:minimal-misspec}(a)"; §5.3 "The right panel of Figure~\ref{fig:minimal-misspec}"; caption "Top: … Bottom: …" | Use "(a)"/"(b)" everywhere: §5.3 → "Panel (b) of Figure~\ref{fig:minimal-misspec}"; caption → "(a) deleting $X_1\to Y$ … (b) corruption of an identified prior …". Append to caption: "Grey dotted: plain BO, identical under every condition." |
| Fig 3 caption | as is | append "Grey dotted: plain BO on the joint arm, which shares fixed \QCBO{}'s floor. Vertical line: mean refinement trigger ($7.3$)." |
| Fig 4 caption | "Top: CBO minimization. Middle: … Bottom: …" + red box | "Top: CBO minimization, with plain BO (grey dotted) on Toy and Synthetic; on Coral BO's value ($\approx9.28\times10^{3}$) is annotated because it lies three orders of magnitude above the causal methods (§\ref{sec:exp-design}). Middle: … Bottom: …" (red box: see §3.8) |
| Table 1 caption | "Controlled misspecifications and measured paired trajectory deviations. Rows whose quotient is unchanged are predicted invariant by Proposition~\ref{prop:contract}." | append: "The two $\sup|\Delta|$ columns are the paired incumbent deviation of \QCBO{} and of the graph-free \BO{} baseline; the latter is zero by construction." |
| Table 3 + §5.5 ¶2 | text describes a paired-difference column the table lacks | either drop the sentence "For $\downarrow$ rows the paired difference is …" or replace the hand-coded tabular with `\input{tables/family_effects}` (4 columns incl. paired difference with 95 % CI: Toy $-0.005$ $[-0.016, +0.007]$; Synthetic $+2.19$ $[+2.06, +2.31]$; Coral $+0.38$ $[+0.33, +0.44]$; DCBO $-0.29$, $-2.45$, $-1.69$; MCBO $-0.77$, $+0.0001$). The `\input` also removes the hand-drift that produced §3.1. |

### 3.6 §5.5 prose about the family result

Written: nothing qualitative about Toy/Synthetic/Coral beyond the table. If a sentence is added, the exact reading of the new numbers is: Toy is a tie (paired $-0.005$, CI spans 0; 7/0/3 seeds worse/tie/better), Synthetic and Coral favour fine CBO ($+2.19$, $+0.38$), and BO is competitive only where the joint arm is a good arm (Toy).

### 3.7 §2.2 red TODO ("Put a simple sentence explaining the estimated causal value")

Covered by §4.2 below (complete prior). Minimal version of the sentence: "Here $\widehat{\E}_{\widehat{\Gg}}[Y\mid\doo(A=a)]$ is the plug-in estimate of the identified functional — for example the backdoor adjustment $\int\widehat{\E}[Y\mid A=a,Z=z]\,\mathrm d\widehat P(z)$ — computed from the observational data alone."

### 3.8 The MCBO red box on Fig 4

Not resolved by this rerun, on purpose. The ToyGraph MCBO plateau is real in the archived traces (17/20 seeds never improve; the other three stop by trial 4–6). The QMCBO engine was out of scope, and the CSVs alone cannot separate optimizer stagnation from the logger starting inside BO (recorded as the open instrumented-run item in `RUN_TODO.md`). Fig 4's bottom row is not publishable until that diagnostic is run. This is the one remaining experiment with a hard dependency.

## 4. The six discussion points

### 4.1 "After 2.2, CBO step by step for a generic reader"

Add after Eq. (5) / the estimator sentence, replacing nothing:

> **One CBO run, step by step.** Given $\widehat{\Gg}$, target $Y$, manipulable $\mathbf X$, observational data $\mathcal D^O$, domains and costs: (1) *Arms.* Compute $\ES(\widehat{\Gg})=\mathrm{MIS}(\widehat{\Gg})$. (2) *Priors.* For each arm run identification on $\widehat{\Gg}$; if identified, estimate $m_A(a)$ and $\sigma_A(a)^2$ from $\mathcal D^O$ (Appendix~\ref{app:estimators}), else use the observational mean and variance of $Y$. (3) *Surrogates.* Place the GP of Eq.~\eqref{eq:cbo-gp} on each arm and condition it on the arm's initial interventional points. (4) *Observe or intervene.* Draw $u\sim U[0,1)$; if $u<\epsilon_t$ and observational budget remains, reveal a fresh batch of observational rows and return to (2). (5) *Intervene.* Choose $(A,a)$ maximising $\mathrm{EI}_A(a)/c(A,a)$, execute $\doo(A=a)$, append the outcome to that arm's data, update its GP and the incumbent. (6) Repeat until $T$ trials are spent.

This also houses the protocol of §3.3, so the experiments section can refer back to it.

### 4.2 "Describe the complete causal prior, not only the mean"

Written (Eq. 4 + text): `f_A\sim\mathcal{GP}(m_A^{\widehat{\Gg}},k_A)` with $k_A$ left generic.

Should read:

> $f_A\sim\mathcal{GP}\!\big(m_A^{\widehat{\Gg}},\,k_A^{\widehat{\Gg}}\big),\qquad k_A^{\widehat{\Gg}}(a,a')=\sigma_f^2\exp\!\Big(-\tfrac{\|a-a'\|^2}{2\ell^2}\Big)+\sigma_A^{\widehat{\Gg}}(a)\,\sigma_A^{\widehat{\Gg}}(a'),$
> where $m_A^{\widehat{\Gg}}(a)=\widehat{\E}_{\widehat{\Gg}}[Y\mid\doo(A=a)]$ and $\sigma_A^{\widehat{\Gg}}(a)^2=\widehat{\mathrm{Var}}_{\widehat{\Gg}}[Y\mid\doo(A=a)]$ are the plug-in estimates of the identified mean and variance functionals \citep{agliettiCausalBayesianOptimization2020b}; the rank-one term inflates prior uncertainty where the observational estimate itself is uncertain. Non-identified arms use $m_A=\bar Y_{\mathrm{obs}}$ and $\sigma_A^2=s^2_{\mathrm{obs}}(Y)$. Hyperparameters $(\sigma_f^2,\ell)$ are initialised at $1$ and refit by marginal likelihood after every update; observation noise is fixed at $10^{-10}$ because interventional outcomes are exact population expectations.

Then Eq. (10) (two-tier prior) should state the pair $(m_A^\Pi,\sigma_A^\Pi)$ rather than the mean alone, and the "supplied graph affects CBO through two objects" map becomes $(\ES,\{(m_A,\sigma_A)\}_A)$.

### 4.3 "Do we need non-manipulable variables as singletons? What if they were clusters?" — open it in §3.1, discuss in the appendix

§3.1 sentence to add after "Hence a cluster that contains a manipulable variable contains only manipulable variables": "Coarsening the non-manipulable variables is possible and enlarges the protected class; Appendix~\ref{app:nonmanip} discusses what it costs."

Appendix draft:

> **Coarsening non-manipulable variables.** Let $\Pi$ additionally merge non-manipulable variables. Three things change and one does not. (i) *Action space — unchanged.* $\mathcal U(\Pi)$ depends only on the manipulable clusters, so $V(\Pi)$, $\Delta(\Pi)$ and Propositions~\ref{prop:price}–\ref{prop:floor} are untouched. (ii) *Protected class — larger.* Every within-cluster edit among non-manipulable variables becomes quotient-invisible, so $E_\Pi(\Gg)$ grows and Proposition~\ref{prop:contract} protects against strictly more errors. (iii) *Priors — weaker and costlier.* Identification now runs on a coarser C-DAG: adjustment sets are unions of clusters, so an effect identified at the fine level can become non-identified (an arm drops to the observational-mean tier), and the functionals that remain identified integrate over joint cluster distributions, raising the dimension of the plug-in estimators of Appendix~\ref{app:estimators}. Merging a confounder into a cluster with its children can also hide a bidirected edge inside the cluster and silently flip an arm's tier. (iv) *Validity — must be re-checked.* Merging a mediator chain across a directed path can create a cycle in the quotient (Proposition~\ref{prop:refinement-validity} in reverse). The singleton choice keeps the estimator boundary of Assumption~1 at the variable level, where the supported functionals are standard; it trades some robustness for concrete priors. Exploration sets can also move: a non-manipulable cluster on a directed path changes which manipulable clusters are ancestors of $C_Y$ in $\widehat{\Gg}^\Pi_{\bar S}$ only if the merge removes or adds a quotient path, which the validity check catches.

### 4.4 "Fig 2(b) as the zero-price / full-support example, then a figure for cluster-contained confounding"

The suite already instantiates the theory more precisely than "Fig 2(b)". Exact accounting (closed forms in §5.2–5.4):

| SCM | Full support (Def. 1) | Cluster-contained confounding (Def. 2) | $\Delta(\Pi)$ | Role |
|---|---|---|---|---|
| ParallelParent (Fig 2a) | yes up to Gaussian tails (interior minimiser $x_2=-2\in[-3,3]$) | **yes** — after $\doo(X_1,X_2)$ the only exogenous ancestor of $Y$ is $\varepsilon_Y$ | 0 | both conditions hold → Cor. 1 applies; QCBO reaches $0.0021\approx y^\star$ |
| FrontDoor (Fig 2b) | yes up to tails | **no** — $U$ is an exogenous ancestor of both $X_1$ (in $A^\Pi\setminus A$ for $A=\{M\}$) and $Y$ | 0 | conditions fail yet price is zero ($U$ enters $Y$ additively): the "sufficient, not necessary" example |
| MediatedChain (Fig 3) | **no** — natural $X_2=4+\varepsilon_2\notin[-2,2]$ | yes | 3.96 | full-support violation; the price is exactly the clamp |
| *proposed* Exp. D | yes | **no** | > 0 | the missing case: confounding violation with support intact |

So Fig 2(b) is better used as the sufficient-not-necessary remark, and Fig 2(a) as the zero-price exemplar. For the missing panel, a minimal SCM with $\{X_1,X_2\}$ clustered:
$U=\varepsilon_0,\; X_1=0.3\varepsilon_1,\; X_2=U+0.3\varepsilon_2,\; Y=(X_1-2)^2-X_2U+0.1\varepsilon_Y$.
Under $\doo(X_1=2)$ alone, $\E[Y]=-\E[U^2]=-1$; under any $\doo(X_1,X_2=x_2)$, $\E[Y]=(x_1-2)^2-x_2\E[U]\ge0$. Hence $V(\Pi)=0$, $y^\star=-1$, $\Delta(\Pi)=1$ with every support satisfied; the quotient sees $C\leftrightarrow Y$ through $U$. It is a MinimalBench unit (same runner, closed-form oracle) and I can implement and run it in an afternoon; the 30-seed CBO/QCBO/HQCBO/BO comparison would give the figure. Please sanity-check the closed form before I build it.

### 4.5 "Exploring MAB comparisons"

The teal CMAB paragraph you added sits in §2.2 but is related work; move it to §6 and end it with the connection to this paper: the BO baseline already plays the structure-free role, and a discrete-arm causal bandit (UCB over the MIS arms with discretised domains, à la structural causal bandits) would isolate continuous-domain surrogate modelling from graph knowledge. I would offer that as an optional appendix experiment, not a main-text one — it is cheap on the minimal suite if a reviewer asks. Also check `leeStructuralCausalBandits` vs `lee2018structural`: they look like two keys for the same paper.

### 4.6 "In HQCBO: can we start surrogates for what's abstracted without causal information?"

Yes, and it strengthens the guarantee. Today the refined phase (a) prunes the exposed sub-arms with the refined quotient's MIS rule and (b) gives them C-DAG priors from the refined quotient (that is why Corollary `adaptive-invariance` needs quotient agreement at *every* visited partition). Proposition `qexposure` already says the coarse inputs cannot identify sub-cluster effects, so (b) necessarily re-imports fine-graph information. Two structure-free variants: **priors-free** — exposed arms start on the observational-mean tier (removes the prior dependence, keeps MIS pruning); **fully graph-free** — expose all $2^k-1$ sub-arms of the split cluster on the uninformative tier (the refined phase then depends on nothing but $\Pi_j$, and the protected class of HQCBO equals that of $\Pi_0$). The price is a slower post-trigger phase and, for the second variant, more arms. On MediatedChain the split cluster has $k=2$, so both variants are a 30-unit run; the current HQCBO reaches $0.040$ within a few trials of the trigger, so the ablation would show directly whether the convenience of causal priors buys anything at $T=60$. Worth a remark in §3.3 either way; I can run the ablation.

### 4.7 Joel — "move all SCM background to §2"

Move the *formalism*, keep the *datasets*: the semi-Markovian SCM definition (currently a paragraph before Theorem 1 in §4.3) and the NPSEM sentence inside Definition 2 belong in §2.1 next to the latent-projection paragraph; the three dataset SEMs (Eqs. 17–19) stay in §5 because they are specifications.

## 5. Figures changed (all regenerated from `results/v2`, committed on the branch)

| File | What changed | What to look for |
|---|---|---|
| `figures/minimal_misspec.pdf` (Fig 2) | new traces; **BO added** to (a) and (b) as grey dotted | QCBO A1/A2 lie exactly on A0; A3 departs; BO flat near 0 in (a), plateaus ≈0.6 in (b) |
| `figures/minimal_refine.pdf` (Fig 3) | new traces; trigger annotation now `7.3` (was rounded to `7`); **BO added** | BO sits on fixed-QCBO's floor $V(\Pi)=4$ |
| `figures/family_suite.pdf` (Fig 4) | top row re-drawn; **BO added** on Toy/Synthetic; Coral annotated "BO ≈ 9.28×10³ (off-axis)"; middle/bottom rows byte-identical inputs | Toy BO ≈ CBO ≈ QCBO; Synthetic BO mid-field |
| `tables/minimal_taxonomy.tex` (Table 1) | A3 21.5 → 21.6; new BO column, all 0 | needs `\BO` macro |
| `tables/family_effects.tex` | three CBO rows updated, paired-difference column | optional `\input` replacement for Table 3 |
| `tables/ceo_minimal.tex`, `tables/cost_accounting.tex` | CBO/QCBO reference columns refreshed | not used by this source; available if the CEO comparison returns |

## 6. Where things are, and what I can run next

- Branch `rerun/online-obs-v2` (PR #4, unmerged): validated results, archives, verification gate, `RERUN_ONLINE_OBS_REPORT.md`, this memo, and the figure changes above. `main` is untouched.
- Offers, in order of value: (1) the instrumented MCBO ToyGraph diagnostic that unblocks Fig 4's red box; (2) the Exp. D unit of §4.4; (3) the HQCBO prior ablation of §4.6; (4) the discrete causal-bandit baseline of §4.5.
