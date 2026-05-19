# CCBO vs. MFACBO: Positioning Memo

**Reference paper:** Zeitler, J. (2025). *Towards MFACBO: Multi-Fidelity Abstraction Causal Bayesian Optimization in the Context of the Abstraction-Fidelity Connection.* Accepted at the 1st Workshop on Causal Abstractions and Representations (CAR) at UAI 2025.

## 1. What MFACBO proposes

MFACBO is a **position paper** that proposes integrating Causal Bayesian Optimization (CBO, Aglietti 2020) with Multi-Information-Source / Multi-Fidelity BO (MISO, Poloczek 2017). The unifying idea is *causal abstraction* (Beckers 2019): the low-fidelity measurement `Y_low` is a **coarsening of the outcome `Y_high`** rather than a separate physical signal, with model discrepancy `δ_l(x) = f(l, x) − g(x)` playing the role of the abstraction-mapping distance `d_v`.

Concrete formulation:
- Surrogate: GP prior on `f` with `f(l, x) = g(x) + δ_l(x)` for `l > 0` (Poloczek-style MISO).
- Acquisition: cost-normalised expected gain (`MKG` — modified knowledge-gradient), selecting both a design `x` and a fidelity level `l`.
- Empirical plan: synthetic toy from Eq. 9 (a CBO-style 3-node DAG `X → Z → Y_high → Y_low`) plus wind-tunnel Causal Chambers as a real testbed.

**Crucially, the paper presents no experiments yet** — Sections 3 and 5 are sketches of a method and experimental design rather than results.

## 2. What CCBO does

CCBO **coarsens the input side of the DAG**, not the outcome. Manipulable variables are partitioned into clusters, a C-DAG is built (Anand 2023; Lee 2019 latent projection), and causal effect identification (back-door / front-door / g-computation) is carried out at the cluster level. This C-DAG do-effect is used as a GP prior; the BO loop then refines it from data. RCCBO additionally **discovers** the partition online via the RePaRe structure-discovery routine.

## 3. Where the two meet

Both can be cast as instances of Beckers's approximate causal abstraction:

| Concept             | MFACBO                              | CCBO                                |
|---------------------|-------------------------------------|-------------------------------------|
| What is abstracted? | Outcome `Y_high → Y_low`            | Manipulable inputs `(X_1,…,X_n) → clusters` |
| Abstraction map `τ` | `Y_high ↦ Y_low` (coarse measurement) | Variable-to-cluster surjection `π` on inputs |
| Distance `d_v`      | `δ_l(x) = f(l,x) − f(0,x)`          | C-DAG identification gap (KL between cluster-level and fine-DAG do-effect) |
| Cost saving         | Cheaper *measurements* of `Y`       | Smaller *search space* (cluster-level interventions, identifiable via C-DAG) |

Both methods accept a noisy / approximate abstraction and let BO recover the true objective from data; both use the same GP machinery. They are not competitors — they address orthogonal axes of "what to coarsen".

## 4. Where they differ

1. **Required infrastructure.** MFACBO needs ≥ 2 measurement fidelities of the same outcome (an MISO setting). CCBO needs only one fidelity but a structural model over manipulable variables.
2. **Identification theory.** MFACBO's `Y_low` is just a noisy proxy; identification is not part of its theory. CCBO's C-DAG identification (Anand do-calculus, Lee latent projection) is a hard correctness criterion — coarsenings that are not identifiable are rejected or fall back to an uninformative GP prior.
3. **Online discovery.** MFACBO assumes the abstraction map is given. RCCBO **discovers** the partition from interventional data (via RePaRe), an angle MFACBO calls out as future work ("MFACBO emerges as a data-driven method to learn approximate causal abstraction mappings", Abstract).
4. **Causal target.** MFACBO's `do(X)` is an intervention on the original fine-grained variables; the abstraction is applied only to the readout `Y`. CCBO's `do(X̄)` is a **cluster-level** intervention with a real semantic gap to a fine-grained `do(X)` — flagged in `theory_cdag_identification.md` and the paper; in practice CBO still recovers the truth from data.

## 5. Hybrid sketch — MFACBO + CCBO

A natural composition: coarsen the **inputs** (CCBO partition) **and** the **outcome** (MFACBO fidelities). The surrogate becomes
```
f(l, x̄) = g(x̄) + δ_l(x̄),
   g ~ GP with C-DAG identification prior
   δ_l ~ GP, l > 0
```
where `x̄` is a cluster-level intervention. Acquisition would select `(cluster-ES, intervention value, fidelity)`. This is not built here, but the existing code already separates (i) CCBO partition discovery from (ii) BO acquisition cleanly enough that an MFBO multi-output GP could be dropped in at the surrogate layer.

## 6. Drop-in related-work paragraph for `ccbo_paper.tex`

```latex
\paragraph{Multi-fidelity and abstraction-based BO.}
Closest in spirit to ours is the concurrent position paper of
\citet{zeitler2025mfacbo}, which proposes MFACBO --- a hybrid of CBO and
Multi-Information-Source BO~\citep{poloczek2017multi} unified through
\citet{beckers2019approximate} causal abstraction. MFACBO coarsens the
\emph{outcome} (\(Y_{\text{low}}\) as a cheap, biased proxy of
\(Y_{\text{high}}\)); CCBO is complementary in that it coarsens the
\emph{manipulable inputs} into cluster-level interventions and exploits
identification on the resulting C-DAG~\citep{anand2023causal,lee2019latent}.
The two axes are orthogonal: one could combine an MFACBO multi-output GP
over fidelities with a CCBO C-DAG prior over cluster-level interventions,
recovering both cost-saving mechanisms in a single surrogate.
We leave this hybrid to future work.
```

BibTeX entry:
```bibtex
@inproceedings{zeitler2025mfacbo,
  title  = {Towards {MFACBO}: Multi-Fidelity Abstraction Causal {B}ayesian Optimization in the Context of the Abstraction-Fidelity Connection},
  author = {Zeitler, Jakob},
  booktitle = {1st Workshop on Causal Abstractions and Representations (CAR) at UAI},
  year   = {2025},
}
```
