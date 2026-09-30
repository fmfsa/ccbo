# Third-party code notice

This directory is adapted from the reference implementation of Causal Bayesian
Optimization by Aglietti et al. (2020):

- Repository: https://github.com/VirgiAgl/CausalBayesianOptimization
- Paper: V. Aglietti, X. Lu, A. Paleyes and J. González. Causal Bayesian
  Optimization. AISTATS 2020.

The upstream repository does not include a license. The code here is used for
non-commercial research with attribution. The MIT license in the repository
root does not cover this directory.

Changes made for this project include the MinimalBench graphs
(`graphs/ParallelParent*.py`, `graphs/FrontDoor*.py`,
`graphs/MediatedChain*.py`), a corrected minimal-intervention-set list for
`ToyGraph`, and modifications that let the optimizer run on quotient graphs
under the experimental protocol described in the paper.
