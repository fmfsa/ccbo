"""The nonempty MIS convention need not preserve all-nonempty optimum.

Scientific regression for Lemma 1's scope; no Monte Carlo or optimizer run.
"""
from fractions import Fraction

from ccbo.coarsening import compute_MIS


def test_nonempty_mis_can_lose_null_equivalent_nonempty_action():
    graph = {'vertices': {'A', 'Z', 'Y'}, 'di': {('A', 'Y')}, 'bi': set()}
    mis = compute_MIS(graph, {'A', 'Z'}, 'Y')
    assert mis == [frozenset({'A'})]
    # A=.1 U, U~N(0,1), Y=A², independent Z; D_A=[1,2].
    # Fixing A (with or without Z) minimizes a² at a=1. Fixing Z alone
    # leaves E[A²]=.1². Use exact arithmetic to expose the strict gap.
    best = {frozenset({'A'}): Fraction(1),
            frozenset({'Z'}): Fraction(1, 100),
            frozenset({'A', 'Z'}): Fraction(1)}
    null = Fraction(1, 100)
    nonempty_value = min(best.values())
    mis_value = min(best[arm] for arm in mis)
    assert (nonempty_value, mis_value) == (Fraction(1, 100), Fraction(1))
    assert nonempty_value < mis_value
    assert min(null, nonempty_value) == min(null, mis_value)
