"""
POMIS correctness tests against hand-derived Lee & Bareinboim 2018 ground
truth.

The POMIS characterisation (LB18): X ⊆ M is a POMIS for Y iff
``IB(G_{X̄}, Y) ∩ M = X``, where G_{X̄} removes every arrowhead into X
(incoming directed edges and bidirected edges incident to X), and
``IB(G, Y) = Pa(MUCT(G, Y)) \\ MUCT(G, Y)`` with MUCT the minimal set
containing Y closed under c-components and descendants within G[An(Y)].

Each case below documents its full hand derivation, so a failure pinpoints
where the implementation deviates from the definitions.
"""

import pytest

from ccbo.coarsening import _admg, compute_POMIS, project_out_hidden


def _pomis_set(admg, M, target='Y'):
    return set(compute_POMIS(admg, M, target))


def test_unconfounded_chain():
    """X -> Z -> Y, no confounders, M = {X, Z}.

    MUCT(G, Y) = {Y} for every mutilation (no bidirected edges), so
    IB = Pa(Y) = {Z} unless Z is intervened on. Only do(Z) satisfies
    IB = X. POMIS = {{Z}}.
    """
    g = _admg('XZY', di=[('X', 'Z'), ('Z', 'Y')])
    assert _pomis_set(g, ['X', 'Z']) == {frozenset({'Z'})}


def test_lb18_confounded_chain():
    """Z -> X -> Y with X <-> Y, M = {X, Z}  (LB18 motivating example).

    - X = {}:   MUCT = {X, Y} (c-component), IB = {Z} != {}      -> no
    - X = {X}:  do(X) removes X <-> Y, MUCT = {Y}, IB = {X} = X  -> POMIS
    - X = {Z}:  MUCT = {X, Y}, IB = {Z} = X                      -> POMIS
    - X = {X,Z}: MUCT = {Y}, IB = {X} != {X, Z}                  -> no

    {X} is only recovered if mutilation removes the bidirected edge into X;
    {Z} is the LB18 insight that intervening upstream can preserve the
    exploitable X-Y confounding.
    """
    g = _admg('ZXY', di=[('Z', 'X'), ('X', 'Y')], bi=[('X', 'Y')])
    assert _pomis_set(g, ['X', 'Z']) == {frozenset({'X'}), frozenset({'Z'})}


def test_descendant_closure_required():
    """A <-> Y, A -> B -> Y, M = {A, B}.

    MUCT(G, Y): c-component gives {A, Y}; descendant closure pulls in B
    (A -> B, B in An(Y)), so MUCT = {A, B, Y} and IB = {}.
    - X = {}:   IB = {} = X                                       -> POMIS
    - X = {A}:  do(A) removes A <-> Y; A drops out of An(Y) terms;
                MUCT = {Y}, IB = {B} != {A}                       -> no
    - X = {B}:  A -> B removed, A no longer an ancestor of Y;
                MUCT = {Y}, IB = {B} = X                          -> POMIS
    - X = {A,B}: MUCT = {Y}, IB = {B} != {A, B}                   -> no

    Without descendant closure MUCT(G, Y) = {A, Y} and IB = {B}, which
    wrongly excludes the empty set.
    """
    g = _admg('ABY', di=[('A', 'B'), ('B', 'Y')], bi=[('A', 'Y')])
    assert _pomis_set(g, ['A', 'B']) == {frozenset(), frozenset({'B'})}


def test_completegraph_pomis():
    """CompleteGraph projected ADMG (Aglietti 2020), M = {B, D, E}.

    Graph: B->C, C->D, A->E, C->E, D->Y, E->Y; bidirected A<->Y, B<->Y
    (from projecting out U1, U2). Hand derivation with M-restricted POMIS
    (IB ∩ M = X):

    - {}:      MUCT: cc(Y)={A,B,Y}; desc(A)={A,E,Y}, desc(B)={B,C,D,E,Y}
               -> MUCT = all, IB = {} = X                          -> POMIS
    - {B}:     B<->Y gone; cc(Y)={A,Y}; desc(A)={A,E,Y};
               IB = {C,D}, ∩M = {D} != {B}                         -> no
    - {D}:     C->D cut; cc(Y)={A,B,Y}; desc -> {A,B,C,E,Y};
               IB = {D} = X                                        -> POMIS
    - {E}:     A->E, C->E cut; cc(Y)={A,B,Y}; desc -> {A,B,C,D,Y};
               IB = {E} = X                                        -> POMIS
    - {B,D}:   B<->Y, C->D cut; cc(Y)={A,Y}; desc(A)={A,E,Y};
               IB = {C,D}, ∩M = {D} != {B,D}                       -> no
    - {B,E}:   cc(Y)={A,Y}; desc(A)={A} (A->E cut);
               IB = {D,E} != {B,E}                                 -> no
    - {D,E}:   cc(Y)={A,B,Y}; desc(A)={A}, desc(B)={B,C};
               IB = {D,E} = X                                      -> POMIS
    - {B,D,E}: cc(Y)={A,Y}; desc(A)={A}; IB = {D,E} != X           -> no
    """
    projected = project_out_hidden('CompleteGraph')
    got = _pomis_set(projected, ['B', 'D', 'E'])
    expected = {frozenset(), frozenset({'D'}), frozenset({'E'}),
                frozenset({'D', 'E'})}
    assert got == expected


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
