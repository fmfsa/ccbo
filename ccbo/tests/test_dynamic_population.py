from collections import OrderedDict
import numpy as np
import pytest

from ccbo.qdcbo.population import (RegularizedNonStationarySEM, regularized_graph,
                                  simulate_prefix, population_response)


def square_history():
    static = OrderedDict([('X', lambda e, t, s: e),
                          ('Y', lambda e, t, s: s['X'][t] + e)])
    dynamic = OrderedDict([('X', lambda e, t, s: s['X'][t-1]+e),
                           ('Y', lambda e, t, s: s['Y'][t-1]**2 + e)])
    return static, dynamic


def test_past_target_is_resimulated_not_clamped_to_mean():
    static, dynamic = square_history()
    history = {'X': [2., None], 'Y': [None, None]}
    mean, se, _ = population_response(static, dynamic, history, ('X',), [0.], 1,
                                     samples=100000, seed=1000)
    # E[(2+epsilon)^2]=5; clamping Y0 to its mean would incorrectly give4.
    assert abs(mean-5.) < 6*se
    assert abs(mean-4.) > .8
    history['Y'][0] = 2.
    with pytest.raises(ValueError, match='must not clamp'):
        population_response(static, dynamic, history, ('X',), [0.], 1, samples=16)


def test_local_common_random_numbers_and_response_dimensions():
    static, dynamic = square_history()
    history = {'X': [None, None], 'Y': [None, None]}
    np.random.seed(24); expected = np.random.random(5)
    np.random.seed(24)
    a = population_response(static, dynamic, history, ('X',), [0.], 1, samples=32, seed=0)
    population_response(static, dynamic, history, ('X',), [0.], 1,
                        samples=64, seed=0, purpose='independent-score-v1')
    b = population_response(static, dynamic, history, ('X',), [0.], 1, samples=32, seed=0)
    assert a == b
    assert np.array_equal(np.random.random(5), expected)
    with pytest.raises(ValueError, match='dimension'):
        population_response(static, dynamic, history, ('X',), [], 1, samples=16)


def test_regularized_companion_is_defined_at_zero_and_has_actual_crosslags():
    import networkx as nx
    sem = RegularizedNonStationarySEM(1)
    blank = {v: [None]*3 for v in ('X', 'Z', 'Y')}
    eps = {v: np.zeros((3, 4)) for v in blank}
    sample = simulate_prefix(sem.static(), sem.dynamic(), blank, 2, 4, 0, 'test', epsilon=eps)
    assert all(np.isfinite(v).all() for v in sample.values())
    assert np.array_equal(sample['X'], np.zeros((3, 4)))
    # Bounded denominator and linear-growth envelope, independent of outputs.
    grid = np.linspace(-100, 100, 1001)
    assert np.all(np.sqrt(1+grid**2) >= 1)
    assert np.all(np.abs(grid / np.sqrt(1+grid**2)) <= np.abs(grid)+1e-12)
    graph = nx.MultiDiGraph()  # actual graphviz construction has no T yet
    for t in range(3):
        graph.add_edges_from([(f'X_{t}', f'Z_{t}'), (f'Z_{t}', f'Y_{t}')])
        if t:
            graph.add_edges_from((f'{v}_{t-1}', f'{v}_{t}') for v in blank)
    corrected = regularized_graph(graph)
    assert corrected.has_edge('X_0', 'Z_1')
    assert corrected.has_edge('Z_1', 'Y_2')
    assert not corrected.has_edge('Z_0', 'Y_1')
    assert not graph.has_edge('X_0', 'Z_1')  # historical graph preserved


def test_vectorized_regularized_sem_matches_scalar_reference():
    sem = RegularizedNonStationarySEM(1)
    blank = {v: [None]*3 for v in ('X', 'Z', 'Y')}
    blank['X'][0] = 0.; blank['Z'][1] = 0.
    epsilon = {v: np.random.RandomState(10+i).normal(size=(3, 8))
               for i, v in enumerate(blank)}
    vectorized = simulate_prefix(sem.static(), sem.dynamic(), blank, 2,
                                  8, 0, 'test', epsilon=epsilon)
    for particle in range(8):
        state = {v: np.zeros(3) for v in blank}
        for t in range(3):
            for v, fn in (sem.static() if t == 0 else sem.dynamic()).items():
                state[v][t] = blank[v][t] if blank[v][t] is not None else fn(epsilon[v][t, particle], t, state)
        for v in blank:
            assert np.allclose(vectorized[v][:, particle], state[v])


def test_training_target_accepts_actual_backend_keyword_contract():
    from types import SimpleNamespace
    from ccbo.qdcbo.population import make_training_target
    static, dynamic = square_history()
    model = SimpleNamespace(true_initial_sem=static, true_sem=dynamic,
                            feedback_samples=32, population_seed=0, population_events=[])
    target = make_training_target(model, ('X',))
    history = {'X': [2., None], 'Y': [None, None]}
    value = target(current_target='Y_1', intervention_levels=np.array([[0.]]),
                   assigned_blanket=history)
    assert np.isfinite(value)
    assert len(model.population_events) == 1
    assert model.population_events[0]['x'] == [0.]
    assert model.population_events[0]['training_mean'] == value
    assert history['X'][1] is None
