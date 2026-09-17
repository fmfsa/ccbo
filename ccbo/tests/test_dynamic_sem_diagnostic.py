"""Tests for the independent diagnostic, not endorsements of the approximation."""
import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / 'scripts' / 'validate_dynamic_sem.py'
spec = importlib.util.spec_from_file_location('dynamic_diagnostic', PATH)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_temporal_double_counting_analytic_witness():
    case = diagnostic.analytic_case()
    assert case['expected_mean_bias'] == 7.5
    assert case['summed_conditional_variance'] == 5.5625
    assert case['true_conditional_variance'] == .25
    assert diagnostic.analytic_case(0, 0)['expected_mean_bias'] == 0


def fixture():
    return {'per_t': [dict(t=0, best_so_far=[1e7, 3., 2.],
                          per_trial_cost=[0., 1., 2.],
                          outcome_values=[1e7, 1e7, 3., 2.],
                          chosen_sets=[['X'], ['X', 'Z']],
                          levels_by_set={'X': [None, [[0.]], None],
                                         'X+Z': [None, None, [[1., 2.]]]} )]}


def test_event_reconstruction_keeps_sentinel_out_of_intervention_events():
    result = diagnostic.audit_payload(fixture())
    assert result['ok']
    assert result['events'][0]['eligible_recommendation'] is False
    assert result['events'][1]['y'] == 3.
    assert result['events'][2]['y'] == 2.


def test_event_reconstruction_detects_shifted_outcome_and_unknown_schedule():
    data = fixture()
    data['per_t'][0]['outcome_values'][2:] = [2., 3.]
    assert not diagnostic.audit_payload(data)['ok']
    data = fixture()
    data['per_t'][0]['chosen_sets'].pop()
    assert not diagnostic.audit_payload(data)['ok']
