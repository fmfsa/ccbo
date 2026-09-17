"""Analytic witness for duplicated temporal conditional means."""
from ccbo.tests import dynamic_fixtures as diagnostic


def test_temporal_double_counting_analytic_witness():
    case = diagnostic.analytic_case()
    assert case['expected_mean_bias'] == 7.5
    assert case['summed_conditional_variance'] == 5.5625
    assert case['true_conditional_variance'] == .25
    assert diagnostic.analytic_case(0, 0)['expected_mean_bias'] == 0
