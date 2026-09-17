"""Protect the distinction between outcome equality and decision equality."""
from copy import deepcopy

import pytest

from ccbo.trace_checks import compare_qmcbo_pair
from scripts.emit_qmcbo_tables import paired_identity


def log():
    return {"iterations": [{"iter": i, "X": [1.0, float(i)], "score": float(i)}
                           for i in range(2)]}


def test_same_final_is_not_same_trajectory_or_decisions():
    a, b = log(), log()
    b["iterations"][0]["X"][1] = 0.5
    result = compare_qmcbo_pair([0.0, 1.0], [0.5, 1.0], a, b, 2)
    assert result["identical_final"]
    assert not result["identical_incumbent_trajectory"]
    assert not result["identical_decisions_and_scores"]
    assert result["max_abs_intervention_deviation"] == 0.5


def test_identical_incumbents_do_not_hide_different_sampled_scores():
    a, b = log(), log()
    b["iterations"][0]["score"] = -2.0
    result = compare_qmcbo_pair([0.0, 1.0], [0.0, 1.0], a, b, 2)
    assert result["identical_incumbent_trajectory"]
    assert not result["identical_decisions_and_scores"]
    assert result["max_abs_score_deviation"] == 2.0


@pytest.mark.parametrize("mutation", ["missing", "order", "nan", "dimension"])
def test_invalid_evidence_cannot_pass_as_identity(mutation):
    a = log()
    if mutation == "missing":
        a["iterations"].pop()
    elif mutation == "order":
        a["iterations"][0]["iter"] = 1
    elif mutation == "nan":
        a["iterations"][0]["score"] = float("nan")
    else:
        a["iterations"][0]["X"].append(0.0)
    with pytest.raises(ValueError):
        compare_qmcbo_pair([0.0, 1.0], [0.0, 1.0], a, deepcopy(a), 2)


def test_missing_seed_is_not_silently_removed_from_denominator():
    unit = {"traj": [0.0, 1.0], "log": log()}
    with pytest.raises(ValueError, match="expected paired seeds"):
        paired_identity({0: unit, 1: unit}, {0: unit}, 2, 2)


def test_three_counts_are_reported_separately():
    a = {"traj": [0.0, 1.0], "log": log()}
    b = deepcopy(a)
    b["traj"][0] = 0.25
    b["log"]["iterations"][0]["X"][1] = 0.5
    result = paired_identity({0: a}, {0: b}, 1, 2)
    assert result["identical_final"] == 1
    assert result["identical_incumbent_trajectory"] == 0
    assert result["identical_decisions_and_scores"] == 0


def metadata_log(env='ToyGraph', algo='QMCBO', seed=0, trials=2, perturbed=False):
    from ccbo.trace_checks import QMCBO_E2_EDITS
    return {'schema': 1, 'unit': {'suite': 'qmcbo', 'env': env, 'algo': algo,
            'seed': seed, 'num_trials': trials,
            'misspec': QMCBO_E2_EDITS[env] if perturbed else ''},
            'iterations': [{'iter': i, 'X': [1.0, float(i)], 'score': float(i)}
                           for i in range(trials)]}


@pytest.mark.parametrize('env', ['ToyGraph', 'PSAGraph'])
def test_e1_sidecar_cannot_be_used_as_e2_evidence(env):
    from ccbo.trace_checks import validate_qmcbo_metadata
    evidence = metadata_log(env=env)
    with pytest.raises(ValueError, match='misspec'):
        validate_qmcbo_metadata(evidence, env, 'QMCBO', 0, 2, perturbed=True)
    validate_qmcbo_metadata(metadata_log(env=env, perturbed=True),
                            env, 'QMCBO', 0, 2, perturbed=True)


@pytest.mark.parametrize('field,value', [('suite', 'other'), ('env', 'PSAGraph'),
    ('algo', 'MCBO'), ('seed', 1), ('num_trials', 100)])
def test_wrong_unit_metadata_is_rejected(field, value):
    from ccbo.trace_checks import validate_qmcbo_metadata
    evidence = metadata_log()
    evidence['unit'][field] = value
    with pytest.raises(ValueError, match=field):
        validate_qmcbo_metadata(evidence, 'ToyGraph', 'QMCBO', 0, 2)


def test_wrong_schema_is_rejected():
    from ccbo.trace_checks import validate_qmcbo_metadata
    evidence = metadata_log()
    evidence['schema'] = 2
    with pytest.raises(ValueError, match='schema'):
        validate_qmcbo_metadata(evidence, 'ToyGraph', 'QMCBO', 0, 2)


def test_collector_requires_expected_graph_view(tmp_path):
    import json
    import pandas as pd
    from scripts.emit_qmcbo_tables import _collect
    path = tmp_path / 'trial_results_QMCBO_ToyGraph_0.csv'
    pd.DataFrame({'current_optimal': [0.0, 1.0]}).to_csv(path, index=False)
    path.with_suffix('.decisions.json').write_text(json.dumps(metadata_log()))
    with pytest.raises(ValueError, match='misspec'):
        _collect(tmp_path, 2, perturbed=True)


def test_verifier_final_count_does_not_require_same_earlier_incumbents(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from scripts import verify_traces as verify
    monkeypatch.setattr(verify, 'QMCBO_ENVS', ('ToyGraph',))
    monkeypatch.setattr(verify, 'QMCBO_SEEDS', 1)
    monkeypatch.setattr(verify, 'QMCBO_PUB', None)
    monkeypatch.setattr(verify, 'QMCBO_E2_IDENTICAL', {('ToyGraph', 'MCBO'): (1, 0.0)})
    monkeypatch.setattr(verify, 'FAILURES', [])
    for perturbed in (False, True):
        folder = tmp_path / 'qmcbo' / ('_e2' if perturbed else '')
        folder.mkdir(parents=True, exist_ok=True)
        for method in ('MCBO', 'QMCBO'):
            path = folder / f'trial_results_{method}_ToyGraph_0.csv'
            values = list(range(100))
            if perturbed and method == 'MCBO':
                values[0] = -1
            pd.DataFrame({'current_optimal': values}).to_csv(path, index=False)
            path.with_suffix('.decisions.json').write_text(json.dumps(
                metadata_log(algo=method, trials=100, perturbed=perturbed)))
    verify.verify_qmcbo(tmp_path)
    assert verify.FAILURES == []
    # The verifier validates MCBO sidecars too, not only QMCBO equality pairs.
    wrong = tmp_path / 'qmcbo/_e2/trial_results_MCBO_ToyGraph_0.decisions.json'
    wrong.write_text(json.dumps(metadata_log(algo='MCBO', trials=100)))
    verify.verify_qmcbo(tmp_path)
    assert any('graph view' in failure for failure in verify.FAILURES)
