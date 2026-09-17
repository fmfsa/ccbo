#!/usr/bin/env python3
"""Bounded scientific diagnostic for DCBO's temporal SEM approximation.

No optimization or GP fitting is performed. The real stock/Q fitting dispatch
is exercised with exact linear projection models on a 64-point moment-exact
cubature rule. Real SEM builders and samplers then evaluate a fixed common
blanket. This isolates emission-plus-transition composition from fit error.

Run in the pinned DCBO environment:
  PYTHONPATH=. python scripts/validate_dynamic_sem.py --output work/dynamic.json
Add --require-accurate to exit 2 when the temporal mean/variance is inaccurate.
--skip-backends --archive-dir DIR audits existing sidecars/CSVs without DCBO.
Existing source modules and artifacts are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def analytic_case(intercept=3.0, root_mean=1.0):
    """X_t=mu+eX_t; Y0=c+2X0+.5eY0; Y1=c+2X1+.5Y0+.5eY1.

    All e independent N(0,1); Z_t=eZ_t is an irrelevant manipulable root.
    The conditional mean and variance at fixed (X1=x,Y0=b) are c+2x+.5b
    and .25. Separate regressions on X1 and Y0 double-count marginal means
    and count unconditioned parent variation as conditional noise.
    """
    mean_y0 = intercept + 2 * root_mean
    mean_y1 = intercept + 2 * root_mean + 0.5 * mean_y0
    return dict(intercept=intercept, root_mean=root_mean,
                mean_y0=mean_y0, mean_y1=mean_y1,
                var_y0=4.25, true_conditional_variance=0.25,
                summed_conditional_variance=5.5625,
                expected_mean_bias=mean_y1)


def moment_data(case):
    """Degree-two Gaussian moment cubature, not a purported observational run."""
    import numpy as np
    e = np.asarray(list(itertools.product((-1.0, 1.0), repeat=6)))
    x = case['root_mean'] + e[:, [0, 3]]
    z = e[:, [1, 4]]
    y0 = case['intercept'] + 2 * x[:, 0] + 0.5 * e[:, 2]
    y1 = case['intercept'] + 2 * x[:, 1] + 0.5 * y0 + 0.5 * e[:, 5]
    return {'X': x, 'Z': z, 'Y': np.column_stack((y0, y1))}


class LinearProjection:
    """Population linear conditional moments on the moment-exact design."""
    def __init__(self, x, y):
        import numpy as np
        x, y = np.asarray(x), np.asarray(y)
        design = np.column_stack((np.ones(len(x)), x))
        self.coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        self.variance = np.mean((y - design @ self.coefficients) ** 2, axis=0)

    def predict(self, x):
        import numpy as np
        x = np.atleast_2d(x)
        design = np.column_stack((np.ones(len(x)), x))
        return design @ self.coefficients, np.tile(self.variance, (len(x), 1))


def backend_probe():
    import numpy as np
    import networkx as nx
    from ccbo.qdcbo.quotient_dbn import build_quotient_dbn, ensure_dcbo_on_path
    ensure_dcbo_on_path()
    from ccbo.qdcbo import qdcbo_base as qb, stock_fixes as sf
    from dcbo.utils.sem_utils.sem_estimate import build_sem_hat

    graph = nx.MultiDiGraph()
    for t in range(2):
        graph.add_nodes_from(f'{v}_{t}' for v in ('X', 'Z', 'Y'))
        graph.add_edge(f'X_{t}', f'Y_{t}')
    graph.add_edge('Y_0', 'Y_1')
    graph.T = 2

    def parents(node, t):
        return tuple(p for p in graph.predecessors(node)
                     if int(p.rsplit('_', 1)[1]) == t)

    rows, sources = [], {}
    for module in (qb, sf, sys.modules[build_sem_hat.__module__]):
        source = Path(module.__file__)
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()

    # Nonzero means detect duplicated means; centered control shows that
    # agreement of means alone does not validate temporal uncertainty.
    for case_name, case in [('noncentered', analytic_case()),
                            ('centered', analytic_case(0.0, 0.0))]:
        data = moment_data(case)
        with patch.object(sf._gu, 'fit_gp', LinearProjection):
            emission = sf.fixed_fit_arcs(graph, data, emissions=True)
            transition = sf.fixed_fit_arcs(graph, data, emissions=False)
        stock_hat = build_sem_hat(graph, emission, transition)
        builders = [('DCBO-corrected-time-index', stock_hat, None)]
        for label, partition in [('QDCBO-singleton', [['X'], ['Z']]),
                                 ('QDCBO-coarse', [['X', 'Z']])]:
            spec = build_quotient_dbn(graph, partition)
            with patch.object(qb, 'fit_joint_gp', LinearProjection):
                emit_q = qb.fit_quotient_arcs(spec, data, emissions=True)
                trans_q = qb.fit_quotient_arcs(spec, data, emissions=False)
            builders.append((label, qb.make_qsem_hat(spec)(
                G=spec.G_quotient, emission_fncs=emit_q,
                transition_fncs=trans_q), spec))

        # One declared incoming blanket, shared by all methods. All roots
        # are clamped, so no KDE draw or numerical fitting randomness enters.
        for label, hat, spec in builders:
            for x in (-2.0, 0.0, 2.0):
                blanket = {'X': [0.0, x], 'Z': [0.0, 0.0],
                           'Y': [7.0, None]}
                moments = []
                for moment in (0, 1):
                    kwargs = dict(static_sem=hat().static(moment),
                                  dynamic_sem=hat().dynamic(moment),
                                  timesteps=2, interventions=blanket, seed=0)
                    if spec is None:
                        sample = sf.fixed_sequential_sample_from_SEM_hat(
                            node_parents=parents, **kwargs)
                    else:
                        sample = qb.sequential_sample_from_qsem_hat(
                            spec=spec, stock_quirks=False, **kwargs)
                    assert sample['X'][1] == x and sample['Z'][1] == 0.0
                    assert sample['Y'][0] == 7.0
                    moments.append(float(sample['Y'][1]))
                truth = case['intercept'] + 2 * x + 0.5 * 7.0
                row = dict(case=case_name, method=label, current_x=x,
                           incoming_blanket={'X_0': 0.0, 'Z_0': 0.0, 'Y_0': 7.0},
                           predicted_mean=moments[0], true_mean=truth,
                           mean_error=moments[0] - truth,
                           predicted_variance=moments[1], true_variance=0.25,
                           expected_mean_bias=case['expected_mean_bias'])
                # These assertions validate the diagnostic witness, NOT
                # the correctness of the inherited temporal approximation.
                assert np.isclose(row['mean_error'], case['expected_mean_bias'], atol=1e-10)
                assert np.isclose(moments[1], case['summed_conditional_variance'], atol=1e-10)
                rows.append(row)

    return dict(status='inaccurate_temporal_prior_confirmed',
                diagnostic_witness_passed=True, fitted_gp_count=0,
                mean_max_absolute_error=max(abs(r['mean_error']) for r in rows),
                variance_max_absolute_error=max(abs(r['predicted_variance'] - .25) for r in rows),
                source_sha256=sources, cases=rows,
                interpretation='Real dispatch sums two full-response conditional regressions. '
                'Moment-exact fits isolate composition error, not finite-data GP error. '
                'This witness is not an estimate of performance harm on benchmark tasks.')


def audit_payload(payload, csv_rows=None):
    """Validate pinned protocol: first observation, then interventions per slice.

    Refuse unsupported schedules instead of guessing events from scalar costs.
    In upstream, outcomes starts with one sentinel, then appends once per
    trial; chosen_sets appends only for interventions; levels are trial-indexed.
    """
    issues, events = [], []
    for block in payload['per_t']:
        t = block['t']
        best, costs = block['best_so_far'], block['per_trial_cost']
        outcomes, chosen = block['outcome_values'], block['chosen_sets']
        n = len(best)
        if (len(costs) != n or len(outcomes) != n + 1 or
                len(chosen) != n - 1 or costs[0] != 0):
            issues.append(f'slice {t}: unsupported schedule/length; cannot reconstruct safely')
            continue
        running = outcomes[0]
        for j in range(n):
            y = outcomes[j + 1]
            if j == 0:
                event = dict(t=t, trial=j, kind='observe', arm=None, x=None,
                             y=None, cost=costs[j], best=best[j],
                             eligible_recommendation=False)
                if y != outcomes[0] or best[j] != outcomes[0]:
                    issues.append(f'slice {t} trial0: unexpected observation/sentinel update')
            else:
                arm = chosen[j - 1]
                key = '+'.join(arm)
                levels = block['levels_by_set'].get(key)
                x = levels[j] if levels is not None and j < len(levels) else None
                active = [k for k, v in block['levels_by_set'].items()
                          if j < len(v) and v[j] is not None]
                if x is None or active != [key]:
                    issues.append(f'slice {t} trial{j}: levels disagree with intervention sequence')
                if costs[j] != len(arm):
                    issues.append(f'slice {t} trial{j}: per-variable unit cost mismatch')
                running = min(running, y)
                if not math.isclose(running, best[j], rel_tol=1e-12, abs_tol=1e-12):
                    issues.append(f'slice {t} trial{j}: incumbent not aligned with outcome')
                event = dict(t=t, trial=j, kind='intervene', arm=arm, x=x,
                             y=y, cost=costs[j], best=best[j],
                             eligible_recommendation=True)
            events.append(event)
        if 'eligible_recommendation' in block:
            expected_flags = [False] + [True] * (n - 1)
            if block['eligible_recommendation'] != expected_flags:
                issues.append(f'slice {t}: recommendation eligibility differs from events')
            if block.get('recommended_values') != [None] + best[1:]:
                issues.append(f'slice {t}: reported recommendations expose sentinel or differ from events')
    if csv_rows is not None:
        lookup = {(int(r['time_index']), int(r['trial_index'])): r for r in csv_rows}
        if len(lookup) != len(events):
            issues.append('CSV event count differs from reconstructed events')
        for e in events:
            row = lookup.get((e['t'], e['trial']))
            if row is None:
                issues.append(f"CSV missing slice{e['t']} trial{e['trial']}")
                continue
            coherent = str(payload.get('unit', {}).get('mechanism_protocol', '')).startswith('coherent')
            blank_recommendation = coherent and not e['eligible_recommendation']
            if blank_recommendation and row['best_so_far_value'] != '':
                issues.append('Coherent CSV exposes sentinel as recommendation')
            elif not blank_recommendation and not math.isclose(float(row['best_so_far_value']), e['best'], rel_tol=1e-12, abs_tol=1e-12):
                issues.append('CSV incumbent mismatch')
            if e['kind'] == 'observe':
                if row['y'] != '':
                    issues.append('CSV exposes an observation as intervention outcome')
            elif row['y'] == '' or not math.isclose(float(row['y']), e['y'], rel_tol=1e-12, abs_tol=1e-12):
                issues.append('CSV selected outcome off-by-one/mismatch')
    return dict(ok=not issues, issues=issues, events=events,
                scope='default pinned schedule: first observation then interventions',
                first_observation_has_no_executed_incumbent=True)


def archive_probe(directory):
    results = []
    for path in sorted(Path(directory).glob('*.decisions.json')):
        payload = json.loads(path.read_text())
        if payload.get('unit', {}).get('suite') != 'qdcbo':
            continue
        csv_path = path.with_name(path.name.replace('.decisions.json', '.csv'))
        csv_rows = None
        if csv_path.exists():
            with csv_path.open(newline='') as f:
                csv_rows = list(csv.DictReader(f))
        result = audit_payload(payload, csv_rows)
        result['file'] = str(path)
        result['csv_checked'] = csv_rows is not None
        # Full trace retained only for failures to keep aggregate small.
        result['event_count'] = len(result.pop('events'))
        results.append(result)
    return dict(files=len(results), all_pass=bool(results) and all(r['ok'] for r in results),
                csv_files_checked=sum(r['csv_checked'] for r in results), results=results)


def coherent_probe(gp_smoke=False):
    """Validate the opt-in repair separately from the historical witness."""
    import networkx as nx
    from ccbo.qdcbo.quotient_dbn import build_quotient_dbn
    from ccbo.qdcbo.coherent import fit_conditional_mechanisms, integrate_predictive
    graph = nx.MultiDiGraph()
    for t in range(2):
        graph.add_nodes_from(f'{v}_{t}' for v in ('X', 'Z', 'Y'))
        graph.add_edge(f'X_{t}', f'Y_{t}')
    graph.add_edge('Y_0', 'Y_1'); graph.T = 2
    rows = []
    for label, partition in [('fine', [['X'], ['Z']]), ('coarse', [['X', 'Z']])]:
        q = build_quotient_dbn(graph, partition)
        fitted = fit_conditional_mechanisms(q, moment_data(analytic_case()), LinearProjection)
        for x in (-2., 0., 2.):
            mu, var = integrate_predictive(q, fitted, 1,
                {'X': [0., x], 'Z': [0., 0.], 'Y': [7., None]}, samples=16, seed=0)
            rows.append(dict(partition=label, x=x, predicted_mean=mu,
                             true_mean=6.5+2*x, predicted_variance=var,
                             true_variance=.25))
    max_error = max(max(abs(r['predicted_mean']-r['true_mean']),
                        abs(r['predicted_variance']-.25)) for r in rows)
    assert max_error < 1e-10, rows
    result = dict(status='exact_linear_witness_passed', cases=rows,
                  maximum_error=max_error, gp_smoke='not_requested')
    if gp_smoke:
        import tempfile
        from ccbo.qdcbo.runner import run_unit, decisions_payload, write_csv
        setup_results = {}
        for setup in ('stat', 'ind', 'nonstat'):
            traces, timing, menus = {}, {}, {}
            for label, algo, partition, edit, menu in [
                ('fine', 'DCBO', None, '', 'native'),
                ('singleton', 'QDCBO', [['X'], ['Z']], '', 'native'),
                ('fine-coarse-menu', 'DCBO', None, '', 'coarse'),
                ('coarse', 'QDCBO', [['X', 'Z']], '', 'native'),
                ('coarse-protected', 'QDCBO', [['X', 'Z']], 'e2', 'native')]:
                model, elapsed = run_unit(setup, algo, 3, 3, 1000, misspec=edit,
                    n_obs=12, num_anchor_points=9, partition=partition,
                    mechanism_protocol='coherent', predictive_samples=16,
                    action_menu=menu)
                payload = decisions_payload(model, algo, setup, 3, 3, 1000, edit, False)
                with tempfile.TemporaryDirectory(prefix='coherent-csv-gate-') as tmp:
                    path = write_csv(model, label, setup, 3, 3, 1000, edit, tmp)
                    with open(path, newline='') as f:
                        csv_rows = list(csv.DictReader(f))
                    alignment = audit_payload(payload, csv_rows)
                assert alignment['ok'], alignment
                traces[label] = {'per_t': payload['per_t'],
                                 'assigned_blanket': payload['assigned_blanket']}
                menus[label] = payload['exploration_sets']
                timing[label] = elapsed
            assert traces['fine'] == traces['singleton'], f'{setup}: coherent singleton trace mismatch'
            assert traces['coarse'] == traces['coarse-protected'], f'{setup}: protected coarse trace mismatch'
            assert menus['fine-coarse-menu'] == menus['coarse'], f'{setup}: action menus differ'
            setup_results[setup] = dict(seconds=timing, traces=traces, menus=menus)
        result['gp_smoke'] = dict(status='passed', settings=dict(T=3, trials=3,
            seed=1000, n_obs=12, anchor_points=9, integration_particles=16,
            setups=['stat', 'ind', 'nonstat']), results=setup_results)
    return result


def population_probe():
    """Bounded end-to-end gate of the NEW stochastic-prefix objective."""
    from ccbo.qdcbo.runner import run_unit, decisions_payload, write_csv
    import tempfile
    results = {}
    def learner_trace(model):
        return dict(events=[{k: e[k] for k in ('t', 'trial', 'arm', 'x', 'history',
                    'training_mean', 'training_mc_se', 'recommendation_event')}
                    for e in model.population_events],
                    policies=[{k: c[k] for k in ('t', 'arm', 'x', 'history',
                              'training_mean', 'recommendation_event')} for c in model.policy_commits])
    for setup in ('stat', 'ind', 'nonstat_regularized'):
        models, timing = {}, {}
        for label, algo, partition, edit, menu in [
            ('fine', 'DCBO', None, '', 'native'),
            ('singleton', 'QDCBO', [['X'], ['Z']], '', 'native'),
            ('fine-coarse-menu', 'DCBO', None, '', 'coarse'),
            ('coarse', 'QDCBO', [['X', 'Z']], '', 'native'),
            ('coarse-protected', 'QDCBO', [['X', 'Z']], 'e2', 'native')]:
            model, elapsed = run_unit(setup, algo, 3, 3, 1000, misspec=edit,
                n_obs=12, num_anchor_points=9, partition=partition,
                mechanism_protocol='coherent', predictive_samples=16,
                action_menu=menu, objective_protocol='population-mc',
                feedback_samples=128, score_samples=1024)
            assert len(model.policy_commits) == 3 and len(model.population_events) == 6
            for event in model.population_events:
                assert all(value is None for value in event['history']['Y'])
                assert event['recommendation_event'] <= event['event_id']
                for prior_t in range(event['t']):
                    committed = model.policy_commits[prior_t]
                    for variable in ('X', 'Z'):
                        expected = (committed['x'][committed['arm'].index(variable)]
                                    if variable in committed['arm'] else None)
                        assert event['history'][variable][prior_t] == expected
            with tempfile.TemporaryDirectory(prefix='population-csv-gate-') as tmp:
                path = write_csv(model, label, setup, 3, 3, 1000, edit, tmp)
                with open(path, newline='') as f:
                    rows = list(csv.DictReader(f))
                assert len(rows) == 9
                for row in rows:
                    if int(row['trial_index']) == 0:
                        assert row['population_recommendation_mean'] == ''
                        assert row['eligible_recommendation'] == 'False'
            models[label] = model; timing[label] = elapsed
        assert learner_trace(models['fine']) == learner_trace(models['singleton'])
        assert learner_trace(models['coarse']) == learner_trace(models['coarse-protected'])
        assert models['fine-coarse-menu'].exploration_sets == models['coarse'].exploration_sets
        if setup == 'stat':
            alternate, _ = run_unit(setup, 'DCBO', 3, 3, 1000,
                n_obs=12, num_anchor_points=9, mechanism_protocol='coherent',
                predictive_samples=16, objective_protocol='population-mc',
                feedback_samples=128, score_samples=2048)
            assert learner_trace(alternate) == learner_trace(models['fine']), 'Scoring precision changed learner'
        results[setup] = dict(seconds=timing, traces={key: learner_trace(model)
                for key, model in models.items()}, scored_commits={key: model.policy_commits
                for key, model in models.items()})
    return dict(status='passed', settings=dict(T=3, trials=3, seed=1000, n_obs=12,
        anchor_points=9, integration_particles=16, training_mc=128, scoring_mc=1024),
        results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--skip-backends', action='store_true')
    parser.add_argument('--require-accurate', action='store_true')
    parser.add_argument('--coherent', action='store_true',
                        help='also validate the opt-in union-parent repair')
    parser.add_argument('--gp-smoke', action='store_true',
                        help='with --coherent, run 15 tiny real-GP units over all three setups')
    parser.add_argument('--population-smoke', action='store_true',
                        help='run 16 tiny GP units for the new prefix-population protocol')
    args = parser.parse_args()
    result = {'schema': 1, 'diagnostic': 'temporal-sem-validity-v1'}
    if not args.skip_backends:
        result['backend'] = backend_probe()
    if args.archive_dir:
        result['archive_alignment'] = archive_probe(args.archive_dir)
    if args.coherent:
        result['coherent'] = coherent_probe(args.gp_smoke)
    elif args.gp_smoke:
        parser.error('--gp-smoke requires --coherent')
    if args.population_smoke:
        result['population'] = population_probe()
    if len(result) == 2:
        parser.error('select backend diagnostic or --archive-dir')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in
                    ('backend', 'archive_alignment', 'coherent', 'population')}, sort_keys=True))
    print(f'Report: {args.output}')
    if 'backend' in result:
        print('Temporal prior:', result['backend']['status'])
    if 'archive_alignment' in result:
        a = result['archive_alignment']
        print(f"Archive alignment: {a['files']} sidecars; all_pass={a['all_pass']}")
        if not a['all_pass']:
            raise SystemExit(1)
    if args.require_accurate and 'backend' in result:
        b = result['backend']
        if max(b['mean_max_absolute_error'], b['variance_max_absolute_error']) > 1e-8:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
