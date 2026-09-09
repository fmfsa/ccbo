"""Decision-log sidecars (engine v3): bit-exact round trip and the
full-precision invariance surface."""

import json
import os

import numpy as np
import pytest

from ccbo import decision_log as dlog


def test_roundtrip_is_bit_exact(tmp_path):
    xs = [0.1 + 0.2, 1 / 3, 1e-17, 123456.789012345678, -2.5e-300]
    log = [{'trial': 0, 'type': 'intervene', 'u': 0.37, 'arm': 'X1',
            'x': xs, 'y_new': 0.5, 'cum_cost': 7.0, 'incumbent': 0.5,
            'epsilon': 0.11, 'forced': None}]
    payload = dlog.build_payload(
        unit=dict(suite='t', scm='S', cond='A0', arm='QCBO', seed=0, trials=1,
                  n_init=3, n_obs=100, type_cost=1, max_N=150),
        init=dict(partition=[['X1']], es=[['X1']], prior_mask=[True],
                  x_list=[[[0.1]]], y_list=[[1.0]], init_cost=3.0,
                  incumbent=1.0, gate_status={'X1': 'identified'}),
        phases=[dlog.phase_record('main', [['X1']], [True], log)],
        rows=[(0, 1.0, 3.0, 'init', ''), (1, 0.5, 7.0, 'X1', ';'.join(repr(v) for v in xs))])
    path = str(tmp_path / 'u.decisions.json')
    dlog.write(path, payload)
    back = dlog.read(path)
    assert back['phases'][0]['trials'][0]['x'] == xs          # exact doubles
    assert back['rows'][1]['x'] == xs
    assert dlog.decisions_signature(back) == dlog.decisions_signature(payload)
    assert back['unit']['estimator']['variance_policy'] == 'predictive'
    assert back['unit']['estimator']['noise_fixed'] == 1e-10


def test_signature_distinguishes_a_single_ulp():
    def mk(x):
        log = [{'trial': 0, 'type': 'intervene', 'u': 0.1, 'arm': 'X', 'x': [x],
                'y_new': 0.0, 'cum_cost': 1.0, 'incumbent': 0.0}]
        return {'phases': [dlog.phase_record('main', [['X']], [False], log)]}
    a, b = 0.7, np.nextafter(0.7, 1.0)
    assert f"{a:.6g}" == f"{b:.6g}"                             # invisible to the old CSV
    assert dlog.decisions_signature(mk(a)) != dlog.decisions_signature(mk(b))


def test_sidecar_path():
    assert dlog.sidecar_path('/x/ParallelParent_A0_QCBO_seed0.csv') == \
        '/x/ParallelParent_A0_QCBO_seed0.decisions.json'


@pytest.mark.slow
def test_protected_pair_identical_at_full_precision_and_control_differs():
    from ccbo import minibench as mb
    from ccbo.minimal_suite import run_plain_unit
    sig = {}
    for cond in ("A0", "A1", "A3"):
        rows, info = run_plain_unit(mb.PP_NAME, cond, "QCBO", 0, 5, num_interventions=3)
        sig[cond] = dlog.decisions_signature(info["decision_log"])
        # CSV x_values are the repr of the logged doubles
        for r, rec in zip(rows, info["decision_log"]["rows"]):
            assert r[4] == ';'.join(repr(v) for v in rec['x'])
    assert sig["A0"] == sig["A1"]
    assert sig["A0"] != sig["A3"]
