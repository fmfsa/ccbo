"""Full-precision per-unit decision logs (engine v3, 2026-09).

Every MinimalBench / family unit writes ``<unit>.decisions.json`` next to
its CSV.  The payload records everything the quotient-contract coupling
claims to hold fixed or identical: the initial design and its cost, the
exploration set and per-arm prior mask of every optimizer phase, and the
complete per-trial log of the CBO loop (observe/intervene draw ``u`` and
``epsilon_t``, forced flags, arm, intervention values at full double
precision, outcome, cumulative cost, incumbent).  ``decisions_signature``
reduces a payload to the tuple sequence the invariance gate compares.

JSON round-trips Python floats exactly (``repr`` semantics), so equality of
two signatures is equality of the original doubles.
"""

import json
import os
import subprocess

import numpy as np

SCHEMA_VERSION = 1


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, frozenset):
        return sorted(str(v) for v in x)
    return x


def engine_sha():
    """Git SHA of the engine that produced a unit (env override for LSF)."""
    sha = os.environ.get("CCBO_GIT_SHA")
    if sha:
        return sha
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ["git", "-C", here, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def estimator_info():
    from ccbo.adjustment import ESTIMATOR_INFO
    from ccbo.cbo.utils.BO_functions import NOISE_VAR
    return {**dict(ESTIMATOR_INFO), 'noise_fixed': float(NOISE_VAR)}


def phase_record(name, es, prior_mask, trial_log, global_offset=0):
    """One optimizer phase: its arms, prior mask and per-trial log with a
    ``global_trial`` index into the unit's trial axis."""
    entries = []
    for e in trial_log:
        rec = dict(e)
        rec['global_trial'] = int(global_offset) + int(rec.get('trial', 0))
        entries.append(rec)
    return {'phase': name, 'es': [list(a) for a in es],
            'prior_mask': [bool(b) for b in prior_mask],
            'trials': entries}


def rows_to_records(rows):
    """CSV rows ``(trial, best_y, cum_cost, arm, x_values)`` with the
    intervention values as float lists."""
    out = []
    for (t, y, c, a, xv) in rows:
        xs = [float(v) for v in str(xv).split(';')] if xv not in ('', None) else []
        out.append({'trial': int(t), 'best_y': float(y), 'cum_cost': float(c),
                    'arm': a, 'x': xs})
    return out


def build_payload(unit, init, phases, rows, refine=None, extra=None):
    payload = {
        'schema': SCHEMA_VERSION,
        'unit': dict(unit, engine_sha=engine_sha(), estimator=estimator_info()),
        'init': init,
        'phases': phases,
        'refine': refine,
        'rows': rows_to_records(rows),
    }
    if extra:
        payload['extra'] = extra
    return _jsonable(payload)


def write(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def read(path):
    with open(path) as fh:
        return json.load(fh)


def decisions_signature(payload):
    """Per-trial ``(global_trial, type, u, arm, x, y_new, cum_cost,
    incumbent)`` over all phases — the exact-equality surface of the
    quotient contract at full precision."""
    sig = []
    for ph in payload.get('phases', []):
        for e in ph.get('trials', []):
            sig.append((e.get('global_trial'), e.get('type'), e.get('u'),
                        e.get('arm'), tuple(e.get('x') or ()),
                        e.get('y_new'), e.get('cum_cost'), e.get('incumbent')))
    return sig


def sidecar_path(csv_path):
    base = csv_path[:-4] if csv_path.endswith('.csv') else csv_path
    return base + ".decisions.json"
