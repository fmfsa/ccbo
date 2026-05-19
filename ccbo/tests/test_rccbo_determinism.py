"""
Regression test: rccbo() is deterministic given a seed.

Locks in the A1 audit fixes (per-phase re-seeding + per-call seed threading
through the intervention callback). Two runs at the same seed must produce
identical (global_opt, partition_history, exploration_set, final_partition).
"""

import numpy as np

from ccbo.rccbo.rccbo import rccbo


def _strip_nondeterministic(result):
    """Return only the fields that should be byte-identical across runs."""
    return {
        'global_opt': list(result['global_opt']),
        'partition_history': result['partition_history'],
        'refinement_trials': result['refinement_trials'],
        'exploration_set': result['exploration_set'],
        'final_partition': result['final_partition'],
    }


def test_rccbo_seed_reproducibility():
    """Two rccbo calls with the same seed must produce identical results.

    Short budget keeps wall time low (~minutes); the determinism property
    holds regardless of budget.
    """
    kwargs = dict(
        experiment='CompleteGraph',
        total_trials=15,
        k_phase=5,
        num_interventions=5,
        type_cost=1,
        initial_num_obs_samples=50,
        task='min',
        num_mc_samples=500,
    )

    r1 = rccbo(seed=0, **kwargs)
    r2 = rccbo(seed=0, **kwargs)

    a, b = _strip_nondeterministic(r1), _strip_nondeterministic(r2)
    assert a['global_opt'] == b['global_opt'], (
        f"global_opt mismatch:\n  run1={a['global_opt']}\n  run2={b['global_opt']}"
    )
    assert a['partition_history'] == b['partition_history']
    assert a['refinement_trials'] == b['refinement_trials']
    assert a['exploration_set'] == b['exploration_set']
    assert a['final_partition'] == b['final_partition']


if __name__ == '__main__':
    test_rccbo_seed_reproducibility()
    print('PASS: rccbo determinism')
