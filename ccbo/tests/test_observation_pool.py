"""Observation-pool contract for the CBO loop.

Every observe action must reveal a *fresh* batch of previously unseen rows,
the final batch may be short, and nothing may be revealed past the effective
cap. The pre-fix implementation returned the constant slice
``full_obs[initial : initial + batch]`` on every call, so observations were
silently duplicated and the declared budget was never enforced.

Dataset-free: these are pure properties of ``observe`` and of the cursor
bookkeeping in ``ccbo/cbo/cbo.py``.
"""

import numpy as np
import pandas as pd
import pytest

from ccbo.cbo.utils import observe
from ccbo.cbo.cbo import _check_observation_prefix

N0, BATCH, CAP = 100, 20, 150


def _pool(n=5000, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"X": rng.randn(n), "Y": rng.randn(n)})


def _drain(pool, start=N0, cap=CAP, batch=BATCH):
    """Every batch a run would reveal, until the budget is exhausted."""
    cursor, batches = start, []
    while cursor < cap:
        rows, cursor = observe(pool, cursor, batch, cap)
        batches.append(rows)
    return batches, cursor


def test_first_batch_starts_at_the_initial_cursor():
    pool = _pool()
    rows, cursor = observe(pool, N0, BATCH, CAP)
    assert cursor == N0 + BATCH
    assert np.allclose(rows["X"].values, pool["X"].values[N0:N0 + BATCH])


def test_second_batch_is_disjoint_from_the_first():
    pool = _pool()
    first, cursor = observe(pool, N0, BATCH, CAP)
    second, cursor = observe(pool, cursor, BATCH, CAP)
    assert np.allclose(second["X"].values,
                       pool["X"].values[N0 + BATCH:N0 + 2 * BATCH])
    assert not set(np.round(first["X"], 12)) & set(np.round(second["X"], 12))


def test_final_batch_is_truncated_to_the_remaining_budget():
    pool = _pool()
    batches, cursor = _drain(pool)
    # cap 150, start 100, batch 20 -> 20 + 20 + 10
    assert [len(b) for b in batches] == [20, 20, 10]
    assert cursor == CAP


def test_no_observation_past_the_cap():
    pool = _pool()
    _, cursor = _drain(pool)
    rows, cursor_after = observe(pool, cursor, BATCH, CAP)
    assert len(rows) == 0
    assert cursor_after == cursor == CAP


def test_every_revealed_row_is_revealed_exactly_once():
    pool = _pool()
    batches, cursor = _drain(pool)
    seen = np.concatenate([b["X"].values for b in batches])
    assert len(seen) == cursor - N0 == len(np.unique(seen))
    assert np.allclose(seen, pool["X"].values[N0:cursor])


def test_slicing_is_positional_not_label_based():
    """A non-default index must not change which rows come back."""
    pool = _pool()
    shuffled = pool.copy()
    shuffled.index = np.random.RandomState(7).permutation(len(pool))
    plain, _ = observe(pool, N0, BATCH, CAP)
    odd, _ = observe(shuffled, N0, BATCH, CAP)
    assert np.allclose(plain["X"].values, odd["X"].values)


def test_cap_is_the_pool_when_the_pool_is_shorter_than_max_N():
    """obs_cap = min(max_N, len(pool)); a short pool binds first."""
    pool = _pool(n=130)
    cap = min(CAP, len(pool))
    batches, cursor = _drain(pool, cap=cap)
    assert cursor == 130
    assert [len(b) for b in batches] == [20, 10]


def test_resume_continues_from_the_saved_cursor():
    pool = _pool()
    _, cursor = observe(pool, N0, BATCH, CAP)
    resumed, _ = observe(pool, cursor, BATCH, CAP)
    # Same as if the run had never been interrupted.
    straight, c2 = observe(pool, N0, BATCH, CAP)
    straight, _ = observe(pool, c2, BATCH, CAP)
    assert np.allclose(resumed["X"].values, straight["X"].values)


# ---------------------------------------------------------------------------
# Cursor contract
# ---------------------------------------------------------------------------

def test_prefix_check_accepts_a_genuine_prefix():
    pool = _pool()
    _check_observation_prefix(pool.iloc[:N0], pool, N0)


def test_prefix_check_accepts_a_reindexed_prefix():
    """After an observe action the frame is concatenated with ignore_index."""
    pool = _pool()
    grown = pd.concat([pool.iloc[:N0], pool.iloc[N0:N0 + BATCH]],
                      ignore_index=True)
    _check_observation_prefix(grown, pool, N0 + BATCH)


def test_prefix_check_rejects_a_non_prefix_frame():
    pool = _pool()
    with pytest.raises(ValueError, match="not the first"):
        _check_observation_prefix(pool.iloc[50:150].reset_index(drop=True),
                                  pool, N0)


def test_prefix_check_rejects_a_cursor_length_mismatch():
    pool = _pool()
    with pytest.raises(ValueError, match="cursor says"):
        _check_observation_prefix(pool.iloc[:N0], pool, N0 + 1)


def test_prefix_check_rejects_a_cursor_past_the_pool():
    pool = _pool(n=50)
    with pytest.raises(ValueError, match="exceeds the pool"):
        _check_observation_prefix(pool, pool, 999)
