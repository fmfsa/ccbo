"""Strict, shared checks for the model-based paper's paired decision logs."""

import math


QMCBO_E2_EDITS = {"ToyGraph": "del:0:1", "PSAGraph": "add:2:3"}


def validate_qmcbo_metadata(log, env, algo, seed, expected_iterations, *, perturbed=False):
    """Validate provenance against the expected file identity and graph view."""
    if not isinstance(log, dict) or type(log.get("schema")) is not int or log["schema"] != 1:
        raise ValueError("expected QMCBO decision schema 1")
    unit = log.get("unit")
    if not isinstance(unit, dict):
        raise ValueError("missing QMCBO unit metadata")
    if env not in QMCBO_E2_EDITS:
        raise ValueError(f"unknown QMCBO environment {env}")
    expected = {"suite": "qmcbo", "env": env, "algo": algo, "seed": seed,
                "num_trials": expected_iterations,
                "misspec": QMCBO_E2_EDITS[env] if perturbed else ""}
    for key, value in expected.items():
        if type(unit.get(key)) is not type(value) or unit.get(key) != value:
            raise ValueError(f"mismatched QMCBO {key}: expected {value!r}, got {unit.get(key)!r}")


def qmcbo_decisions(log, expected_iterations):
    """Return validated (X, score) records; missing evidence is an error."""
    iterations = log.get("iterations", [])
    if len(iterations) != expected_iterations:
        raise ValueError(f"expected {expected_iterations} iterations, got {len(iterations)}")
    records = []
    width = None
    for i, row in enumerate(iterations):
        if row.get("iter") != i:
            raise ValueError(f"missing or out-of-order iteration {i}")
        x, score = row.get("X"), row.get("score")
        if not isinstance(x, list) or not x:
            raise ValueError(f"missing intervention vector at iteration {i}")
        if width is None:
            width = len(x)
        if len(x) != width:
            raise ValueError(f"inconsistent intervention dimension at iteration {i}")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in x):
            raise ValueError(f"non-finite intervention at iteration {i}")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"missing/non-finite score at iteration {i}")
        records.append((x, score))
    return records


def compare_qmcbo_pair(left, right, left_log, right_log, expected_iterations):
    """Separate final, incumbent-trajectory, and actual decision equality."""
    if len(left) != expected_iterations or len(right) != expected_iterations:
        raise ValueError("incumbent trajectory length differs from the declared trials")
    if not all(math.isfinite(v) for v in left + right):
        raise ValueError("non-finite incumbent")
    a = qmcbo_decisions(left_log, expected_iterations)
    b = qmcbo_decisions(right_log, expected_iterations)
    if any(len(x[0]) != len(y[0]) for x, y in zip(a, b)):
        raise ValueError("paired intervention dimensions differ")
    return {
        "identical_final": left[-1] == right[-1],
        "identical_incumbent_trajectory": left == right,
        "identical_decisions_and_scores": a == b,
        "abs_final_deviation": abs(left[-1] - right[-1]),
        "max_abs_incumbent_deviation": max(abs(x-y) for x, y in zip(left, right)),
        "max_abs_intervention_deviation": max(abs(u-v) for (x, _), (y, _) in zip(a, b)
                                              for u, v in zip(x, y)),
        "max_abs_score_deviation": max(abs(x[1]-y[1]) for x, y in zip(a, b)),
    }
