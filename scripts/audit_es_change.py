"""Audit: does the MIS exploration-set rule change ClusterBench10 behavior?

Compares, for every registered ClusterBench10 perturbation variant and for
the coarse / finest / wrong-pi / refined partitions:

  OLD rule (retired): all non-empty cluster subsets up to size 5, kept only
      if identifiable (the deletion gate), then POMIS union.
  NEW rule: MIS of the C-DAG (Lee & Bareinboim 2018), membership never
      gated; identifiability decides the prior tier.

A cell whose (ordered) arm list and tier assignment coincide under both
rules produces byte-identical trajectories, so its checked-in results stay
valid. Any difference is listed explicitly.

Run: PYTHONPATH=. python scripts/audit_es_change.py
"""

import itertools

from ccbo import clusterbench10 as cb
from ccbo.adjustment import _ananke_id_check
from ccbo.coarsening import (get_dag_edges_from_sem, project_out_hidden,
                             build_coarsened_admg, compute_MIS,
                             compute_POMIS)

TARGET = frozenset({"Y"})

COARSE = [frozenset(c) for c in cb.COARSE_CLUSTERS] + [frozenset({"Y"})]
FINEST = [frozenset({v}) for v in cb.MANIPULATIVE] + [frozenset({"Y"})]
WRONGPI = [frozenset({"X1", "X2", "X4"}), frozenset({"X3", "X5"}),
           frozenset({"Y"})]
REFINED = ([frozenset({v}) for v in cb.COARSE_CLUSTERS[0]]
           + [frozenset(cb.COARSE_CLUSTERS[1])] + [frozenset({"Y"})])

PARTITIONS = [("coarse", COARSE), ("finest", FINEST),
              ("wrongpi", WRONGPI), ("refined", REFINED)]


def _admg_for(variant, partition):
    _, nodes, hidden, manip = get_dag_edges_from_sem(variant)
    proj = project_out_hidden(variant)
    hidden = set(hidden or [])
    nonmanip = set(nodes) - hidden - set(manip) - {"Y"}
    admg = build_coarsened_admg(partition, proj, atomic_vertices=nonmanip)
    mnodes = [p for p in partition
              if p != frozenset({"Y"}) and p <= set(manip)]
    return admg, mnodes


def old_rule(admg, mnodes, cap=5):
    """Retired rule: gated subset enumeration (partition order) + POMIS."""
    seen, out = set(), []

    def _add(combo):
        key = frozenset(combo)
        if key in seen or not key or len(key) > cap:
            return
        status, _, _ = _ananke_id_check(admg, set(combo), TARGET)
        ok = status == "identified"
        if not ok:
            return
        seen.add(key)
        out.append(key)

    for r in range(1, min(len(mnodes), cap) + 1):
        for combo in itertools.combinations(mnodes, r):
            _add(frozenset(combo))
    for combo in compute_POMIS(admg, mnodes, TARGET):
        _add(combo)
    return [tuple(sorted(v for c in k for v in c)) for k in out]


def new_rule(admg, mnodes):
    """MIS + per-arm identifiability tier."""
    mis = compute_MIS(admg, mnodes, TARGET)
    arms, tiers = [], {}
    for combo in mis:
        status, _, _ = _ananke_id_check(admg, set(combo), TARGET)
        ok = status == "identified"
        key = tuple(sorted(v for c in combo for v in c))
        arms.append(key)
        tiers[key] = bool(ok)
    return arms, tiers


def main():
    cb.register_variants()
    pids = [p["id"] for p in cb.PERTURBATIONS]
    changed = []
    for pid in pids:
        variant = cb.variant_name(pid)
        for pname, partition in PARTITIONS:
            try:
                admg, mnodes = _admg_for(variant, partition)
            except ValueError as e:   # e.g. cyclic quotient
                print(f"{pid:4s} {pname:8s} SKIP ({e})")
                continue
            old = old_rule(admg, mnodes)
            new, tiers = new_rule(admg, mnodes)
            uninf = sorted(k for k, ok in tiers.items() if not ok)
            same = (sorted(old) == sorted(new)) and not uninf
            flag = "SAME" if same else "DIFF"
            if not same:
                changed.append((pid, pname))
                only_old = sorted(set(old) - set(new))
                only_new = sorted(set(new) - set(old))
                print(f"{pid:4s} {pname:8s} {flag}  "
                      f"old-only={only_old} new-only={only_new} "
                      f"uninformative={uninf}")
            else:
                print(f"{pid:4s} {pname:8s} {flag}  ({len(new)} arms)")
    print()
    if changed:
        print(f"{len(changed)} (variant, partition) cells change under the "
              f"MIS rule:")
        for pid, pname in changed:
            print(f"  {pid} / {pname}")
    else:
        print("No cell changes: checked-in ClusterBench10 results remain "
              "valid under the MIS rule.")


if __name__ == "__main__":
    main()
