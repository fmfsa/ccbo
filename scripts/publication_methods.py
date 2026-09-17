"""Publication names and required controlled-suite inventory; archive keys stay intact."""
from pathlib import Path
import json

PLAIN = ("BO", "BOS", "CBO", "CBONP", "QCBO", "QCBONP")
PRIMARY_GROUPS = tuple(
    [("ParallelParent", c, a) for c in ("A0", "A1", "A2", "A3") for a in PLAIN]
    + [("FrontDoor", c, a) for c in ("B0", "B1") for a in PLAIN]
    + [("MediatedChain", "C0", a) for a in PLAIN + ("HQCBOGF",)])
HISTORICAL_GROUPS = PRIMARY_GROUPS + (("MediatedChain", "C0", "HQCBO"),)
NAMES = {"BOS": "BO-S", "CBONP": "CBO−np", "QCBONP": "QCBO−np",
         "HQCBOGF": "HQCBO", "HQCBO": "HQCBO (historical graph-informed)"}

def publication_name(key, latex=False):
    name = NAMES.get(key, key)
    return name.replace("−", "$-$") if latex else name

def validate_summary(groups, seeds=30):
    for scm, cond, arm in PRIMARY_GROUPS:
        key = f"{scm}_{cond}_{arm}"
        if key not in groups:
            raise ValueError(f"Missing required primary result: {key}")
        if groups[key].get("n") != seeds:
            raise ValueError(f"{key}: expected {seeds} seeds, got {groups[key].get('n')}")

def required_files(root, prefix, seeds=30):
    root = Path(root)
    expected = {root / f"{prefix}_seed{s}.csv" for s in range(seeds)}
    actual = set(root.glob(f"{prefix}_seed*.csv"))
    if actual != expected:
        raise ValueError(f"{prefix}: expected seeds 0..{seeds-1}; "
                         f"missing={sorted(p.name for p in expected-actual)}, "
                         f"unexpected={sorted(p.name for p in actual-expected)}")
    return sorted(expected, key=lambda p: int(p.stem.rsplit('seed', 1)[1]))

def write_plot_provenance(out, root, groups):
    payload = {"source_directory": str(Path(root).resolve()),
               "methods": [{"scm": s, "condition": c, "archive_key": a,
                            "publication_name": publication_name(a)} for s,c,a in groups]}
    Path(out).with_suffix(".provenance.json").write_text(json.dumps(payload, indent=2)+"\n")
