"""Final-release runtime/import audit before the unchanged dynamic runner."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
provenance=json.loads((ROOT/'envs/dynamic-runtime-provenance.json').read_text())
expected=provenance['runtime']
actual={'python':sys.version.split()[0],
        'packages':{name:importlib.metadata.version(name) for name in expected['packages']}}
if actual!=expected:
    raise RuntimeError('Pinned dynamic runtime mismatch: '+json.dumps(actual,sort_keys=True))
from ccbo.qdcbo.quotient_dbn import ensure_dcbo_on_path
ensure_dcbo_on_path()
module_paths={
    'dcbo.methods.dcbo':'third_party/DCBO/dcbo/methods/dcbo.py',
    'dcbo.bases.root':'third_party/DCBO/dcbo/bases/root.py',
    'dcbo.bases.dcbo_base':'third_party/DCBO/dcbo/bases/dcbo_base.py',
    'dcbo.utils.sem_utils.toy_sems':'third_party/DCBO/dcbo/utils/sem_utils/toy_sems.py',
    'ccbo.qdcbo.coherent':'ccbo/qdcbo/coherent.py',
    'ccbo.qdcbo.population':'ccbo/qdcbo/population.py',
}
imports={}
for name,relative in module_paths.items():
    path=Path(importlib.import_module(name).__file__).resolve()
    if path!=(ROOT/relative).resolve():
        raise RuntimeError('Wrong imported dynamic source: '+name)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=provenance['source_sha256'][relative]:
        raise RuntimeError('Wrong imported dynamic source hash: '+name)
    imports[name]={'relative_source':relative,'sha256':sha,'path':str(path)}
actual.update(executable=sys.executable,imports=imports,
    scoring_order_evidence='Frozen runner invokes score_events only after model.run; firewall regression passed. No event phase timestamps.')
outdir=Path(sys.argv[sys.argv.index('--outdir')+1]);outdir.mkdir(parents=True,exist_ok=True)
(outdir/'runtime.json').write_text(json.dumps(actual,indent=2,sort_keys=True)+'\n')
sys.argv[0]=str(ROOT/'ccbo/qdcbo/runner.py')
runpy.run_module('ccbo.qdcbo.runner',run_name='__main__')
