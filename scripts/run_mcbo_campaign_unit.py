"""Runtime/source gate before dispatching the unchanged corrected MCBO CLI."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
provenance=json.loads((ROOT/'envs/mcbo-runtime-provenance.json').read_text())
expected=provenance['runtime']
actual={'python':sys.version.split()[0],
        'packages':{p:importlib.metadata.version(p) for p in expected['packages']}}
if actual!=expected:
    raise RuntimeError('Pinned runtime mismatch: '+json.dumps(actual,sort_keys=True))
from ccbo.qmcbo.quotient import ensure_mcbo_on_path
ensure_mcbo_on_path()
module_paths={'mcbo.mcbo_trial':'third_party/mcbo/mcbo/mcbo_trial.py',
 'mcbo.models.gp_network':'third_party/mcbo/mcbo/models/gp_network.py',
 'mcbo.acquisition_function_optimization.optimize_acqf':'third_party/mcbo/mcbo/acquisition_function_optimization/optimize_acqf.py',
 'functions':'third_party/mcbo/scripts/functions.py'}
imports={}
for name,relative in module_paths.items():
    module=importlib.import_module(name)
    path=Path(module.__file__).resolve()
    if path!=(ROOT/relative).resolve():raise RuntimeError('Wrong imported source: '+name)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=provenance['source_sha256'][relative]:raise RuntimeError('Wrong imported source hash: '+name)
    imports[name]={'relative_source':relative,'sha256':sha,'path':str(path)}
# Manifest runner validates every upstream file too; these record actual imports.
actual.update(executable=sys.executable,imports=imports,vendor_git_revision=provenance['vendor_git_revision'])
outdir=Path(sys.argv[sys.argv.index('--outdir')+1]);outdir.mkdir(parents=True,exist_ok=True)
(outdir/'runtime.json').write_text(json.dumps(actual,indent=2,sort_keys=True)+'\n')
sys.argv[0]=str(ROOT/'scripts/diagnose_mcbo_corrected.py')
runpy.run_path(sys.argv[0],run_name='__main__')
