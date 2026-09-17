"""Protect new experiments from source drift and stale output reuse."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('frozen',Path(__file__).resolve().parents[2]/'scripts/run_frozen_unit.py')
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)

def fixture(tmp):
    root=tmp/'repo';root.mkdir();script=root/'unit.py'
    script.write_text('import pathlib,sys\npathlib.Path(sys.argv[1],"result.json").write_text("{}")\n')
    m={'schema':1,'protocol':{'learning':'noisy','stage':'pilot'},'source_sha256':{'unit.py':frozen.digest(script)},'units':[{'id':'seed0','argv':['{python}','unit.py','{unit_dir}'],'outputs':['result.json']}]}
    return root,script,m

def test_code_drift_rejected(tmp_path):
    root,script,m=fixture(tmp_path);script.write_text('raise RuntimeError("changed")')
    with pytest.raises(ValueError,match='source mismatch'):frozen.run(m,0,tmp_path/'results',root)
    assert not (tmp_path/'results').exists()

def test_resume_checks_output_and_protocol(tmp_path):
    root,script,m=fixture(tmp_path);status=frozen.run(m,0,tmp_path/'results',root)
    assert status['status']=='complete'
    assert frozen.run(m,0,tmp_path/'results',root)==status
    (tmp_path/'results'/frozen.manifest_id(m)/'seed0'/'result.json').write_text('{"altered":true}')
    with pytest.raises(ValueError,match='output changed'):frozen.run(m,0,tmp_path/'results',root)
    old=frozen.manifest_id(m);m['protocol']['learning']='exact'
    assert old!=frozen.manifest_id(m)

def test_missing_output_is_failure(tmp_path):
    root,script,m=fixture(tmp_path);m['units'][0]['outputs'].append('absent.json')
    with pytest.raises(ValueError,match='Missing/empty'):frozen.run(m,0,tmp_path/'results',root)
    with pytest.raises(ValueError,match='incomplete/failed'):frozen.run(m,0,tmp_path/'results',root)
