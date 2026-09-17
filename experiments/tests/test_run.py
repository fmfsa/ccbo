"""Portable paper matrix and fail-closed execution checks (no vendor required)."""
import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

spec = importlib.util.spec_from_file_location('paper_run', Path(__file__).parents[1] / 'run.py')
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def test_paper_matrices_and_scientific_settings():
    for suite, count, seeds in [('static',720,30),('mcbo',200,20),('dynamic',180,20)]:
        units = paper.units(suite)
        assert len(units) == count
        assert len({u['id'] for u in units}) == count
        assert {u['seed'] for u in units} == set(range(2000,2000+seeds))
        assert all(u['mode']=='paper' for u in units)
    mcbo = paper.units('mcbo')
    assert sum('-e2' in u['id'] for u in mcbo) == 80
    assert all(u['options']['--num-trials']=='100' and u['options']['--score-samples']=='100000' for u in mcbo)
    for u in paper.units('dynamic'):
        assert {k:u['options'][k] for k in ['--T','--trials','--mechanism-protocol','--objective-protocol','--predictive-samples','--feedback-samples','--score-samples']} == dict(zip(['--T','--trials','--mechanism-protocol','--objective-protocol','--predictive-samples','--feedback-samples','--score-samples'],['3','10','coherent','population-mc','1024','2048','100000']))


def test_smoke_is_explicit_and_seed_separated():
    for suite in ('static','mcbo','dynamic'):
        assert {u['seed'] for u in paper.units(suite,smoke=True)} == {1000}
        with pytest.raises(ValueError): paper.units(suite,seeds=[1000])
        with pytest.raises(ValueError): paper.units(suite,seeds=[2000],smoke=True)
        with pytest.raises(ValueError): paper.units(suite,seeds=[2000,2000])
    assert paper.units('static',smoke=True)[0]['options']['--stage']=='pilot'
    assert paper.units('static',smoke=True)[0]['options']['--max-purchases']=='3'


def test_failure_is_preserved_and_never_automatically_retried(tmp_path):
    unit = paper.units('static',smoke=True)[0]
    with patch.object(paper.subprocess,'run',return_value=SimpleNamespace(returncode=7)) as call:
        with pytest.raises(RuntimeError): paper.run_unit(unit,tmp_path,'identity')
        with pytest.raises(ValueError): paper.run_unit(unit,tmp_path,'identity')
        assert call.call_count==1
    saved=json.loads((tmp_path/'smoke/static'/unit['id']/'unit.json').read_text())
    assert saved['status']=='failed' and saved['returncode']==7


def test_completed_resume_requires_intact_data_and_same_code(tmp_path):
    unit=paper.units('static',smoke=True)[0]
    def backend(argv,**kwargs):
        out=Path(argv[argv.index('--outdir')+1]);out.mkdir()
        for name in ('config','observations','events','scores'):(out/(name+'.json')).write_text('{}')
        (out/'summary.json').write_text('{"status":"pilot_complete"}')
        return SimpleNamespace(returncode=0)
    with patch.object(paper.subprocess,'run',side_effect=backend) as call:
        paper.run_unit(unit,tmp_path,'identity')
        paper.run_unit(unit,tmp_path,'identity')
        assert call.call_count==1
        with pytest.raises(ValueError): paper.run_unit(unit,tmp_path,'changed-code')
        (tmp_path/'smoke/static'/unit['id']/'result/events.json').write_text('{"changed":true}')
        with pytest.raises(ValueError): paper.run_unit(unit,tmp_path,'identity')
        assert call.call_count==1


def test_scientific_settings_match_original_paper_matrices():
    # Fingerprints independently derived from all 720/200/180 original unit IDs/options.
    expected = {'static': '7448bc7426e5db07fde112e17ad1235f20483a0eb140dea05cee21f7777a7c53', 'mcbo': 'baa705f814ca543ffceea8f3eac40145cdcd046f56644f5a5a6d83b74661fe43', 'dynamic': 'cedbda111bbed0498f2cac1d17b7629688774cdacbc928d3d729ffa6a5e1edd5'}
    for suite, sha in expected.items():
        rows = sorted([{"id":u["id"],"options":u["options"]} for u in paper.units(suite)],key=lambda r:r["id"])
        assert hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest() == sha


def test_parallel_dispatch_is_bounded_and_preserves_active_completion(tmp_path):
    import threading
    import time
    selected = paper.units('static', smoke=True)[:9]
    lock = threading.Lock()
    counts = {'active':0, 'peak':0, 'finished':0}
    def backend(*args):
        with lock:
            counts['active'] += 1
            counts['peak'] = max(counts['peak'], counts['active'])
        time.sleep(.02)
        with lock:
            counts['active'] -= 1
            counts['finished'] += 1
    with patch.object(paper, 'run_unit', side_effect=backend):
        paper.run_many(selected, tmp_path, jobs=3, identity='test')
    assert counts == {'active':0, 'peak':3, 'finished':9}


def test_parallel_failure_does_not_launch_remaining_units_or_kill_peer(tmp_path):
    import threading
    import time
    selected = paper.units('static', smoke=True)[:8]
    rendezvous = threading.Barrier(2)
    started, completed = [], []
    def backend(unit, *args):
        started.append(unit['id'])
        rendezvous.wait(timeout=5)
        if unit == selected[0]:
            raise RuntimeError('deliberate failure')
        time.sleep(.05)
        completed.append(unit['id'])
    with patch.object(paper, 'run_unit', side_effect=backend):
        with pytest.raises(RuntimeError, match='deliberate failure'):
            paper.run_many(selected, tmp_path, jobs=2, identity='test')
    assert set(started) == {u['id'] for u in selected[:2]}
    assert completed == [selected[1]['id']]
    with pytest.raises(ValueError): paper.run_many(selected,tmp_path,jobs=0,identity='test')
    with pytest.raises(ValueError): paper.run_many([selected[0],selected[0]],tmp_path,jobs=2,identity='test')
