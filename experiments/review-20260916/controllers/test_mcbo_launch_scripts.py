"""No scheduler calls: boundaries, resource gate, atomic failure, queued veto."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
BASE='/zhome/15/b/215295/Repositories/ccbo-experiments-20260916'
def module(name):
    s=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Tests(unittest.TestCase):
    def test_syntax_and_concurrency(self):
        for name in ('mcbo_campaign.lsf','mcbo_final_analysis.lsf','mcbo_launch_gate.lsf','submit_mcbo_campaign.sh'):
            subprocess.run(['bash','-n',str(HERE/name)],check=True)
        text=(HERE/'mcbo_campaign.lsf').read_text()
        self.assertIn('[1-200]%2',text);self.assertIn('#BSUB -W 24:00',text);self.assertIn('mem=8GB',text)
        self.assertIn('numended($array_id, == 200)',(HERE/'submit_mcbo_campaign.sh').read_text())
    def test_projection_all_configs_and_cap(self):
        gate=module('check_mcbo_launch_gate')
        rows=[dict(env=e,algo=a,menu=m,wall_seconds=360) for e in ('ToyGraph','PSAGraph') for a,m in (('MCBO','full'),('MCBO','coarse'),('QMCBO','coarse'))]
        self.assertEqual(gate.projection(rows)['total_cpu_hours'],100)
        with self.assertRaises(ValueError):gate.projection(rows[:-1])
        rows[0]['wall_seconds']=36000
        with self.assertRaisesRegex(ValueError,'500CPUh'):gate.projection(rows)
    def run_shim(self,index,returncode=0,states='RUN'):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);base=Path(temp.name)
        launch=base/'server-launch';launch.mkdir();scripts=base/'releases/mcbo-campaign-v3/scripts';scripts.mkdir(parents=True)
        (launch/'record_mcbo_campaign_stop.py').write_text((HERE/'record_mcbo_campaign_stop.py').read_text())
        (launch/'check_mcbo_launch_gate.py').write_text("print('fixture gate')")
        calls=base/'calls.json'
        (scripts/'run_frozen_unit.py').write_text('import json,sys\nfrom pathlib import Path\nPath('+repr(str(calls))+').write_text(json.dumps(sys.argv[1:]))\nraise SystemExit('+str(returncode)+')\n')
        bindir=base/'bin';bindir.mkdir();fake=bindir/'bjobs';fake.write_text('#!/bin/bash\necho '+states+'\n');fake.chmod(0o755)
        shell=(HERE/'mcbo_campaign.lsf').read_text().replace(BASE,str(base)).replace('/zhome/15/b/215295/venvs/mcbo/bin/python',sys.executable)
        script=base/'unit.sh';script.write_text(shell)
        env=dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH'],LSB_JOBINDEX=str(index),LSB_JOBID='987')
        run=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
        return base,script,env,run,calls
    def test_boundary_mapping(self):
        for index,manifest,unit in ((1,'main',0),(120,'main',119),(121,'protected',0),(200,'protected',79)):
            base,script,env,run,calls=self.run_shim(index)
            self.assertEqual(run.returncode,0,run.stderr);argv=json.loads(calls.read_text())
            self.assertEqual(argv[argv.index('--manifest')+1],manifest+'-manifest.json')
            self.assertEqual(argv[argv.index('--index')+1],str(unit))
    def test_first_failure_stops_peer_and_preserves_first_marker(self):
        base,script,env,run,calls=self.run_shim(121,17)
        self.assertEqual(run.returncode,17,run.stderr)
        stop=base/'results/mcbo-campaign-v3/STOP.json';first=stop.read_bytes();before=calls.read_bytes()
        run=subprocess.run(['bash',str(script)],env=dict(env,LSB_JOBINDEX='122'),capture_output=True,text=True)
        self.assertEqual(run.returncode,99);self.assertEqual(stop.read_bytes(),first);self.assertEqual(calls.read_bytes(),before)
    def test_scheduler_killed_peer_blocks_without_science(self):
        base,script,env,run,calls=self.run_shim(2,states='EXIT')
        self.assertEqual(run.returncode,98);self.assertFalse(calls.exists());self.assertTrue((base/'results/mcbo-campaign-v3/STOP.json').exists())
    def test_after_job_queues_gate_without_premature_preflight_read(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);launch=base/'server-launch';launch.mkdir();bindir=base/'bin';bindir.mkdir()
            for name in ('mcbo_launch_gate.lsf','mcbo_campaign.lsf','mcbo_final_analysis.lsf'):
                (launch/name).write_text((HERE/name).read_text())
            log=base/'scheduler.jsonl'
            shim=bindir/'bsub'
            shim.write_text('#!'+sys.executable+'\nimport json,sys\nfrom pathlib import Path\np=Path('+repr(str(log))+')\nrows=p.read_text().splitlines() if p.exists() else []\nrows.append(json.dumps(sys.argv[1:]));p.write_text("\\n".join(rows)+"\\n")\nsys.stdin.read()\nprint("Job <"+str(900+len(rows))+"> is submitted")\n')
            shim.chmod(0o755)
            script=base/'submit.sh';script.write_text((HERE/'submit_mcbo_campaign.sh').read_text().replace(BASE,str(base)).replace('/zhome/15/b/215295/venvs/mcbo/bin/python',sys.executable))
            env=dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH'])
            run=subprocess.run(['bash',str(script),'--after-job','29424826'],env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            calls=[json.loads(x) for x in log.read_text().splitlines()]
            self.assertEqual(calls,[['-w','numended(29424826, == 6)'],['-w','done(901)'],['-w','numended(902, == 200)']])
            again=subprocess.run(['bash',str(script),'--after-job','29424826'],env=env,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0);self.assertEqual(len(log.read_text().splitlines()),3)
if __name__=='__main__':unittest.main()
