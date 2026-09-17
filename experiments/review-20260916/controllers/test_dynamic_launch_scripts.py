"""No scheduler calls: immutable guards, array mapping and deferred gate."""
import json
import importlib.util
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
BASE='/zhome/15/b/215295/Repositories/ccbo-experiments-20260916'
class Tests(unittest.TestCase):
    def test_syntax_resources(self):
        for name in ('dynamic_pilot.lsf','dynamic_campaign.lsf','dynamic_launch_gate.lsf','dynamic_final_analysis.lsf','submit_dynamic_campaign.sh'):
            subprocess.run(['bash','-n',str(HERE/name)],check=True)
        text=(HERE/'dynamic_campaign.lsf').read_text()
        for token in ('[1-180]%2','#BSUB -W 02:00','mem=8GB','dynamic-final-preflight','dynamic-final-campaign'):
            self.assertIn(token,text)
    def test_unverified_analyzer_never_imported(self):
        spec=importlib.util.spec_from_file_location('gate',HERE/'check_dynamic_launch_gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir()
            sentinel=root/'EXECUTED'
            code="from pathlib import Path\nPath("+repr(str(sentinel))+").touch()\n"
            analyzer=root/'scripts/summarize_dynamic_campaign.py';analyzer.write_text(code)
            manifest={'source_sha256':{'scripts/summarize_dynamic_campaign.py':'0'*64}}
            for kind in ('pilot','final'):(root/(kind+'-manifest.json')).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'manifest'):gate.check(root,root/'results')
            self.assertFalse(sentinel.exists())
            gate.IDS={k:hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest() for k in ('pilot','final')}
            with self.assertRaisesRegex(ValueError,'source mismatch'):gate.check(root,root/'results')
            self.assertFalse(sentinel.exists())
    def shim(self,index,code=0,states='RUN'):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);base=Path(temp.name)
        launch=base/'server-launch';launch.mkdir();scripts=base/'releases/dynamic-population-final-v2/scripts';scripts.mkdir(parents=True)
        (launch/'record_dynamic_campaign_stop.py').write_text((HERE/'record_dynamic_campaign_stop.py').read_text())
        (launch/'check_dynamic_launch_gate.py').write_text("print('fixture gate')")
        calls=base/'calls.json'
        (scripts/'run_frozen_unit.py').write_text('import json,sys\nfrom pathlib import Path\nPath('+repr(str(calls))+').write_text(json.dumps(sys.argv[1:]))\nraise SystemExit('+str(code)+')\n')
        bindir=base/'bin';bindir.mkdir();bjobs=bindir/'bjobs';bjobs.write_text('#!/bin/bash\necho '+states+'\n');bjobs.chmod(0o755)
        script=base/'unit.sh';script.write_text((HERE/'dynamic_campaign.lsf').read_text().replace(BASE,str(base)).replace('/zhome/15/b/215295/venvs/ccbo/bin/python',sys.executable))
        env=dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH'],LSB_JOBINDEX=str(index),LSB_JOBID='999')
        run=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
        return base,script,env,run,calls
    def test_boundaries(self):
        for index in (1,180):
            _,_,_,run,calls=self.shim(index);self.assertEqual(run.returncode,0,run.stderr)
            args=json.loads(calls.read_text());self.assertEqual(args[args.index('--index')+1],str(index-1));self.assertEqual(args[args.index('--manifest')+1],'final-manifest.json')
    def test_first_failure_atomic_stop(self):
        base,script,env,run,calls=self.shim(1,17);self.assertEqual(run.returncode,17)
        stop=base/'results/dynamic-final-campaign/STOP.json';first=stop.read_bytes();before=calls.read_bytes()
        run=subprocess.run(['bash',str(script)],env=dict(env,LSB_JOBINDEX='2'),capture_output=True,text=True)
        self.assertEqual(run.returncode,99);self.assertEqual(first,stop.read_bytes());self.assertEqual(before,calls.read_bytes())
    def test_scheduler_peer_veto(self):
        base,_,_,run,calls=self.shim(2,states='EXIT');self.assertEqual(run.returncode,98);self.assertFalse(calls.exists());self.assertTrue((base/'results/dynamic-final-campaign/STOP.json').exists())
    def test_deferred_submission_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);launch=base/'server-launch';launch.mkdir();bindir=base/'bin';bindir.mkdir()
            for name in ('dynamic_launch_gate.lsf','dynamic_campaign.lsf','dynamic_final_analysis.lsf'):
                (launch/name).write_text((HERE/name).read_text())
            log=base/'scheduler.jsonl';shim=bindir/'bsub'
            shim.write_text('#!'+sys.executable+'\nimport json,sys\nfrom pathlib import Path\np=Path('+repr(str(log))+')\nrows=p.read_text().splitlines() if p.exists() else []\nrows.append(json.dumps(sys.argv[1:]));p.write_text("\\n".join(rows)+"\\n")\nsys.stdin.read()\nprint("Job <"+str(900+len(rows))+"> is submitted")\n');shim.chmod(0o755)
            script=base/'submit.sh';script.write_text((HERE/'submit_dynamic_campaign.sh').read_text().replace(BASE,str(base)).replace('/zhome/15/b/215295/venvs/ccbo/bin/python',sys.executable))
            env=dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH'])
            run=subprocess.run(['bash',str(script),'--after-job','29424849'],env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual([json.loads(x) for x in log.read_text().splitlines()],[['-w','numended(29424849, == 9)'],['-w','done(901)'],['-w','numended(902, == 180)']])
            run=subprocess.run(['bash',str(script),'--after-job','29424849'],env=env,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0);self.assertEqual(len(log.read_text().splitlines()),3)
if __name__=='__main__':unittest.main()
