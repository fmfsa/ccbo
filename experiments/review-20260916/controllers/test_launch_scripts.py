"""No scheduler calls: syntax, indices, atomic STOP, and local failing-shim tests."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
BASE="/zhome/15/b/215295/Repositories/ccbo-experiments-20260916"
CAMPAIGN="989629b3691bc2b0a015eff4beee55e5dbb24a287692401acf66f87605bc23d4"


def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+".py"));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class LaunchTests(unittest.TestCase):
    def test_shell_syntax_and_exact_array_mapping(self):
        for name in ("matched_replication.lsf","matched_final_analysis.lsf","matched_launch_gate.lsf","submit_matched_campaign.sh"):
            subprocess.run(["bash","-n",str(HERE/name)],check=True)
        text=(HERE/"matched_replication.lsf").read_text()
        self.assertIn("[1-720]%4",text);self.assertIn("#BSUB -W 08:00",text)
        self.assertIn("#BSUB -n 1",text);self.assertIn('rusage[mem=6GB]',text)
        self.assertIn('$((LSB_JOBINDEX-1))',text)
        manifest=json.loads((HERE.parent/"releases/matched-preflight-v3/replication-manifest.json").read_text())
        mapped=[manifest["units"][i-1]["id"] for i in range(1,721)]
        self.assertEqual(len(mapped),len(set(mapped)));self.assertEqual(len(mapped),720)
        self.assertIn('numended($array_id, == 720)',(HERE/"submit_matched_campaign.sh").read_text())

    def test_first_failure_marker_is_atomic_and_immutable(self):
        m=module("record_campaign_stop")
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"STOP.json"
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                won=list(ex.map(lambda i:m.record(path,dict(index=i,status="stopped")),range(20)))
            self.assertEqual(sum(won),1)
            before=path.read_bytes();self.assertIn(json.loads(before)["index"],range(20))
            self.assertFalse(m.record(path,dict(index=999)))
            self.assertEqual(before,path.read_bytes())
            self.assertEqual(list(Path(d).glob(".STOP-*")),[])

    def test_failure_stops_next_queued_task_without_overwriting(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);launch=base/"server-launch";launch.mkdir()
            release=base/"releases/matched-preflight-v3";scripts=release/"scripts";scripts.mkdir(parents=True)
            (launch/"record_campaign_stop.py").write_text((HERE/"record_campaign_stop.py").read_text())
            (launch/"check_matched_launch_gate.py").write_text("print('test gate passed')\n")
            sentinel=base/"calls.txt"
            (scripts/"run_frozen_unit.py").write_text("from pathlib import Path\np=Path("+repr(str(sentinel))+")\np.write_text(p.read_text()+'run\\n' if p.exists() else 'run\\n')\nraise SystemExit(17)\n")
            script=(HERE/"matched_replication.lsf").read_text().replace(BASE,str(base)).replace("/zhome/15/b/215295/venvs/ccbo/bin/python",sys.executable)
            task=base/"unit.sh";task.write_text(script)
            bin_dir=base/"bin";bin_dir.mkdir()
            bjobs=bin_dir/"bjobs";bjobs.write_text("#!/bin/bash\nprintf 'RUN\\nPEND\\n'\n");bjobs.chmod(0o755)
            env=dict(os.environ,LSB_JOBINDEX="1",LSB_JOBID="987",PATH=str(bin_dir)+os.pathsep+os.environ["PATH"])
            first=subprocess.run(["bash",str(task)],env=env,capture_output=True,text=True)
            self.assertEqual(first.returncode,17,first.stderr)
            stop=base/"results/matched-replication"/CAMPAIGN/"STOP.json"
            saved=stop.read_bytes();record=json.loads(saved)
            self.assertEqual(record["array_index"],"1");self.assertEqual(record["exit_code"],17)
            second=subprocess.run(["bash",str(task)],env=dict(env,LSB_JOBINDEX="2"),capture_output=True,text=True)
            self.assertEqual(second.returncode,99)
            self.assertEqual(sentinel.read_text(),"run\n")
            self.assertEqual(stop.read_bytes(),saved)

    def test_scheduler_peer_failure_blocks_unit_and_records_stop(self):
        for state, expected in (("RUN\nEXIT",98),("UNKWN",98),("ZOMBI",98),("",2)):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as d:
                base=Path(d);launch=base/"server-launch";launch.mkdir()
                scripts=base/"releases/matched-preflight-v3/scripts";scripts.mkdir(parents=True)
                (launch/"record_campaign_stop.py").write_text((HERE/"record_campaign_stop.py").read_text())
                (launch/"check_matched_launch_gate.py").write_text("print('test gate passed')\n")
                sentinel=base/"unit-ran"
                (scripts/"run_frozen_unit.py").write_text("from pathlib import Path\nPath("+repr(str(sentinel))+").touch()\n")
                bin_dir=base/"bin";bin_dir.mkdir()
                args=base/"scheduler-args"
                bjobs=bin_dir/"bjobs"
                bjobs.write_text("#!"+sys.executable+"\nimport os,sys\nfrom pathlib import Path\nPath("+repr(str(args))+").write_text(' '.join(sys.argv[1:]))\nprint(os.environ['TEST_STATES'])\n")
                bjobs.chmod(0o755)
                script=(HERE/"matched_replication.lsf").read_text().replace(BASE,str(base)).replace("/zhome/15/b/215295/venvs/ccbo/bin/python",sys.executable)
                task=base/"unit.sh";task.write_text(script)
                env=dict(os.environ,LSB_JOBINDEX="7",LSB_JOBID="987",TEST_STATES=state,PATH=str(bin_dir)+os.pathsep+os.environ["PATH"])
                result=subprocess.run(["bash",str(task)],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,expected,result.stderr)
                self.assertEqual(args.read_text(),"-a -noheader -o stat 987")
                self.assertFalse(sentinel.exists())
                stop=json.loads((base/"results/matched-replication"/CAMPAIGN/"STOP.json").read_text())
                self.assertEqual(stop["array_index"],"7")
                self.assertEqual(stop["exit_code"],expected)

    def test_resource_phase_gate_uses_all_timings_without_matrix_selection(self):
        gate=module("check_matched_launch_gate")
        rows=[dict(method=m,wall_seconds=t) for m,t in zip(("CBO","QCBO","HQCBO","CBO-FALLBACK","CEO"),(41,22,22,17,7200))]
        result=gate.resource_projection(rows)
        self.assertAlmostEqual(result["projected_cpu_hours"],187.175)
        self.assertEqual(result["queue_limits"]["array_concurrency"],4)
        with self.assertRaisesRegex(ValueError,"all five"):
            gate.resource_projection(rows[:-1])
        rows[-1]["wall_seconds"]=20000
        with self.assertRaisesRegex(ValueError,"phase cap exceeded"):
            gate.resource_projection(rows)

    def test_runtime_versions_and_timing_values_fail_closed(self):
        gate=module("check_matched_launch_gate")
        provenance=gate.runtime_provenance(dict(gate.PRINCIPAL_VERSIONS),"3.10.18")
        self.assertEqual(provenance["python"],"3.10.18")
        with self.assertRaisesRegex(ValueError,"Python3.10"):
            gate.runtime_provenance(dict(gate.PRINCIPAL_VERSIONS),"3.11.0")
        wrong=dict(gate.PRINCIPAL_VERSIONS);wrong["GPy"]="0.0"
        with self.assertRaisesRegex(ValueError,"package version"):
            gate.runtime_provenance(wrong,"3.10.20")
        for value in (0.,-1.,float("nan"),float("inf")):
            rows=[dict(method=m,wall_seconds=1.) for m in ("CBO","QCBO","HQCBO","CBO-FALLBACK","CEO")]
            rows[-1]["wall_seconds"]=value
            with self.assertRaisesRegex(ValueError,"positive and finite"):
                gate.resource_projection(rows)
        text=(HERE/"submit_matched_campaign.sh").read_text()
        self.assertIn('--after-job',text)
        self.assertIn('gate_args=(-w "numended($after_job, == 5)")',text)
        self.assertIn('bsub -w "done($gate_id)"',text)

    def test_stop_marker_vetoes_even_apparent_complete_analysis(self):
        m=module("run_matched_final_analysis")
        audit=dict(failures=[],failure_count=0,failure_counts_by_status={},inference_status="complete",comparisons=["not published"],seed_level_metrics=["not published"])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"STOP.json";path.write_text('{}')
            result=m.apply_stop_veto(audit,path)
            self.assertEqual(result["inference_status"],"blocked")
            self.assertEqual(result["comparisons"],[])
            self.assertNotIn("seed_level_metrics",result)
            self.assertEqual(result["failure_counts_by_status"],{"stopped":1})

if __name__=="__main__":unittest.main()
