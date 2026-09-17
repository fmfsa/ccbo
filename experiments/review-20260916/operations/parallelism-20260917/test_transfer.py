import importlib.util,json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('transfer',Path(__file__).with_name('transfer.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_scheduler_parser_and_failure_guard(self):
  with patch.object(m,'command',return_value='PSUSP ccbo_test[61]\nRUN ccbo_test[62]\n'):
   self.assertEqual(m.states('1',Path('.')),{61:'PSUSP',62:'RUN'})
  for state in ('EXIT','UNKWN','ZOMBI'):
   with self.assertRaises(ValueError):m.fail_states({61:state})
  with patch.object(m,'command',return_value=''):
   with self.assertRaises(ValueError):m.states('1',Path('.'))
 def test_completed_validation_never_executes_science_and_rejects_drift(self):
  with tempfile.TemporaryDirectory() as d,patch.object(m,'BASE',Path(d)):
   c=m.SETTINGS['static'];release=Path(d)/'releases'/c['release'];release.mkdir(parents=True)
   source=release/'source.py';source.write_text('pass\n')
   unit={'id':'example','outputs':['result.json']}
   manifest={'schema':1,'source_sha256':{'source.py':m.sha(source)},'units':[unit]}
   (release/'replication-manifest.json').write_text(json.dumps(manifest))
   _,ident,_,folder=m.manifest_unit('static',c,1);folder.mkdir(parents=True)
   output=folder/'result.json';output.write_text('{}')
   status={'status':'complete','manifest_id':ident,'unit':unit,'output_sha256':{'result.json':m.sha(output)}}
   (folder/'execution.json').write_text(json.dumps(status))
   with patch.object(m.subprocess,'run',side_effect=AssertionError('No executable calls allowed')):
    m.validate_complete('static',c,[1])
    status['status']='running';(folder/'execution.json').write_text(json.dumps(status))
    with self.assertRaisesRegex(ValueError,'Incomplete'):m.validate_complete('static',c,[1])
    status['status']='complete';(folder/'execution.json').write_text(json.dumps(status));output.write_text('{"changed":true}')
    with self.assertRaisesRegex(ValueError,'hash mismatch'):m.validate_complete('static',c,[1])
    source.write_text('changed')
    with self.assertRaisesRegex(ValueError,'source drift'):m.validate_complete('static',c,[1])
 def test_finalizer_failure_never_resumes(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'plan.json';path.write_text(json.dumps({'kind':'mcbo','settings':m.SETTINGS['mcbo'],'supplement_job':'2','transferred':[61]}))
   with patch.object(m,'stop_path',return_value=Path(d)/'STOP.json'),patch.object(m,'array_counts',return_value={'NJOBS':1,'DONE':0,'EXIT':1}),patch.object(m,'states',return_value={61:'EXIT'}),patch.object(m,'command') as command:
    with self.assertRaises(ValueError):m.finalize(path)
    command.assert_not_called()
   self.assertEqual(json.loads(path.read_text())['status'],'blocked-finalization')
if __name__=='__main__':unittest.main()
