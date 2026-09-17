"""All final drivers must reject changed manifests/sources before execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
HERE=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_all_final_drivers_reject_before_import(self):
        for family in ('dynamic','mcbo','matched'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                spec=importlib.util.spec_from_file_location('driver_'+family,HERE/f'run_{family}_final_analysis.py')
                driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
                root=Path(directory);(root/'scripts').mkdir();sentinel=root/'EXECUTED'
                relative=f'scripts/summarize_{family}_campaign.py'
                (root/relative).write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').touch()\n')
                manifest={'source_sha256':{relative:'0'*64}}
                for key in driver.EXPECTED_MANIFEST_IDS:
                    (root/(key+'-manifest.json')).write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'manifest'):driver.verified_analyzer(root)
                self.assertFalse(sentinel.exists())
                canonical=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                driver.EXPECTED_MANIFEST_IDS={key:canonical for key in driver.EXPECTED_MANIFEST_IDS}
                with self.assertRaisesRegex(ValueError,'source mismatch'):driver.verified_analyzer(root)
                self.assertFalse(sentinel.exists())

if __name__=='__main__':unittest.main()
