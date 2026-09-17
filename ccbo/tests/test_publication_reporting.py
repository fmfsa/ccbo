"""Required publication rows and archival provenance must not disappear silently."""
import tempfile
from pathlib import Path
import unittest
import pandas as pd
from scripts.publication_methods import (PRIMARY_GROUPS, HISTORICAL_GROUPS,
    publication_name, required_files, validate_summary)
from scripts.emit_cost_analysis import validate_trace_costs

class PublicationReportingTests(unittest.TestCase):
    def test_complete_primary_inventory_and_missing_a2(self):
        groups = {'_'.join(key): {'n': 30} for key in PRIMARY_GROUPS}
        validate_summary(groups)
        self.assertNotIn(('MediatedChain', 'C0', 'HQCBO'), PRIMARY_GROUPS)
        self.assertIn(('MediatedChain', 'C0', 'HQCBO'), HISTORICAL_GROUPS)
        del groups['ParallelParent_A2_CBO']
        with self.assertRaisesRegex(ValueError, 'ParallelParent_A2_CBO'):
            validate_summary(groups)

    def test_seed_count_and_exact_seed_inventory(self):
        groups = {'_'.join(key): {'n': 30} for key in PRIMARY_GROUPS}
        groups['ParallelParent_A2_CBO']['n'] = 29
        with self.assertRaisesRegex(ValueError, 'expected 30 seeds'):
            validate_summary(groups)
        with tempfile.TemporaryDirectory() as directory:
            for seed in (0, 2):
                Path(directory, f'unit_seed{seed}.csv').touch()
            with self.assertRaisesRegex(ValueError, 'missing=.*seed1'):
                required_files(directory, 'unit', 2)

    def test_publication_names_keep_archival_variant_unambiguous(self):
        self.assertEqual(publication_name('HQCBOGF'), 'HQCBO')
        self.assertIn('historical', publication_name('HQCBO'))
        self.assertEqual(publication_name('BOS'), 'BO-S')
        self.assertEqual(publication_name('CBONP'), 'CBO−np')

    def test_split_cost_is_charged_once_at_trigger(self):
        rows = [{'trial': 0, 'arm': 'init', 'cum_cost': 6},
                {'trial': 1, 'arm': 'X1+X2', 'cum_cost': 14},
                {'trial': 2, 'arm': '', 'cum_cost': 14}]
        df = pd.DataFrame(rows)
        log = {'rows': rows, 'refine': {'trigger_step': 1, 'split_init_cost': 6}}
        validate_trace_costs(df, log)
        log['refine']['split_init_cost'] = 12
        with self.assertRaisesRegex(ValueError, 'do not reconcile'):
            validate_trace_costs(df, log)

if __name__ == '__main__':
    unittest.main()
