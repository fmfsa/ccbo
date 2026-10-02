#!/usr/bin/env python3
"""Run the paper's static, ClusterChain, model-based and dynamic experimental matrices."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).resolve().parent / 'configs'
COMMANDS = {
    'static': ['scripts/run_matched_controlled.py'],
    'clusterchain': ['scripts/run_matched_controlled.py'],
    'mcbo': ['scripts/run_mcbo.py'],
    'dynamic': ['-m', 'ccbo.qdcbo.runner'],
}


def load_config(suite):
    return json.loads((CONFIGS / (suite + '.json')).read_text())


def units(suite, seeds=None, smoke=False):
    config = load_config(suite)
    selected = list(seeds) if seeds is not None else ([1000] if smoke else config['seeds'])
    allowed = range(1000, 1005) if smoke else config['seeds']
    if not selected or len(selected) != len(set(selected)) or any(s not in allowed for s in selected):
        raise ValueError('Use distinct paper seeds (2000 onwards), or pilot seeds 1000–1004 with --smoke')
    result = []
    for case in config['cases']:
        for seed in selected:
            options = dict(config['defaults'], **case['options'])
            options['--seed'] = str(seed)
            if smoke:
                if suite in ('static', 'clusterchain'):
                    options.update({'--stage': 'pilot', '--max-purchases': '3'})
                elif suite == 'mcbo':
                    options['--num-trials'] = '2'
                else:
                    options['--trials'] = '3'
            result.append({'id': case['id'].format(seed=seed), 'suite': suite,
                           'mode': 'smoke' if smoke else 'paper', 'seed': seed,
                           'options': options})
    if len({u['id'] for u in result}) != len(result):
        raise ValueError('Duplicate configuration IDs')
    return result


def command(unit, folder):
    argv = [sys.executable, *COMMANDS[unit['suite']]]
    for key, value in unit['options'].items():
        argv.extend([key, value])
    return argv + ['--outdir', str(folder / 'result')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_identity():
    """Detect mixed local implementations without storing source snapshots."""
    files = [p for p in (ROOT / 'ccbo').rglob('*.py') if 'tests' not in p.parts]
    files += list((ROOT / 'scripts').glob('*.py'))
    files += list(CONFIGS.glob('*.json'))
    files += [Path(__file__).resolve()]
    # Include fetched upstream implementations; absent vendors are diagnosed by their backend.
    for vendor in ('DCBO', 'mcbo'):
        path = ROOT / 'third_party' / vendor
        if path.exists():
            files += list(path.rglob('*.py'))
    content = '\n'.join(str(p.relative_to(ROOT)) + ':' + digest(p) for p in sorted(set(files)))
    return hashlib.sha256(content.encode()).hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def output_hashes(folder, suite):
    result = folder / 'result'
    if suite in ('static', 'clusterchain'):
        expected = [result / (name + '.json') for name in ('config', 'observations', 'events', 'scores', 'summary')]
        summary = json.loads((result / 'summary.json').read_text())
        if summary.get('status') not in ('pilot_complete', 'budget_complete'):
            raise ValueError('Static backend did not complete')
    else:
        expected = list(result.glob('*.csv')) + list(result.glob('*.decisions.json')) + list(result.glob('*_info.json'))
        if len(expected) != 3:
            raise ValueError('Expected one CSV, decision ledger and info file')
    if any(not p.is_file() or not p.stat().st_size for p in expected):
        raise ValueError('Missing/empty backend output')
    return {str(p.relative_to(folder)): digest(p) for p in sorted(expected)}


def run_unit(unit, outdir, identity=None):
    identity = identity or code_identity()
    # Pilot and paper results cannot collide even under the same output directory.
    folder = Path(outdir).resolve() / unit['mode'] / unit['suite'] / unit['id']
    status_path = folder / 'unit.json'
    if folder.exists():
        status = json.loads(status_path.read_text())
        if status.get('status') != 'complete' or status.get('unit') != unit or status.get('code_identity') != identity:
            raise ValueError('Existing incomplete or incompatible unit; preserve it and use a new output directory')
        if status['output_sha256'] != output_hashes(folder, unit['suite']):
            raise ValueError('Existing completed outputs changed')
        return status
    folder.mkdir(parents=True, exist_ok=False)
    status = {'schema': 1, 'unit': unit, 'code_identity': identity,
              'status': 'running', 'started_unix': time.time()}
    save(status_path, status)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED='0', CCBO_GIT_SHA=identity)
    env.update({name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    try:
        result = subprocess.run(command(unit, folder), cwd=ROOT, env=env, check=False)
        status['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Backend exited with code ' + str(result.returncode))
        status['output_sha256'] = output_hashes(folder, unit['suite'])
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        status['finished_unix'] = time.time()
        save(status_path, status)
    return status


def run_many(selected, outdir, jobs=1, identity=None):
    """Bound dispatch; stop launching on failure and let active units finish."""
    if jobs < 1:
        raise ValueError('--jobs must be positive')
    if len({u['id'] for u in selected}) != len(selected):
        raise ValueError('Duplicate unit IDs')
    identity = identity or code_identity()
    stopped = threading.Event()
    iterator = iter(enumerate(selected, 1))

    def execute(index, unit):
        if stopped.is_set():
            return
        print(f"[{index}/{len(selected)}] {unit['id']} ({unit['mode']})", flush=True)
        try:
            return run_unit(unit, outdir, identity)
        except BaseException:
            stopped.set()
            raise

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        active = set()
        first_error = None
        while True:
            while len(active) < jobs and not stopped.is_set():
                item = next(iterator, None)
                if item is None:
                    break
                active.add(pool.submit(execute, *item))
            if not active:
                break
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except BaseException as error:
                    stopped.set()
                    first_error = first_error or error
            # Never kill active scientific processes: their completion/failure
            # records remain valid even when a peer has failed.
        if first_error is not None:
            raise first_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=COMMANDS, required=True)
    parser.add_argument('--seed', type=int, action='append', help='Repeat for selected declared seeds')
    parser.add_argument('--unit', help='Exact ID from --list')
    parser.add_argument('--index', type=int, help='Zero-based position in the selected matrix')
    parser.add_argument('--smoke', action='store_true', help='Pilot seeds and 2–3 decisions; never paper results')
    parser.add_argument('--list', action='store_true', help='List IDs/options without importing a backend')
    parser.add_argument('--jobs', type=int, default=1, help='Concurrent independent units; one CPU thread each')
    parser.add_argument('--outdir', type=Path, default=Path('results/paper'))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    selected = units(args.suite, args.seed, args.smoke)
    if args.unit:
        selected = [u for u in selected if u['id'] == args.unit]
        if not selected:
            parser.error('Unit not in selected matrix')
    if args.index is not None:
        if not 0 <= args.index < len(selected):
            parser.error('Index outside selected matrix')
        selected = [selected[args.index]]
    if args.list:
        print(json.dumps(selected, indent=2))
        return
    run_many(selected, args.outdir, args.jobs)


if __name__ == '__main__':
    main()
