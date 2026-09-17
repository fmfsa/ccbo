"""Hash the reporting source, archived inputs and generated paper assets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(paths):
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths) if p.is_file()}


def main():
    source = []
    for directory in ('ccbo', 'scripts', 'envs'):
        source.extend(p for p in (ROOT/directory).rglob('*')
                      if p.suffix in {'.py', '.sh', '.txt', '.toml', '.lsf'})
    source += [ROOT/'pyproject.toml', ROOT/'uv.lock']
    source_hashes = hashes(source)
    reporting_id = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    inputs = hashes((ROOT/'artifacts/traces').glob('*'))
    asset_paths = list((ROOT/'paper/figures').glob('*.pdf')) + list((ROOT/'paper/figures').glob('*.tex'))
    asset_paths += list((ROOT/'paper/figures').glob('*.provenance.json'))
    asset_paths += list((ROOT/'paper/tables').glob('*.tex'))
    asset_paths += list((ROOT/'paper/revisions').glob('*'))
    summaries = [p for p in (ROOT/'artifacts/summaries').glob('*.json') if p.name != 'revision_manifest.json']
    payload = {
        'schema': 1,
        'reporting_source_id': 'sha256:'+reporting_id,
        'meaning': 'Content fingerprint of the listed source/environment files, not a Git commit or public release.',
        'engine_revisions': {
            'minimal_and_family_cbo': 'f75a19c3bd2cff4d59916e2d40f8ba43a791bf58',
            'qdcbo_and_qmcbo': '9f25b415ff90502533cf3b013c0a10fd14571781'},
        'source_sha256': source_hashes,
        'archive_and_provenance_sha256': inputs,
        'paper_asset_sha256': hashes(asset_paths),
        'summary_sha256': hashes(summaries),
        'regeneration_command': 'PYTHON=/path/to/python bash scripts/reproduce_results.sh /path/to/scratch',
        'scope': 'Archived scientific evidence regenerated; optimizers are not rerun by this command.',
        'main_manuscript': 'Not present; current Overleaf integration and final render remain pending.',
    }
    out = ROOT/'artifacts/summaries/revision_manifest.json'
    out.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    print(f'wrote {out.relative_to(ROOT)}; {payload["reporting_source_id"]}')


if __name__ == '__main__':
    main()
