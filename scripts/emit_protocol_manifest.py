"""Extract actual per-run settings and observation schedules from archived evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('artifacts/summaries/protocol_manifest.json'))
    args = parser.parse_args()
    groups = defaultdict(list)
    units = []
    for suite in ('minimal', 'family_cbo', 'qdcbo', 'qmcbo'):
        paths = sorted((args.results_dir / suite).rglob('*.decisions.json'))
        if not paths:
            raise ValueError(f'missing evidence for {suite}')
        for p in paths:
            d = json.loads(p.read_text())
            u = d['unit']
            row = {'source': str(p.relative_to(args.results_dir)),
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'unit': u}
            if suite in ('minimal', 'family_cbo'):
                events = [t for phase in d['phases'] for t in phase['trials']]
                types = [t['type'] for t in events]
                if len(events) != u['trials']:
                    raise ValueError(f'{p}: event count differs from declared trials')
                if set(types) - {'observe', 'intervene'}:
                    raise ValueError(f'{p}: unknown event type')
                row['schedule'] = {
                    'decision_rounds': len(events),
                    'observation_rounds': types.count('observe'),
                    'sequential_interventions': types.count('intervene'),
                    'new_observational_rows': sum(t['n_obs_after']-t['n_obs_before']
                                                  for t in events if t['type']=='observe'),
                    'initial_cost': d['init']['init_cost'],
                    'split_initial_cost': (d.get('refine') or {}).get('split_init_cost', 0),
                    'initial_exploration_set': d['init']['es'],
                }
                key = '|'.join([suite, u.get('scm', u.get('dataset', '')), u['cond'], u['arm']])
                groups[key].append(row)
            units.append(row)
    ceo = []
    for p in sorted((args.results_dir/'ceo_minimal').glob('*.meta.json')):
        d=json.loads(p.read_text())
        keys=('scm','cond','seed','trials','ninit','n_obs','pool_pids','es','init_cost',
              'num_anchor_points','objective','alias_of')
        ceo.append({'source':str(p.relative_to(args.results_dir)),
                    'settings':{k:d[k] for k in keys if k in d}})
    if not ceo:
        raise ValueError('missing CEO metadata')
    summaries={}
    for key, rows in sorted(groups.items()):
        fields=('decision_rounds','observation_rounds','sequential_interventions','new_observational_rows',
                'initial_cost','split_initial_cost')
        summaries[key]={'seeds':sorted(r['unit']['seed'] for r in rows),
                        'n':len(rows),
                        'ranges':{f:[min(r['schedule'][f] or 0 for r in rows),
                                     max(r['schedule'][f] or 0 for r in rows)] for f in fields}}
    result={'schema':1,'source':'archived sidecars; no optimizer rerun',
            'schedule_groups':summaries, 'units':units, 'ceo':ceo}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(f'{len(units)} decision logs, {len(ceo)} CEO metadata records; wrote {args.out}')


if __name__=='__main__':
    main()
