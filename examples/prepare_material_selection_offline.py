"""Build/search the offline pool; never call an API or start a trainer."""
import argparse
import hashlib
import json
from pathlib import Path

from therapist_harness.material_selection_offline import (
    DiagnosticBrief, MaterialSelection, build_index, inventory, prepare_candidates,
    build_selector_request, validate_selection,
)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('index')
    b.add_argument('--train', required=True)
    b.add_argument('--exclude', nargs='+', required=True)
    b.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('query')
    q.add_argument('--index', required=True)
    q.add_argument('--diagnosis', required=True)
    q.add_argument('--exposures', help='JSON mapping of train recording ID to cumulative exposure count')
    q.add_argument('--limit', type=int, default=24)
    q.add_argument('--out', type=Path, required=True)
    v = sub.add_parser('validate')
    v.add_argument('--index', required=True)
    v.add_argument('--candidates', required=True)
    v.add_argument('--selection', required=True)
    v.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'index':
        sources = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                   for p in [args.train, *args.exclude]}
        data = build_index(rows(args.train), [r for p in args.exclude for r in rows(p)], sources)
        write(args.out / 'index.json', data)
        write(args.out / 'inventory.json', inventory(data))
        write(args.out / 'diagnostic_brief.schema.json', DiagnosticBrief.model_json_schema())
        write(args.out / 'material_selection.schema.json', MaterialSelection.model_json_schema())
        summary = dict(recordings=len(data['recordings']), texts=len(data['profiles']),
                       patients=len({r['patient_id'] for r in data['recordings']}),
                       index_sha256=data['index_sha256'], training_connected=False,
                       pronunciation_unresolved_characters=sum(p['coverage']['unresolved_count']
                                                               for p in data['profiles'].values()))
        write(args.out / 'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False))
    elif args.command == 'query':
        pack = prepare_candidates(read(args.index), read(args.diagnosis),
                                  read(args.exposures) if args.exposures else None, args.limit)
        write(args.out / 'candidates.json', pack)
        write(args.out / 'selector_request.json', build_selector_request(pack))
        print(json.dumps([dict(focus_id=p['focus_id'], eligible=p['eligible_count'],
                               displayed=len(p['candidates'])) for p in pack['focuses']]))
    else:
        write(args.out, validate_selection(read(args.index), read(args.candidates), read(args.selection)))


if __name__ == '__main__':
    main()
