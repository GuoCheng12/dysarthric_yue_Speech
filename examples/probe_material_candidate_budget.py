"""One real selector call with a wider catalog and an unchanged recorded diagnosis."""
import argparse
from collections import Counter
from pathlib import Path
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.material_selection_offline import (
    MaterialSelection, build_selector_request, prepare_candidates, validate_selection,
)
from therapist_harness.real_sampling_mas import _call
from therapist_harness.schema import AgentConfig
from therapist_harness.laicai_credentials import load_laicai_key
from therapist_harness.store import sha256_file, write_immutable_json


def read(path):
    return json.loads(path.read_text())


def main(args):
    stage = args.source_stage
    assert stage.name == 'stage-001', 'This bounded probe uses a fresh first-round diagnosis.'
    protocol = read(stage.parents[1] / 'protocol.json')
    index_path = Path(protocol['material_curriculum']['index_path'])
    index = read(index_path)
    original = read(stage / 'decision/candidates.json')
    assert not original['exposure_available']
    assert read(stage / 'decision/diagnosis/receipt.json')['validated']
    assert prepare_candidates(index, original['diagnosis'], {}, 24) == original
    args.out.mkdir(parents=True, exist_ok=False)
    pack = prepare_candidates(index, original['diagnosis'], {}, args.limit)
    write_immutable_json(args.out / 'candidates.json', pack)
    config = AgentConfig.model_validate(protocol['agent'])
    write_immutable_json(args.out / 'protocol.json', dict(probe='candidate_budget_only',
        candidate_limit=args.limit, source_stage=str(stage), agent=config.model_dump(mode='json'),
        source_sha256={str(p): sha256_file(p) for p in
            [index_path, stage / 'decision/candidates.json', stage / 'decision/diagnosis/decision.json',
             Path(__file__), REPO / 'src/therapist_harness/material_selection_offline.py',
             REPO / 'src/therapist_harness/real_sampling_mas.py']},
        training_changed=False, new_diagnosis_generated=False))
    load_laicai_key()
    selection, sources, seconds = _call(config, phase='select', context={},
        directory=args.out / 'selection', request=build_selector_request(pack),
        output_type=MaterialSelection, role='Material Selection Agent',
        validate=lambda output: validate_selection(index, pack, output.model_dump()))
    result = validate_selection(index, pack, selection.model_dump())
    write_immutable_json(args.out / 'selected_materials.json', result)
    previous = read(stage / 'decision/selected_materials.json')
    def summary(item):
        return {k: item[k] for k in ('unique_recordings', 'patients', 'texts')} | {
            'difficulty_counts': dict(Counter(r['initial_difficulty'] for r in item['recordings']))}
    write_immutable_json(args.out / 'result.json', dict(passed=True, actual_model=config.model,
        old_selection=summary(previous), expanded_selection=summary(result), seconds=seconds,
        sources=sources, training_connected=False,
        interpretation='Actual selector behavior under a wider displayed catalog; not a training or performance result.'))
    print(json.dumps(read(args.out / 'result.json'), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-stage', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=48)
    parser.add_argument('--out', required=True, type=Path)
    main(parser.parse_args())
