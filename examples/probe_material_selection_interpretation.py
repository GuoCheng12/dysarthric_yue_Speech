"""Paired selector probe: clarify difficulty/exposure meaning without quotas."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.laicai_credentials import load_laicai_key
from therapist_harness.material_selection_offline import (
    MaterialSelection, build_selector_request, prepare_candidates, validate_selection,
)
from therapist_harness.real_sampling_mas import _call
from therapist_harness.schema import AgentConfig
from therapist_harness.store import sha256_file, write_immutable_json


CLARIFICATION = """
选择材料时还要区分下面几件事：
初始识别难度是原始模型在该录音上的表现。单凭难度高，不能把录音或标注判为不可靠；难度低也不等于更能纠正当前错误。
prior_exposures 只记录抽到几次，不是学习效果或过拟合程度。已见过的相关材料仍可继续练习，不要只追求未见材料。
先考虑这些录音怎样帮助完成诊断目标，再考虑不同患者、句子、识别难度和已知曝光的覆盖。容易与困难材料都可以有用途，请在简短理由中说明取舍。
不用机械平分难度，不必选满列表，也不规定每类必须选多少；是否选择仍由你根据本项目标和候选决定。
"""


def main(args):
    read = lambda path: json.loads(path.read_text())
    stage = args.source_stage.resolve()
    protocol_path = stage.parents[1] / 'protocol.json'
    protocol = read(protocol_path)
    index_path = Path(protocol['material_curriculum']['index_path'])
    index = read(index_path)
    source_pack = read(stage / 'decision/candidates.json')
    assert read(stage / 'decision/diagnosis/receipt.json')['validated']
    counts = Counter()
    source_paths = [protocol_path, index_path, stage / 'decision/candidates.json',
                    stage / 'decision/diagnosis/decision.json']
    for previous in sorted(stage.parent.glob('stage-*')):
        if previous.name >= stage.name:
            continue
        path = previous / 'actual_exposure.json'
        counts.update(read(path)['utterance_ids'])
        source_paths.append(path)
    pack = prepare_candidates(index, source_pack['diagnosis'], dict(counts), source_pack['candidate_limit'])
    # Both arms use accurate prior counts, identical diagnosis, and identical candidates.
    for old, new in zip(source_pack['focuses'], pack['focuses'], strict=True):
        assert old['focus_id'] == new['focus_id']
        assert [r['recording_id'] for r in old['candidates']] == [r['recording_id'] for r in new['candidates']]
    request = build_selector_request(pack)
    clarified = deepcopy(request)
    clarified['input'][0]['content'] += '\n' + CLARIFICATION
    assert request['input'][1:] == clarified['input'][1:] and request['text'] == clarified['text']
    config = AgentConfig.model_validate(protocol['agent'])
    group_path = Path(protocol['groups_path'])
    membership = {uid: group for group, rows in read(group_path).items() for uid in rows['train_ids']}
    source_paths += [group_path, Path(__file__),
                     REPO / 'src/therapist_harness/material_selection_offline.py',
                     REPO / 'src/therapist_harness/prompts/material_selection_offline.md',
                     REPO / 'src/therapist_harness/real_sampling_mas.py']
    args.out.mkdir(parents=True, exist_ok=False)
    write_immutable_json(args.out / 'candidates.json', pack)
    write_immutable_json(args.out / 'protocol.json', dict(
        source_stage=str(stage), agent=config.model_dump(mode='json'), prior_total_exposures=sum(counts.values()),
        source_sha256={str(p): sha256_file(p) for p in source_paths},
        change='System prompt clarification only; same corrected exposure metadata in both arms.',
        training_connected=False, dev_test_transcripts_given_to_selector=False,
        performance_acceptance_changed=False, new_diagnosis_generated=False))
    load_laicai_key()

    def run(name, payload):
        selection, sources, seconds = _call(config, phase='select', context={},
            directory=args.out / name, request=payload, output_type=MaterialSelection,
            role='Material Selection Agent',
            validate=lambda output: validate_selection(index, pack, output.model_dump()))
        selected = validate_selection(index, pack, selection.model_dump())
        write_immutable_json(args.out / name / 'selected_materials.json', selected)
        return dict(seconds=seconds, sources=sources,
            **{key: selected[key] for key in ('unique_recordings', 'patients', 'texts')},
            group_counts=dict(Counter(membership[r['recording_id']] for r in selected['recordings'])),
            prior_exposure_counts=dict(Counter(counts[r['recording_id']] for r in selected['recordings'])),
            choices=selected['choices'])

    # The API deadline uses SIGALRM, so each call must execute on the main thread.
    results = {name: run(name, payload)
               for name, payload in [('baseline', request), ('clarified', clarified)]}
    result = dict(passed=True, actual_model=config.model, arms=results,
        training_connected=False, interpretation=(
            'One paired stochastic selector probe, not evidence of ASR improvement or a causal '
            'estimate of the prompt effect. Both arms use the same train candidates and accurate '
            'past exposure counts; only the system clarification differs.'))
    write_immutable_json(args.out / 'result.json', result)
    print(json.dumps({name: {k: v for k, v in arm.items() if k not in ('sources', 'choices')}
                      for name, arm in results.items()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-stage', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
