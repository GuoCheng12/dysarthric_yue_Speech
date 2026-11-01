#!/usr/bin/env python3
"""DPO after actual MAS reference selection, with honest source composition."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mas_selection_dpo_probe import REPO, read, rows, save, jsonl
from mas_round2_reference_pilot_v1 import verify as verify_pilot
from mas_exploratory_round2_dpo_v1 import allocation, exposure_plan, match_exposures, train, verify
from mas_evaluation_report import compare
from therapist_harness.preference_data import build_preference_pairs
from therapist_harness.store import sha256_file
from therapist_harness.text import text_normalize

ARMS = ['original_reference', 'MAS_reference']
SCRATCH = Path('/tmp/qwen3_asr_round2_reference_DPO_v1')


def normalize(pairs, common):
    result = []
    for row in pairs:
        if row['audio_id'] not in common: continue
        chosen = row['chosen'].rstrip('。！？!?')
        assert text_normalize(chosen) == text_normalize(row['chosen']) != text_normalize(row['rejected'])
        result.append(dict(row, chosen=chosen, metadata=dict(row['metadata'], original_target_text=row['chosen'],
            label_transform='strip_terminal_sentence_punctuation_v2')))
    return result


def prepare(base, root):
    pilot = base / 'round2_reference_selection_probe_v1'; plan = verify_pilot(pilot)
    selected = read(pilot / 'selection_receipt.json'); assert selected['status'] == 'complete'
    assert all(sha256_file(Path(f)) == h for f, h in selected['source_sha256'].items())
    source = Path(plan['original_data_root']); previous = Path(plan['previous_DPO_root'])
    prior = verify(previous); cached = previous / 'adaptive_MAS'
    selection = {x['audio_id']: x['option'] for x in read(pilot / 'MAS_selection/selection.json')['choices']}
    files = [Path(__file__), Path(__file__).with_name('mas_exploratory_round2_dpo_v1.py'),
        pilot / 'plan.json', pilot / 'selection_receipt.json', pilot / 'data_complete.json',
        previous / 'protocol.json', previous / 'results.json', cached / 'results.json', cached / 'verified.json',
        cached / 'preferences.jsonl', cached / 'expected_exposures.json', cached / 'training_receipt_copies.json',
        cached / 'training_receipts/training_exposures.jsonl']
    streams = {}; receipts = {}
    for name, directory in [('original', source), ('alternate', pilot)]:
        sample = directory / 'sampling'; receipt = read(sample / 'sampling_receipt.json'); receipts[name] = receipt
        assert receipt['status'] == 'complete' and receipt['parent_files_unchanged']
        assert receipt['current_asr_sha256'] == plan['current_parent_sha256']
        assert receipt['initial_R_only_sha256'] == prior['source_protocol']['initial_adapter']['sha256']
        assert receipt['learned_residual_loaded'] and not receipt['current_residual_merged']
        for filename, field in [('validated_inputs.jsonl', 'validated_inputs_sha256'), ('candidates.jsonl', 'candidates_sha256'), ('greedy.jsonl', 'greedy_sha256')]:
            assert sha256_file(sample / filename) == receipt[field]
        inputs = rows(sample / 'validated_inputs.jsonl'); candidates = rows(sample / 'candidates.jsonl'); greedy = rows(sample / 'greedy.jsonl')
        for x in inputs:
            assert sha256_file(Path(x['audio'])) == x['audio_sha256'] == x['review']['candidate_audio_sha256']
            ref = Path(x['review']['transcription_receipt_path'])
            assert sha256_file(ref) == x['review']['transcription_receipt_sha256']; files.append(ref)
        pairs, audit = build_preference_pairs(inputs, candidates, plan['current_parent_sha256'], expected_samples=4, greedy_candidates=greedy)
        assert pairs == rows(sample / 'preferences.jsonl') and audit == read(sample / 'preference_audit.json')
        streams[name] = dict(inputs={x['audio_id']: x for x in inputs}, candidates={x['sample_id']: x for x in candidates},
            greedy={x['audio_id']: x for x in greedy}, pairs=pairs, receipt=str(sample / 'sampling_receipt.json'))
        files += [sample / n for n in ['sampling_receipt.json', 'validated_inputs.jsonl', 'candidates.jsonl', 'greedy.jsonl', 'preferences.jsonl', 'preference_audit.json']]
    fields = ['base_model_sha256', 'current_asr_sha256', 'initial_R_only_sha256', 'temperature', 'top_p', 'top_k',
        'seed', 'samples_per_audio', 'max_new_tokens', 'language', 'prompt', 'greedy_config', 'real_train_sha256', 'feedback_sha256']
    assert all(receipts['original'][k] == receipts['alternate'][k] for k in fields)
    composed_inputs, composed_candidates, composed_greedy, provenance = [], [], [], []
    for key in streams['original']['inputs']:
        option = selection.get(key, 'original'); stream = streams[option]; x = stream['inputs'][key]
        composed_inputs.append(x); composed_greedy.append(stream['greedy'][key])
        candidates = [stream['candidates'][f'{key}::{i}'] for i in range(4)]; composed_candidates += candidates
        if option == 'alternate':
            old = streams['original']['inputs'][key]
            assert old['clean_gt'] == x['clean_gt'] and old['audio_sha256'] != x['audio_sha256']
            for y in candidates: assert y['seed'] == streams['original']['candidates'][y['sample_id']]['seed']
        provenance.append(dict(audio_id=key, selected_source=option, source_sampling_receipt=stream['receipt'],
            source_sampling_receipt_sha256=sha256_file(Path(stream['receipt'])), audio_sha256=x['audio_sha256'],
            candidate_sample_ids=[v['sample_id'] for v in candidates],
            current_asr_sha256=plan['current_parent_sha256'], reference_wav_sha256=x['reference_wav_sha256']))
    selected_pairs, audit = build_preference_pairs(composed_inputs, composed_candidates, plan['current_parent_sha256'],
        expected_samples=4, greedy_candidates=composed_greedy)
    raw = dict(original_reference=streams['original']['pairs'], MAS_reference=selected_pairs)
    common = set.intersection(*[{x['audio_id'] for x in raw[a]} for a in ARMS]); assert common
    scheduled = allocation(read(source / 'slots.json'), common, read(source / 'role_policy.json'))
    root.mkdir(exist_ok=False); SCRATCH.mkdir(exist_ok=False)
    jsonl(root / 'selected_inputs.jsonl', composed_inputs); jsonl(root / 'selected_candidates.jsonl', composed_candidates)
    jsonl(root / 'selected_greedy.jsonl', composed_greedy); save(root / 'selected_preference_audit.json', audit)
    save(root / 'source_composition.json', dict(schema_version='composed_sampling_subset_v1', actual_new_joint_sampling=False,
        entries=provenance, sources={name: data['receipt'] for name, data in streams.items()},
        scope='Exact per-audio selection from two independent actual current-parent sampling receipts. Reused originals and newly sampled alternate waveforms are not represented as one joint sampling job.'))
    save(root / 'allocation.json', scheduled)
    save(root / 'evaluation_target_descriptor.json', read(previous / 'evaluation_target_descriptor.json'))
    arguments, expected, normalized = {}, {}, {}
    for arm in ARMS:
        out = root / arm; out.mkdir(); normalized[arm] = normalize(raw[arm], common)
        jsonl(out / 'preferences.jsonl', normalized[arm])
        a = dict(prior['arguments']['adaptive_MAS'], synthetic_train_file=str(out / 'preferences.jsonl'),
            synthetic_allocation_file=str(root / 'allocation.json'), output_dir=str(SCRATCH / arm / 'training'))
        arguments[arm] = a
        expected[arm] = exposure_plan(a, rows(a['real_train_file']), normalized[arm], scheduled, read(a['real_loss_factors'])['factors'])
        save(out / 'expected_exposures.json', expected[arm]); files += [out / 'preferences.jsonl', out / 'expected_exposures.json']
    match_exposures(*[expected[a] for a in ARMS])
    reusable = normalized['original_reference'] == rows(cached / 'preferences.jsonl')
    reusable = reusable and expected['original_reference'] == rows(cached / 'training_receipts/training_exposures.jsonl')
    ignore = {'synthetic_train_file', 'synthetic_allocation_file', 'output_dir'}
    assert {k: v for k, v in arguments['original_reference'].items() if k not in ignore} == {k: v for k, v in prior['arguments']['adaptive_MAS'].items() if k not in ignore}
    result_roots = dict(original_reference=str(cached if reusable else root / 'original_reference'))
    identical = normalized['MAS_reference'] == normalized['original_reference'] and expected['MAS_reference'] == expected['original_reference']
    result_roots['MAS_reference'] = result_roots['original_reference'] if identical else str(root / 'MAS_reference')
    save(root / 'reuse_decisions.json', dict(original_control_reused=reusable, MAS_selected_data_identical_to_control=identical,
        result_roots=result_roots, basis='Compare full normalized pair records, actual exposures/order/masses, starting parent and every training argument except artifact paths. Reuse is an existing actual training/evaluation result, not a new run.'))
    save(root / 'cpu_smoke.json', dict(status='passed', two_source_preferences_rebuilt=True, exact_per_audio_sampling_provenance=True,
        common_parent_decoding_and_seed=True, identical_real_and_synthetic_exposures=True,
        warm_GPU_smoke=prior['same_engine_GPU_smoke']))
    save(root / 'data_receipt.json', dict(common_audio=len(common), common_logical_slots=sorted(common),
        selected_changed_references=sum(x == 'alternate' for x in selection.values()),
        changed_references_contributing_to_DPO=sum(selection.get(k) == 'alternate' for k in common),
        pair_rows={a: len(normalized[a]) for a in ARMS}, final_generation={name:[{k:x[k] for k in ['sample_id','attempt','generation_mode','effective_instruction_id','reference_wav_sha256','output_wav_sha256','admitted']} for x in read(directory/'generation/generation_summary.json')['outputs'] if x['sample_id'] in selection] for name,directory in [('original',source),('alternate',pilot)]},
        matching='Same target text/patient and initial generation recipe/seed. Actual mode or seed changes from the unchanged retry rulebook remain in the final-generation receipt. Common actual paired slots have identical role exposures.'))
    files += list(root.glob('*.json')) + list(root.glob('*.jsonl'))
    save(root / 'protocol.json', dict(plan=prior['plan'], source_protocol=prior['source_protocol'], arguments=arguments,
        previous_policy_trainable_sha256=prior['previous_policy_trainable_sha256'], same_engine_GPU_smoke=prior['same_engine_GPU_smoke'],
        source_sha256={**prior['source_sha256'], **plan['source_sha256'], **selected['source_sha256'], **{str(f): sha256_file(f) for f in files}},
        result_roots=result_roots, original_ledgers_unchanged=True, parent_promoted=False,
        actual_new_joint_sampling=False, scope='Within-patient reference selection after actual pilot outputs. All-correct audio may be selected but contributes no invented negative. Control reuse only after exact input/config/exposure checks.'))
    verify(root); print(dict(status='prepared', common_audio=len(common), control_reused=reusable, identical=identical,
        changed_training_references=read(root / 'data_receipt.json')['changed_references_contributing_to_DPO']), flush=True)


def report(base, root):
    cfg = verify(root); paths = {a: Path(p) for a, p in cfg['result_roots'].items()}
    results = {a: read(p / 'results.json') for a, p in paths.items()}
    assert all(x['status'] == 'complete' for x in results.values())
    for arm, path in paths.items():
        assert rows(path / 'training_receipts/training_exposures.jsonl') == read(root / arm / 'expected_exposures.json')
        assert read(path / 'training_receipts/training_result.json')['policy_trainable_sha256_before'] == cfg['previous_policy_trainable_sha256']
    match_exposures(*[rows(paths[a] / 'training_receipts/training_exposures.jsonl') for a in ARMS])
    paired = compare(paths[ARMS[0]] / 'evaluation_run/evaluation/predictions.jsonl', paths[ARMS[1]] / 'evaluation_run/evaluation/predictions.jsonl')
    save(root / 'paired_reference_selection.json', paired)
    targets = {a: read(p / 'current_target_response.json')['monitored_cluster_changes'][0] for a, p in paths.items()}
    save(root / 'results.json', dict(status='complete', arms=results, result_sources={a: str(p) for a, p in paths.items()},
        MAS_minus_original_CER_pp=paired['groups']['overall']['delta_pp'],
        bootstrap95_pp=paired['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        target_changes=targets, reuse_decisions=read(root / 'reuse_decisions.json'),
        actual_new_training_runs=len({p for p in paths.values() if p.parent == root}),
        source_composition_receipt=str(root / 'source_composition.json'), parent_promoted=False,
        scope='Repeated511 development feedback, selected seed42 parent, no postprocessing. Coarse accepted target labels are not word-verified audio.'))
    lines = ['保持内容和患者，比较原 reference 与 MAS 根据试生成转写选择的 reference。', '',
        '| 条件 | CER % | 字符错误 | macro CER % | hard 错误 | 織删除 |', '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = results[arm]['metrics']
        lines.append(f'| {arm} | {100*m["overall"]["pooled_cer"]:.6f} | {m["overall"]["total_edits"]} | {100*m["overall"]["patient_macro_cer"]:.6f} | {m["hard"]["total_edits"]} | {targets[arm]["after"]["error_count"]} |')
    lines += ['', '两个分支均以1373错的本轮共同 parent 为起点，DPO/KL teacher 固定为初始 R-only。',
        '复用记录、实际新训练次数及两份采样来源见 results.json、reuse_decisions.json、source_composition.json。',
        '本助手与 MAS 使用三文本和真实 ASR 采样记录，没有实际听辨或逐字音频保真证明。']
    with (root / 'results.zh.md').open('x') as handle: handle.write('\n'.join(lines) + '\n')
    print({a: dict(errors=x['metrics']['overall']['total_edits'], CER_percent=100*x['metrics']['overall']['pooled_cer']) for a, x in results.items()}, flush=True)


def supervise(base, root):
    cfg = verify(root); processes = []; launched = set()
    for gpu, arm in enumerate(ARMS):
        out = Path(cfg['result_roots'][arm])
        if out.parent != root or out in launched: continue
        launched.add(out); actual_arm = out.name
        command = [sys.executable, str(Path(__file__).resolve()), 'train', str(base), '--arm', actual_arm]
        with (root / (actual_arm + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        save(root / (actual_arm + '.launch.json'), dict(pid=child.pid, gpu=gpu, command=command)); processes.append(child)
    codes = [p.wait() for p in processes]; assert all(x == 0 for x in codes), codes
    report(base, root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare', 'train', 'report', 'supervise', 'all']); parser.add_argument('base', type=Path)
    parser.add_argument('--arm', choices=ARMS, default=ARMS[0]); args = parser.parse_args()
    base = args.base.resolve(); root = base / 'round2_reference_selection_DPO_v1'
    if args.stage == 'all':
        pilot = base / 'round2_reference_selection_probe_v1'; launch = read(base / 'round2_reference_selection_execution_v1/supervisor_launch.json')
        while not (pilot / 'selection_receipt.json').exists():
            assert Path(f'/proc/{launch["pid"]}').exists(), 'Reference pilot ended before producing a validated MAS selection.'
            time.sleep(10)
        prepare(base, root); supervise(base, root)
    elif args.stage == 'prepare': prepare(base, root)
    elif args.stage == 'train': train(base, root, args.arm)
    elif args.stage == 'report': report(base, root)
    else: supervise(base, root)
