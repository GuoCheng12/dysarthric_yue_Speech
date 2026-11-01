#!/usr/bin/env python3
"""Matched fresh MAS/reference-control data and corrected preference training."""

import argparse
from pathlib import Path
import shutil
import sys

from mas_selection_dpo_probe import REPO, execute, read, rows, save, jsonl
from mas_matched_generation_control import prepare as prepare_control, verify as verify_control
from mas_ASR_target_format_probe import copy_receipts
from mas_edit_mask_dpo_probe import command, check_training, exposure_plan
from mas_residual_dpo_probe import evaluation_command
from mas_evaluation_report import compare
from report_ASR_target_format import summarize
from therapist_harness.preference_data import build_preference_pairs
from therapist_harness.training_allocation import compile_role_allocation
from therapist_harness.store import sha256_file, sha256_directory
from therapist_harness.text import text_normalize

ARMS = ['MAS_reference', 'random_reference']
SCRATCH = Path('/tmp/qwen3_asr_adaptive96_training_recovery_v2')


def control_prepare(base):
    main = base / 'adaptive_mas96_data_v1'
    root = base / 'adaptive96_random_reference_data_recovery_v2'
    assert read(main / 'new_proposal_verified.json')['status'] == 'passed'
    prepare_control(main, root)
    files = [Path(__file__), main / 'new_proposal_verified.json', main / 'data_stage_execution.json',
        REPO / 'examples/generate_reviewed_sg_recovery_v2.py', REPO / 'examples/generate_reviewed_sg_resumable.py']
    save(root / 'data_stage_execution.json', dict(source_sha256={str(f):sha256_file(f) for f in files},
        scope='Matched control generation and current-parent sampling only; no legacy training or independent MAS proposal.'))


def control_data(base):
    root = base / 'adaptive96_random_reference_data_recovery_v2'
    verify_control(root)
    binding = read(root / 'data_stage_execution.json')
    assert all(sha256_file(Path(f)) == h for f,h in binding['source_sha256'].items())
    p = read(root / 'protocol.json'); c = p['preference_loop']
    generation = [sys.executable, str(REPO / 'examples/generate_reviewed_sg_recovery_v2.py'),
        '--manifest', str(root / 'manifest.jsonl'), '--out-dir', str(root / 'generation'),
        '--generation-plan', str(root / 'generation_plan.json'), '--review-mode', 'asr_transcript', '--review-asr-worker',
        '--audio-review-model', c['audio_review_model'], '--audio-review-base-url', c['audio_review_base_url'],
        '--review-asr-model', p['base_model_dir'], '--review-asr-adapter', c['review_asr_adapter'],
        '--review-asr-adapter-sha256', c['review_asr_adapter_sha256'],
        '--review-asr-base-model-sha256', c['review_asr_base_model_sha256']]
    if c['review_allow_pending']: generation.append('--review-allow-pending')
    execute(root, generation, 'generation')
    sample = [sys.executable, str(REPO / 'examples/sample_asr_preferences.py'), 'sample',
        '--inputs', str(root / 'generation/accepted.jsonl'), '--real-train', p['real_train']['path'],
        '--feedback', p['feedback_split']['path'], '--out-dir', str(root / 'sampling'),
        '--base-model', p['base_model_dir'], '--current-adapter', p['initial_adapter']['path'],
        '--expected-parent-sha256', p['initial_adapter']['sha256'], '--samples', str(c['samples_per_audio']),
        '--seed', str(p['lora']['seed']), '--temperature', str(c['temperature']), '--top-p', str(c['top_p']),
        '--max-new-tokens', str(c['max_new_tokens']), '--include-greedy']
    execute(root, sample, 'sampling')
    verify_control(root)
    assert all(sha256_file(Path(f)) == h for f,h in binding['source_sha256'].items())
    save(root / 'data_stage_result.json', dict(status='complete', actual_DPO_training_executed=False,
        independent_sampling_receipt=str(root / 'sampling/sampling_receipt.json'),
        actual_MAS_proposal_reused=True, new_MAS_proposal_call=False))


def training_prepare(base, root):
    plan_path = base / 'adaptive96_downstream_plan_recovery_v2.json'; plan = read(plan_path)
    assert all(sha256_file(Path(f)) == h for f,h in plan['source_sha256'].items())
    roots = dict(MAS_reference=base / 'adaptive_mas96_data_v1/round_001',
                 random_reference=base / 'adaptive96_random_reference_data_recovery_v2')
    assert read(base / 'adaptive_mas96_data_v1/data_stage_result.json')['status'] == 'complete'
    assert read(roots['random_reference'] / 'data_stage_result.json')['status'] == 'complete'
    old = read(base / 'ASR_target_format_dpo_probe_v1/protocol.json')
    parent = old['source_protocol']['initial_adapter']['sha256']
    data, receipts = {}, {}
    sources = [Path(__file__), plan_path, base / 'ASR_target_format_dpo_probe_v1/protocol.json',
        base / 'adaptive_mas96_data_v1/data_stage_execution.json',
        roots['random_reference'] / 'data_stage_execution.json', roots['random_reference'] / 'provenance.json']
    for arm, directory in roots.items():
        sample = directory / 'sampling'; receipt = read(sample / 'sampling_receipt.json')
        assert receipt['status'] == 'complete' and receipt['parent_files_unchanged']
        assert receipt['current_asr_sha256'] == parent and receipt['samples_per_audio'] == 4
        for filename, field in [('candidates.jsonl','candidates_sha256'), ('validated_inputs.jsonl','validated_inputs_sha256'), ('greedy.jsonl','greedy_sha256')]:
            assert sha256_file(sample / filename) == receipt[field]
        inputs = rows(sample / 'validated_inputs.jsonl')
        assert all(sha256_file(Path(x['audio'])) == x['audio_sha256'] for x in inputs)
        rebuilt, audit = build_preference_pairs(inputs, rows(sample / 'candidates.jsonl'), parent,
            expected_samples=4, greedy_candidates=rows(sample / 'greedy.jsonl'))
        assert rebuilt == rows(sample / 'preferences.jsonl') and audit == read(sample / 'preference_audit.json')
        data[arm] = rebuilt; receipts[arm] = receipt
        sources += [sample / n for n in ['sampling_receipt.json','candidates.jsonl','validated_inputs.jsonl',
                                        'greedy.jsonl','preferences.jsonl','preference_audit.json']]
        sources += [directory / 'materialization_sidecar.json', directory / 'generation_plan.json']
    fields = ['base_model_sha256','current_asr_sha256','samples_per_audio','seed','temperature','top_p','top_k',
              'max_new_tokens','language','prompt','do_sample','real_train_sha256','feedback_sha256']
    assert all(receipts[ARMS[0]][k] == receipts[ARMS[1]][k] for k in fields)
    common = set.intersection(*[{x['audio_id'] for x in data[a]} for a in ARMS])
    assert common, 'No common actual preference opportunities; do not invent negatives.'
    root.mkdir(exist_ok=False); assert shutil.disk_usage('/tmp').free > 250 * 1024 ** 2
    SCRATCH.mkdir(exist_ok=False)
    arguments, plans, allocations = {}, {}, {}
    for arm in ARMS:
        out = root / arm; out.mkdir(); (SCRATCH / arm).mkdir()
        selected = []
        for row in data[arm]:
            if row['audio_id'] not in common: continue
            chosen = row['chosen'].rstrip('。！？!?')
            assert text_normalize(chosen) == text_normalize(row['chosen']) != text_normalize(row['rejected'])
            selected.append(dict(row, chosen=chosen, metadata=dict(row['metadata'],
                original_target_text=row['chosen'], label_transform='strip_terminal_sentence_punctuation_v2')))
        jsonl(out / 'preferences.jsonl', selected)
        allocations[arm] = compile_role_allocation(read(roots[arm] / 'materialization_sidecar.json'), selected, 12, 8, 42)
        a = dict(old['arguments']['edit_eos'], synthetic_train_file=str(out / 'preferences.jsonl'),
            synthetic_allocation_file=str(root / 'allocation.json'), output_dir=str(SCRATCH / arm / 'training'))
        plans[arm] = exposure_plan(a, rows(a['real_train_file']), selected, allocations[arm], read(a['real_loss_factors'])['factors'])
        save(out / 'expected_exposures.json', plans[arm]); arguments[arm] = a
        sources += [out / 'preferences.jsonl', out / 'expected_exposures.json']
    assert allocations[ARMS[0]] == allocations[ARMS[1]]
    assert [x for x in plans[ARMS[0]] if x['source'] == 'real'] == [x for x in plans[ARMS[1]] if x['source'] == 'real']
    occurrences = lambda records: {(x['step'],x['source'],x['occurrence'],x['audio_id']) for x in records}
    assert occurrences(plans[ARMS[0]]) == occurrences(plans[ARMS[1]])
    save(root / 'allocation.json', allocations[ARMS[0]])
    save(root / 'data_receipt.json', dict(common_logical_audio_ids=sorted(common), common_audio=len(common),
        full_pair_audio={a:len({x['audio_id'] for x in data[a]}) for a in ARMS}, full_pair_rows={a:len(data[a]) for a in ARMS},
        admitted_audio={a:receipts[a]['greedy_count'] for a in ARMS},
        independent_receipts={a:str(roots[a] / 'sampling/sampling_receipt.json') for a in ARMS},
        negative_selection='All distinct actual errors averaged within each common audio; no additional LLM negative selection.',
        chosen_encoding='Only terminal punctuation stripped; raw target and original coarse review retained.'))
    save(root / 'cpu_smoke.json', dict(status='passed', common_parent_and_decode=True,
        identical_real_and_synthetic_audio_occurrences=True, identical_real_pair_records=True,
        actual_new_GPU_smoke=False, same_engine_GPU_smoke=str(base / 'edit_mask_dpo_probe_v1/smoke/verified.json')))
    sources += [root / 'allocation.json', root / 'data_receipt.json', root / 'cpu_smoke.json']
    save(root / 'protocol.json', dict(source_protocol=old['source_protocol'], arguments=arguments,
        source_sha256={**old['source_sha256'], **{str(f):sha256_file(f) for f in sources}},
        same_engine_GPU_smoke=str(base / 'edit_mask_dpo_probe_v1/smoke/verified.json'),
        trial_plan=str(plan_path), source_data_roots={a:str(p) for a,p in roots.items()},
        actual_new_DPO_training=True, actual_new_generation=True,
        interpretation='Fresh adaptive MAS content; compare MAS-selected versus random same-patient reference/recipe on common actual preference opportunities. Same real KL retention and modified edit+EOS DPO recipe; all independent sampling provenance retained.',
        checkpoint_storage='Temporary /tmp weights, persistent workspace training receipts and evaluation. No automatic promotion.'))


def verify(root):
    cfg = read(root / 'protocol.json')
    assert all(sha256_file(Path(f)) == h for f,h in cfg['source_sha256'].items())
    return cfg


def train(base, root, arm):
    cfg = verify(root); out = root / arm; a = cfg['arguments'][arm]
    assert read(cfg['same_engine_GPU_smoke'])['status'] == 'passed'
    execute(out, command(a), 'training')
    training = Path(a['output_dir'])
    result, teacher, counts = check_training(training.parent, read(out / 'expected_exposures.json'), 'edit_eos')
    assert teacher['audio'] == read(base / 'ASR_target_format_dpo_probe_v1/edit_eos/training_receipts/real_teacher_cache.json')['audio']
    checkpoint = Path(result['checkpoint']); digest = sha256_directory(checkpoint)
    copy_receipts(training, out)
    save(out / 'verified.json', dict(status='passed', actual_exposures_match=True, all96_real_teachers_unchanged=True,
        actual_scored_token_counts=counts, checkpoint=str(checkpoint), checkpoint_sha256=digest))
    execute(out, evaluation_command(cfg['source_protocol'], out / 'evaluation_run', checkpoint), 'evaluation')
    metrics = read(out / 'evaluation_run/evaluation/summary.json')['metrics']
    parent = read(base / 'parent_eval_check/summary.json')['metrics']
    gate = dict(pooled_improves=metrics['overall']['pooled_cer'] < parent['overall']['pooled_cer'],
        macro_regression_within_limit=metrics['overall']['patient_macro_cer'] - parent['overall']['patient_macro_cer'] <= .001,
        hard_regression_within_limit=metrics['hard']['pooled_cer'] - parent['hard']['pooled_cer'] <= .005)
    save(out / 'paired_parent.json', compare(base / 'parent_eval_check/predictions.jsonl', out / 'evaluation_run/evaluation/predictions.jsonl'))
    save(out / 'results.json', dict(status='complete', metrics=metrics, gate_checks=gate,
        eligible_for_next_round_parent=all(gate.values()), checkpoint=str(checkpoint), checkpoint_sha256=digest,
        actual_new_DPO_training=True, postprocessing_used=False, persistent_checkpoint_exported=False))
    assert sha256_directory(checkpoint) == digest
    verify(root)


def report(base, root):
    verify(root)
    summaries = {a:summarize(base,root / a) for a in ARMS}
    records = [rows(root / a / 'training_receipts/training_exposures.jsonl') for a in ARMS]
    assert [x for x in records[0] if x['source'] == 'real'] == [x for x in records[1] if x['source'] == 'real']
    initial = [read(root / a / 'training_receipts/training_result.json')['policy_trainable_sha256_before'] for a in ARMS]
    assert initial[0] == initial[1]
    paired = compare(root / 'random_reference/evaluation_run/evaluation/predictions.jsonl', root / 'MAS_reference/evaluation_run/evaluation/predictions.jsonl')
    save(root / 'paired_reference_policy.json', paired)
    save(root / 'results.json', dict(status='complete', arms=summaries, parent_promoted=False,
        MAS_minus_random_CER_pp=paired['groups']['overall']['delta_pp'],
        bootstrap95_pp=paired['groups']['overall']['patient_cluster_bootstrap_delta_pp_95'],
        actual_new_DPO_training=True, actual_new_generation=True, postprocessing_used=False,
        source_sha256={str(root / a / 'results.json'):sha256_file(root / a / 'results.json') for a in ARMS}))
    print(summaries, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['control_prepare','control_data','training_prepare','train','report'])
    parser.add_argument('base', type=Path); parser.add_argument('--arm', choices=ARMS, default='MAS_reference')
    args = parser.parse_args(); base = args.base.resolve(); root = base / 'adaptive96_matched_DPO_recovery_v2'
    if args.stage == 'control_prepare': control_prepare(base)
    elif args.stage == 'control_data': control_data(base)
    elif args.stage == 'training_prepare': training_prepare(base,root)
    elif args.stage == 'train': train(base,root,args.arm)
    else: report(base,root)
