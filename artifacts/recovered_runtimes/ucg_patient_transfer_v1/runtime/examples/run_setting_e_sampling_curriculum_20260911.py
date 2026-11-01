"""Fresh audio-LoRA SFT: ordinary shuffle versus MAS data allocation.

The optimizer, scheduler and model stay alive across decisions.  Both branches
use the same full global batches.  A sweep is meaningful only for ordinary
shuffle; MAS stages are fixed numbers of updates, not complete data passes.
"""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import shutil
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.sampling_curriculum_runtime import ExposureSampler, exposure_summary


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def event(directory, stage, **fields):
    from datetime import datetime, timezone
    record = dict(utc=datetime.now(timezone.utc).isoformat(), stage=stage, **fields)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def stage_lengths(protocol, *, smoke=False):
    if smoke:
        return [2, 2]
    interval = protocol['steps_per_stage']
    total = protocol.get('total_optimizer_steps_limit', protocol['stages'] * interval)
    assert interval > 0 and total > 0
    lengths = [min(interval, total - start) for start in range(0, total, interval)]
    assert len(lengths) == protocol['stages'], 'stage count must cover exactly the optimizer-step budget'
    return lengths


def load_inputs(protocol):
    train, feedback, test = [rows(protocol[key]) for key in
                             ('train_manifest', 'feedback_manifest', 'test_manifest')]
    groups = read(protocol['groups_path'])
    sets = [set(row['utt_id'] for row in part) for part in (train, feedback, test)]
    assert all(len(ids) == len(part) > 0 for ids, part in zip(sets, (train, feedback, test)))
    assert protocol['setting'] in ('D', 'E') and protocol['feedback_role'] == 'dev'
    assert all(not sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    assert all(row['prompt'] == '' for part in (train, feedback, test) for row in part)
    patients = [{row['speaker_id'] for row in part} for part in (train, feedback, test)]
    from therapist_harness.sampling_curriculum_data import validate_train_feedback_split
    from therapist_harness.text import text_normalize
    validate_train_feedback_split(train, feedback, protocol['setting'])
    validate_train_feedback_split(train, test, protocol['setting'])
    texts = [{text_normalize(r['clean_gt']) for r in part} for part in (train, feedback, test)]
    if protocol['setting'] == 'E':
        assert not patients[1] & patients[2]
        assert texts[0] == texts[1] == texts[2]
    else:
        assert all(not texts[i] & texts[j] for i,j in ((0,1),(0,2),(1,2)))
    assert all(len(part) == protocol[key] for part,key in
               ((train, 'train_audio'), (feedback, 'dev_audio'), (test, 'test_audio')))
    assert set().union(*(set(group['train_ids']) for group in groups.values())) == sets[0]
    assert sum(len(group['train_ids']) for group in groups.values()) == len(train)
    assert protocol['seed'] == 42 and protocol['global_batch'] == 32
    assert protocol['stages'] > 0 and protocol['steps_per_stage'] > 0
    if protocol.get('material_curriculum'):
        assert protocol['material_curriculum']['sampling_mode'] == 'group_then_material'
        from therapist_harness.material_selection_offline import _check_index
        index = read(protocol['material_curriculum']['index_path'])
        _check_index(index)
        assert {r['recording_id'] for r in index['recordings']} == sets[0]
        assert {r['audio'] for r in index['recordings']} == {r['audio'] for r in train}
        by_id = {r['utt_id']: r for r in train}
        assert all(r['patient_id'] == by_id[r['recording_id']]['speaker_id'] and
                   index['profiles'][r['text_id']]['target_text'] == by_id[r['recording_id']]['clean_gt']
                   for r in index['recordings'])
    if protocol.get('ucg_material_policy'):
        assert protocol['setting'] == 'D', 'UCG material adaptation requires text-disjoint Setting D'
    stage_lengths(protocol)
    return train, feedback, test, groups


def import_resume(protocol, out):
    """Import a verified prefix into a NEW execution; keep the source untouched."""
    checkpoint = Path(protocol['resume_from'])
    source = checkpoint.parent.parent
    proof = read(checkpoint / 'provenance.json')
    assert sha(source / 'preparation.json') == proof['preparation_sha256']
    for name, key in [('adapter_model.safetensors', 'adapter_sha256'),
                      ('training_state.pt', 'training_state_sha256')]:
        assert sha(checkpoint / name) == proof[key]
    previous_protocol = read(source.parent / 'protocol.json')
    for key in ('setting', 'seed', 'global_batch', 'training', 'base_model', 'stages',
                'steps_per_stage', 'total_optimizer_steps_limit', 'input_sha256'):
        assert protocol[key] == previous_protocol[key], f'resume changed {key}'
    for key in ('model', 'base_url', 'reasoning_effort', 'max_output_tokens'):
        assert protocol['agent'][key] == previous_protocol['agent'][key]
    assert protocol.get('material_curriculum') == previous_protocol.get('material_curriculum')
    step = proof['step']
    files = [source.parent / 'protocol.json', source / 'initialization.json',
             source / 'preparation.json', *checkpoint.glob('*')]
    original_events = rows(source / 'events.jsonl')
    prefix = []
    for record in original_events:
        prefix.append(record)
        if record['stage'] in ('stage_complete', 'checkpoint_saved') and record.get('step') == step:
            break
    else:
        raise ValueError('checkpoint has no corresponding completed training boundary')
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'events.jsonl').exists()
    (out / 'events.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in prefix))
    # Evaluation receipts remain genuine old receipts, with explicit import hashes.
    for evaluation in (source / 'evaluations').iterdir():
        if int(evaluation.name.removeprefix('step-')) <= step:
            shutil.copytree(evaluation, out / 'evaluations' / evaluation.name)
            files.extend(evaluation.rglob('*'))
    for stage in sorted(source.glob('stage-*')):
        if int(stage.name.removeprefix('stage-')) <= int(checkpoint.parent.name.removeprefix('stage-')):
            files.extend(stage.glob('reflection/**/*.json'))
            files.extend(stage.glob('actual_exposure.json'))
    for name in ('history.json', 'selected_checkpoint.json', 'events.jsonl'):
        if (source / name).exists():
            files.append(source / name)
    replay = protocol.get('replay_stage_directory')
    if replay:
        files.extend(Path(replay).glob('decision/**/*.json'))
        files.extend(Path(replay).glob('evidence.json'))
        files.extend(Path(replay).glob('sampling_plan.json'))
        end = step + stage_lengths(protocol)[int(checkpoint.parent.name.removeprefix('stage-'))]
        files.extend((source / f'evaluations/step-{end}').rglob('*'))
    hashes = {str(path): sha(path) for path in files if path.is_file()}
    save(out / 'resume_import.json', dict(source_branch=str(source), checkpoint=str(checkpoint),
         imported_through_step=step, prefix_events=len(prefix), input_sha256=hashes,
         note='Old receipts retain their original paths and times; subsequent events belong to this execution.'))
    return hashes


def prepare(args):
    protocol = read(args.protocol)
    train, feedback, test, groups = load_inputs(protocol)
    out = Path(protocol['root']) / args.branch
    assert not (out / 'preparation.json').exists(), 'Use a new execution directory.'
    paths = [Path(args.protocol), *(Path(protocol[key]) for key in
             ('train_manifest', 'feedback_manifest', 'test_manifest', 'groups_path'))]
    resume_hashes = import_resume(protocol, out) if protocol.get('resume_from') else {}
    sources = [Path(__file__), REPO / 'src/therapist_harness/sampling_curriculum_runtime.py',
               REPO / 'examples/run_fresh_setting_e_audio_lr2em4_4gpu_50epochs_20260911.py',
               REPO / 'examples/run_fresh_lora_sft_efgh_20260910.py',
               REPO / 'finetune/scripts/qwen3_asr_lora_sft.py',
               REPO / 'finetune/official/qwen3_asr_sft.py', REPO / 'src/therapist_harness/text.py',
               REPO / 'src/therapist_harness/sampling_curriculum_data.py',
               REPO / 'src/therapist_harness/real_sampling_mas.py']
    sources.extend(REPO / 'src/therapist_harness' / name for name in
                   ('agent.py', 'schema.py', 'store.py', 'phonology.py',
                    'laicai_credentials.py', 'curriculum_v2_training.py', 'sampling_phonology.py'))
    if protocol.get('material_curriculum'):
        paths.append(Path(protocol['material_curriculum']['index_path']))
        sources.extend(REPO / 'src/therapist_harness' / name for name in
                       ('material_sampling_mas.py', 'material_selection_offline.py'))
        sources.append(REPO / 'src/therapist_harness/prompts/material_selection_offline.md')
        if protocol.get('ucg_material_policy'):
            sources.append(REPO / 'src/therapist_harness/ucg_material_context.py')
    for path in (REPO / 'src/therapist_harness/prompts').glob('*sampling*'):
        sources.append(path)
    for path in sources:
        destination = out / 'sources' / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    save(out / 'preparation.json', dict(branch=args.branch, train_audio=len(train),
         feedback_audio=len(feedback), test_audio=len(test), groups=list(groups),
         input_sha256={**resume_hashes, **{str(path.resolve()): sha(path) for path in paths}},
         source_sha256={str(path.resolve()): sha(path) for path in sources},
         global_batch=32, stage_lengths=stage_lengths(protocol),
         expected_exposures=sum(stage_lengths(protocol)) * 32))
    event(out, 'prepared', branch=args.branch, train_audio=len(train), feedback_audio=len(feedback),
          test_audio=len(test), stages=protocol['stages'], steps_per_stage=protocol['steps_per_stage'])


def cpu_smoke(args):
    protocol = read(args.protocol)
    train, feedback, test, groups = load_inputs(protocol)
    ids = [row['utt_id'] for row in train]
    memberships = {key: group['train_ids'] for key, group in groups.items()}
    sampler = ExposureSampler(ids, memberships)
    actual = sampler.draw(len(ids) // 2) + sampler.draw(len(ids) - len(ids) // 2)
    assert len(actual) == len(set(actual)) == len(ids)
    weights = {key: 1 / len(groups) for key in groups}
    saved = sampler.state_dict()
    weighted = sampler.draw(64, weights)
    restored = ExposureSampler(ids, memberships)
    restored.load_state_dict(saved)
    assert restored.draw(64, weights) == weighted
    assert set(weighted) <= set(ids)
    schedule = stage_lengths(protocol)
    result = dict(passed=True, scope='CPU: split identity, sampler coverage, weighted draws, exact sampler continuation and step budget',
                  stage_lengths=schedule, optimizer_steps=sum(schedule), actual_exposure_budget=sum(schedule)*32,
                  gpu_training_checked=False, train_audio=len(train), feedback_audio=len(feedback), test_audio=len(test))
    save(Path(protocol['root']) / 'cpu_smoke.json', result)
    print(json.dumps(result, ensure_ascii=False))


def smoke_subsets(parts, count):
    """Keep shared texts and distinct patients in the reduced Setting E run."""
    from therapist_harness.text import text_normalize
    assert count % 8 == 0
    common = set.intersection(*[{r['clean_gt'] for r in part} for part in parts])
    eligible = [text for text in common if all(
        len({r['speaker_id'] for r in part if r['clean_gt'] == text}) >= 4 for part in parts)]
    eligible.sort(key=lambda text: (len(text_normalize(text)), hashlib.sha256(text.encode()).hexdigest()))
    half = count // 8
    assert len(eligible) >= 2 * half
    targets = eligible[:half] + eligible[-half:]
    subsets = []
    for part in parts:
        selected = []
        for target in targets:
            candidates = sorted((r for r in part if r['clean_gt'] == target),
                                key=lambda r: hashlib.sha256(r['utt_id'].encode()).hexdigest())
            patients = set()
            for row in candidates:
                if row['speaker_id'] not in patients:
                    selected.append(row)
                    patients.add(row['speaker_id'])
                if len(patients) == 4:
                    break
        subsets.append(selected)
    assert all(len(part) == count and {r['clean_gt'] for r in part} == set(targets) for part in subsets)
    return subsets


def restore_training_state(optimizer, scheduler, sampler, state, rank):
    import numpy as np
    import torch
    from therapist_harness.curriculum_v2_training import state_digest
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    sampler.load_state_dict(state['sampler'])
    assert state_digest(optimizer.state_dict()) == state_digest(state['optimizer'])
    assert state_digest(scheduler.state_dict()) == state_digest(state['scheduler'])
    assert sampler.state_dict() == state['sampler']
    assert scheduler.last_epoch == state['step']
    assert all(int(s['step']) == state['step'] for s in optimizer.state.values())
    rng = state['rng_per_rank'][rank]
    assert rng['rank'] == rank
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.random.set_rng_state(rng['cpu'])
    if 'cuda' in rng:
        torch.cuda.set_rng_state(rng['cuda'])


def compare_replayed_losses(original, replayed, start_step, end_step):
    """Check the restored next update; report later numerical drift without CER selection."""
    first = start_step+1
    for key in ('loss', 'grad_norm', 'learning_rate'):
        assert abs(original[first][key]-replayed[first][key]) < 1e-6, f'restored first update changed {key}'
    steps = range(first, end_step+1)
    assert all(original[s]['learning_rate'] == replayed[s]['learning_rate'] for s in steps)
    differences = {str(s): replayed[s]['loss']-original[s]['loss'] for s in steps}
    return dict(first_update_matches=True, learning_rates_match=True, loss_differences=differences,
                maximum_loss_difference=max(abs(d) for d in differences.values()),
                all_losses_bitwise_equal=all(d == 0 for d in differences.values()))


def worker(args):
    # Import the existing GPU helpers only in the worker. Their module globals
    # are local to this new process; no old running task or source is changed.
    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import get_linear_schedule_with_warmup, set_seed
    import run_fresh_setting_e_audio_lr2em4_4gpu_50epochs_20260911 as gpu
    from therapist_harness.curriculum_v2_training import state_digest, preserve_rng
    from therapist_harness.sampling_curriculum_data import build_sampling_evidence, build_sampling_outcome, make_sampling_groups
    from therapist_harness.real_sampling_mas import choose_sampling, reflect_sampling
    from therapist_harness.schema import AgentConfig

    protocol = read(args.protocol)
    out = Path(protocol['root']) / args.branch
    gpu.ROOT, gpu.OUT, gpu.WORLD = out, out, args.world
    gpu.CFG = dict(gpu.old.CONFIGS['G_audio_lr5em5'], lr=2e-4)
    os.environ['TMPDIR'] = str(out / 'tmp')
    Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{args.port}',
                            rank=args.rank, world_size=args.world,
                            timeout=timedelta(minutes=120 if protocol.get('material_curriculum') else 90))
    receipt = read(out / 'preparation.json')
    assert all(sha(path) == digest for key in ('source_sha256', 'input_sha256')
               for path, digest in receipt[key].items())
    train, feedback, test, groups = load_inputs(protocol)
    material_mode = bool(protocol.get('material_curriculum'))
    sampler_type = ExposureSampler
    material_index = None
    if material_mode:
        from therapist_harness.material_sampling_mas import (
            MaterialExposureSampler, choose_material_sampling, reflect_material_sampling,
            augment_material_outcome,
        )
        sampler_type = MaterialExposureSampler
        material_index = read(protocol['material_curriculum']['index_path'])
        reflect_sampling = reflect_material_sampling
    train_by_id = {row['utt_id']: row for row in train}
    sampler = sampler_type(train_by_id, {key: group['train_ids'] for key, group in groups.items()})
    timeline = {key: 0.0 for key in ('loading_seconds', 'training_seconds', 'evaluation_seconds',
                                    'agent_wait_seconds', 'checkpoint_seconds')}
    wall_started_unix = read(out / 'supervisor.json')['started_unix']
    load_start = time.monotonic()
    wrapper, base_hash = gpu.old.build(gpu.CFG)
    timeline['loading_seconds'] = time.monotonic() - load_start
    initial_hash = gpu.adapter_digest(wrapper.model)
    frozen_hash = gpu.frozen_digest(wrapper.model)
    assert len(set(gpu.gather((base_hash, initial_hash)))) == 1
    if args.rank == 0:
        save(out / 'initialization.json', dict(base_tensor_sha256=base_hash, fresh_adapter_sha256=initial_hash,
             seed=42, world_size=args.world, initial_adapter=None, learning_rate=2e-4,
             trainable_parameters=sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)))
    dist.barrier()
    if args.smoke:
        gpu.smoke(wrapper, args.rank, train, list(range(len(train))), base_hash, initial_hash)
        if not material_mode:
            train, feedback, test = smoke_subsets((train, feedback, test), 8*args.world)
        groups = make_sampling_groups(train)
        train_by_id = {row['utt_id']: row for row in train}
        sampler = sampler_type(train_by_id, {key: group['train_ids'] for key, group in groups.items()})
        if args.rank == 0:
            save(out / 'smoke_subset.json', dict(train_ids=list(train_by_id), dev_ids=[row['utt_id'] for row in feedback], test_ids=[row['utt_id'] for row in test],
                 stages=2, steps_per_stage=2, shared_texts=sorted({r['clean_gt'] for r in feedback}),
                 full_material_training_pool=material_mode,
                 note='Current split identity preserved; material smoke uses full splits with four optimizer updates, never a formal performance result.'))
    schedule = stage_lengths(protocol, smoke=args.smoke)
    total_steps = sum(schedule)
    params = [p for p in wrapper.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=total_steps)
    set_seed(42 + args.rank)
    history, before, step, exposed, best_feedback = [], None, 0, 0, float('inf')
    previous_exposures = None
    start_stage, pending_outcome = 0, None
    inherited_timing = protocol.get('resume_timing', {})
    for key in timeline:
        timeline[key] += inherited_timing.get(key, 0)

    def timing():
        elapsed = time.time()-wall_started_unix
        return dict(**timeline, wall_seconds=inherited_timing.get('wall_seconds', 0)+elapsed,
                    gpu_allocated_seconds=inherited_timing.get('gpu_allocated_seconds', 0)+args.world*elapsed,
                    recovery_session_seconds=elapsed)

    def budget_reached():
        budget = protocol.get('wall_time_budget_seconds')
        value = [bool(not args.smoke and budget and timing()['wall_seconds'] >= budget) if args.rank == 0 else None]
        dist.broadcast_object_list(value, src=0)
        return value[0]

    def evaluation_timing():
        measured = timing()
        budget = protocol.get('wall_time_budget_seconds')
        return dict(timing=measured, within_wall_budget=bool(not budget or measured['wall_seconds'] <= budget))

    def evaluate(data, directory):
        torch.cuda.synchronize()
        tick = time.monotonic()
        metrics = gpu.evaluate(wrapper, data, directory, args.rank)
        metrics.update(world_size=args.world,
                       provenance=f'{args.world} rank decoding receipts; original batch-of-8 membership preserved.')
        if args.rank == 0:
            save(directory / 'summary.json', metrics)
        dist.barrier()
        torch.cuda.synchronize()
        timeline['evaluation_seconds'] += time.monotonic()-tick
        decoded = {row['utt_id']: row for row in read(directory / 'predictions.json')}
        predictions = [dict(row, **decoded[row['utt_id']]) for row in data]
        return metrics, predictions

    def finish_reflection(outcome, directory):
        tick = time.monotonic()
        if args.rank == 0:
            event(out, 'agent_started', phase='reflect_and_experience',
                  round_index=outcome['round_index'], step=outcome['evidence']['checkpoint_step'])
            with preserve_rng():
                history.append(reflect_sampling(AgentConfig.model_validate(protocol['agent']), outcome,
                                                 directory=directory / 'reflection', history=history))
            save(out / 'history.json', history)
        dist.barrier()
        timeline['agent_wait_seconds'] += time.monotonic()-tick

    if protocol.get('resume_from'):
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        checkpoint = Path(protocol['resume_from'])
        source = checkpoint.parent.parent
        proof = read(checkpoint / 'provenance.json')
        assert proof['base_tensor_sha256'] == base_hash and proof['fresh_adapter_sha256'] == initial_hash
        assert read(source / 'initialization.json') == read(out / 'initialization.json')
        weights = load_file(str(checkpoint / 'adapter_model.safetensors'))
        # These state_dict tensors share storage with the existing adapter. This
        # avoids the installed PEFT loader's unrelated tensor-parallel import.
        current_weights = get_peft_model_state_dict(wrapper.model)
        assert current_weights.keys() == weights.keys()
        with torch.no_grad():
            for name, value in weights.items():
                assert current_weights[name].shape == value.shape
                current_weights[name].copy_(value)
        assert state_digest(get_peft_model_state_dict(wrapper.model)) == state_digest(weights)
        state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
        assert len(state['rng_per_rank']) == args.world and not state['pending_stage_ids']
        step, exposed, start_stage = state['step'], state['actual_audio_exposures'], state['completed_stages']
        assert step == sum(schedule[:start_stage]) and exposed == step*32

        def old_predictions(at_step):
            decoded = {r['utt_id']: r for r in read(source / f'evaluations/step-{at_step}/feedback/predictions.json')}
            return [dict(row, **decoded[row['utt_id']]) for row in feedback]

        if (checkpoint / 'context.json').exists():
            context = read(checkpoint / 'context.json')
            assert sha(checkpoint / 'context.json') == proof['context_sha256']
            history, before = context['history'], context['previous_feedback']
            feedback_predictions = context['feedback_predictions']
            previous_exposures = context['previous_exposures']
            best_feedback = context['best_feedback']
            pending_outcome = context['pending_outcome']
            episode = checkpoint.parent / 'reflection/episode.json'
            if pending_outcome and episode.exists():
                history.append(read(episode))
                pending_outcome = None
        else:
            # v4 saved only after reflection, so all earlier episodes are complete.
            history = [read(source / f'stage-{n:03d}/reflection/episode.json') for n in range(1, start_stage+1)]
            before = old_predictions(sum(schedule[:start_stage-1])) if start_stage else None
            feedback_predictions = old_predictions(step)
            previous_exposures = read(checkpoint.parent / 'actual_exposure.json')['per_utterance']
            best_feedback = min(e['feedback']['patient_equal_CER_percent'] for e in rows(out / 'events.jsonl')
                                if e['stage'] == 'evaluation_complete')
        feedback_metrics = read(source / f'evaluations/step-{step}/feedback/summary.json')
        test_metrics = read(source / f'evaluations/step-{step}/test/summary.json')
        restore_training_state(optimizer, scheduler, sampler, state, args.rank)
        assert len(set(gpu.gather(gpu.adapter_digest(wrapper.model)))) == 1
        if args.rank == 0:
            save(out / 'selected_checkpoint.json', read(source / 'selected_checkpoint.json'))
            save(out / 'latest_checkpoint.json', dict(step=step, path=str(checkpoint)))
            save(out / 'history.json', history)
            event(out, 'resume_verified', step=step, completed_stages=start_stage,
                  actual_audio_exposures=exposed, checkpoint=str(checkpoint),
                  adapter_exact=True, optimizer_exact=True, scheduler_exact=True, sampler_exact=True,
                  rank_rng_restored=True, timing=timing())
        if pending_outcome:
            finish_reflection(pending_outcome, out / f'stage-{start_stage:03d}')
    else:
        feedback_metrics, feedback_predictions = evaluate(feedback, out / 'evaluations/step-0/feedback')
        best_feedback = feedback_metrics['patient_equal_CER_percent']
        if args.rank == 0:
            initial_path = out / 'checkpoint-initial'
            wrapper.model.save_pretrained(initial_path)
            save(out / 'selected_checkpoint.json', dict(step=0, path=str(initial_path),
                 selection_metric='dev_patient_equal_CER_percent', value=best_feedback))
        dist.barrier()
        test_metrics, _ = evaluate(test, out / 'evaluations/step-0/test')
        if args.rank == 0:
            event(out, 'evaluation_complete', step=0, feedback=feedback_metrics, test=test_metrics, **evaluation_timing())

    for stage, stage_steps in enumerate(schedule):
        if stage < start_stage:
            continue
        if budget_reached():
            break
        directory = out / f'stage-{stage+1:03d}'
        stage_start_step = step
        state_before = (state_digest(optimizer.state_dict()), state_digest(scheduler.state_dict()))
        evidence = build_sampling_evidence(train, feedback_predictions, groups,
                     feedback_rows=feedback, previous_feedback=before, round_index=stage+1, checkpoint_step=step,
                     exposures=previous_exposures, setting=protocol['setting'])
        tick = time.monotonic()
        decision = None
        if args.rank == 0:
            save(directory / 'evidence.json', evidence)
            if args.branch == 'mas':
                if stage == start_stage and protocol.get('replay_stage_directory'):
                    from therapist_harness.real_sampling_mas import SamplingDecision, validate_decision
                    original = Path(protocol['replay_stage_directory'])
                    assert evidence == read(original / 'evidence.json')
                    recorded = read(original / 'decision/decision.json')
                    weights = validate_decision(SamplingDecision.model_validate(recorded), evidence)
                    assert read(original / 'decision/receipt.json')['validated']
                    decision = dict(group_weights=weights, decision=recorded,
                                    sources={str(p): sha(p) for p in (original / 'decision').rglob('*.json')})
                    save(directory / 'reused_decision.json', decision)
                    event(out, 'decision_reused', step=step, source=str(original / 'decision'))
                else:
                    event(out, 'agent_started', phase='prescribe', round_index=stage+1, step=step)
                    with preserve_rng():
                        if material_mode:
                            selected = choose_material_sampling(AgentConfig.model_validate(protocol['agent']), evidence,
                                directory=directory / 'decision', history=history, index=material_index,
                                exposures=dict(sampler.recording_counts), stage_exposures=stage_steps*32,
                                groups=groups,
                                candidate_limit=protocol['material_curriculum']['candidate_limit'],
                                ucg_feedback=dict(feedback_rows=feedback, current_predictions=feedback_predictions,
                                                  previous_predictions=before, checkpoint_step=step)
                                    if protocol.get('ucg_material_policy') else None)
                        else:
                            selected = choose_sampling(AgentConfig.model_validate(protocol['agent']), evidence,
                                                        directory=directory / 'decision', history=history)
                    decision = selected
            else:
                decision = dict(group_weights=None, decision={'reason': '固定普通随机遍历，每条真实录音每次遍历出现一次。'})
        shared = [decision]
        dist.broadcast_object_list(shared, src=0)
        decision = shared[0]
        timeline['agent_wait_seconds'] += time.monotonic()-tick
        assert state_before == (state_digest(optimizer.state_dict()), state_digest(scheduler.state_dict()))
        if budget_reached():
            break
        channels = None
        if material_mode and args.branch == 'mas':
            ids, channels = sampler.draw_group_materials(stage_steps*32, decision['selected_ids'], decision['selected_probability'], decision['group_weights'])
        else:
            ids = sampler.draw(stage_steps * 32, decision['group_weights'])
        if stage == start_stage and protocol.get('replay_stage_directory'):
            assert ids == read(Path(protocol['replay_stage_directory']) / 'sampling_plan.json')['utterance_ids']
        assert len(set(gpu.gather(hashlib.sha256(json.dumps(ids).encode()).hexdigest()))) == 1
        if args.rank == 0:
            save(directory / 'sampling_plan.json', dict(stage=stage+1, checkpoint_step=step,
                 group_weights=decision['group_weights'], utterance_ids=ids,
                 **(dict(selected_probability=decision['selected_probability'], selected_ids=decision['selected_ids'],
                         selected_probability_scope='within_group', channels=channels) if channels is not None else {}),
                 sampler='group_then_material_with_replacement' if channels is not None else
                         'without_replacement' if args.branch=='uniform' else 'with_replacement'))
            event(out, 'stage_started', round_index=stage+1, step=step, group_weights=decision['group_weights'],
                  **(dict(selected_probability=decision['selected_probability'], selected_probability_scope='within_group', selected_recordings=len(decision['selected_ids']))
                     if channels is not None else {}))
        observed = []
        wrapper.model.train()
        wrapper.model.config.use_cache = False
        for offset in range(0, len(ids), 32):
            if budget_reached():
                break
            torch.cuda.synchronize()
            tick = time.monotonic()
            batch = [train_by_id[uid] for uid in ids[offset:offset+32]]
            loss, gradient, actual = gpu.backward(wrapper, batch, args.rank)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            lr = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            seconds = time.monotonic()-tick
            timeline['training_seconds'] += seconds
            observed.extend(actual)
            exposed += len(actual)
            step += 1
            if args.rank == 0:
                event(out, 'loss', step=step, round_index=stage+1, loss=loss, grad_norm=float(norm),
                      learning_rate=lr, actual_audio_exposures=exposed, seconds=seconds, timing=timing())
        if not observed:
            break
        assert observed == ids[:len(observed)] and scheduler.last_epoch == step
        assert all(int(state['step']) == step for state in optimizer.state.values())
        feedback_metrics, after = evaluate(feedback, out / f'evaluations/step-{step}/feedback')
        test_metrics, _ = evaluate(test, out / f'evaluations/step-{step}/test')
        delivery = exposure_summary(observed, train_by_id)
        if args.rank == 0:
            save(directory / 'actual_exposure.json', dict(**delivery, utterance_ids=observed,
                 **(dict(channels=channels[:len(observed)]) if channels is not None else {}),
                 sampling_plan_sha256=sha(directory / 'sampling_plan.json')))
            event(out, 'evaluation_complete', step=step, round_index=stage+1, actual_audio_exposures=exposed,
                  feedback=feedback_metrics, test=test_metrics, **evaluation_timing())
        should_reflect = not budget_reached()
        outcome = None
        if args.branch == 'mas' and should_reflect:
            outcome = build_sampling_outcome(train, feedback_predictions, after, groups,
                          feedback_rows=feedback, decision=decision['decision'], round_index=stage+1,
                          before_checkpoint_step=stage_start_step,
                          checkpoint_step=step, exposures=delivery['per_utterance'], setting=protocol['setting'])
            if material_mode:
                outcome = augment_material_outcome(outcome, feedback_predictions, after,
                    feedback_rows=feedback, index=material_index, channels=channels[:len(observed)], observed=observed, groups=groups)
        if args.rank == 0 and stage == start_stage and protocol.get('replay_stage_directory'):
            original = Path(protocol['replay_stage_directory']).parent
            old_losses = {r['step']: r for r in rows(original / 'events.jsonl') if r['stage'] == 'loss'}
            new_losses = {r['step']: r for r in rows(out / 'events.jsonl') if r['stage'] == 'loss'}
            replay_check = compare_replayed_losses(old_losses, new_losses, stage_start_step, step)
            for role in ('feedback', 'test'):
                def transcripts(root):
                    return {r['utt_id']: r['prediction'] for r in read(root / f'evaluations/step-{step}/{role}/predictions.json')}
                old_text, new_text = transcripts(original), transcripts(out)
                assert old_text.keys() == new_text.keys()
                replay_check[f'{role}_changed_transcripts'] = sum(old_text[k] != new_text[k] for k in old_text)
            save(out / 'replay_verified.json', dict(start_step=stage_start_step, end_step=step,
                 identical_sample_order=True, **replay_check,
                 interpretation='Verified exact restored state and first update; later numerical drift is reported. New dev/test results are used regardless of improvement or degradation.'))
            event(out, 'replay_verified', step=step, **replay_check)
        before, feedback_predictions = feedback_predictions, after
        previous_exposures = delivery['per_utterance']
        improved = feedback_metrics['patient_equal_CER_percent'] < best_feedback
        best_feedback = min(best_feedback, feedback_metrics['patient_equal_CER_percent'])
        tick = time.monotonic()
        rng = gpu.gather(dict(rank=args.rank, python=random.getstate(), numpy=np.random.get_state(),
                              cpu=torch.random.get_rng_state(), cuda=torch.cuda.get_rng_state().cpu()))
        if args.rank == 0:
            checkpoint = directory / 'checkpoint'
            wrapper.model.save_pretrained(checkpoint)
            save(checkpoint / 'context.json', dict(history=history, previous_feedback=before,
                 feedback_predictions=feedback_predictions, previous_exposures=previous_exposures,
                 best_feedback=best_feedback, pending_outcome=outcome))
            torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       rng_per_rank=rng, sampler=sampler.state_dict(), step=step,
                       completed_stages=stage+int(len(observed)==len(ids)),
                       pending_stage_ids=ids[len(observed):], stage_index=stage+1,
                       actual_audio_exposures=exposed), checkpoint / 'training_state.pt')
            save(checkpoint / 'provenance.json', dict(step=step, branch=args.branch, fresh_adapter_sha256=initial_hash,
                 base_tensor_sha256=base_hash, preparation_sha256=sha(out / 'preparation.json'),
                 optimizer_steps=step, scheduler_steps=scheduler.last_epoch,
                 adapter_sha256=sha(checkpoint / 'adapter_model.safetensors'),
                 training_state_sha256=sha(checkpoint / 'training_state.pt'),
                 context_sha256=sha(checkpoint / 'context.json')))
            save(out / 'latest_checkpoint.json', dict(step=step, path=str(checkpoint)))
            if improved:
                save(out / 'selected_checkpoint.json', dict(step=step, path=str(checkpoint),
                     selection_metric='dev_patient_equal_CER_percent', value=best_feedback))
        dist.barrier()
        timeline['checkpoint_seconds'] += time.monotonic()-tick
        if args.rank == 0:
            event(out, 'checkpoint_saved', step=step, path=str(directory / 'checkpoint'),
                  reflection_pending=outcome is not None, timing=timing())
        if outcome is not None:
            finish_reflection(outcome, directory)
        if args.rank == 0:
            event(out, 'stage_complete' if len(observed)==len(ids) else 'stage_stopped',
                  round_index=stage+1, step=step, timing=timing(),
                  planned_stage_steps=stage_steps, actual_stage_steps=len(observed)//32,
                  continuous_optimizer_scheduler=True)

    assert exposed == step * 32
    if not protocol.get('wall_time_budget_seconds'):
        assert step == total_steps
    assert gpu.old.base_digest(wrapper.model) == base_hash and gpu.frozen_digest(wrapper.model) == frozen_hash
    assert len(set(gpu.gather(gpu.adapter_digest(wrapper.model)))) == 1
    assert all(sha(path) == digest for key in ('source_sha256', 'input_sha256')
               for path, digest in receipt[key].items())
    if args.rank == 0:
        save(out / 'complete.json', dict(passed=True, smoke=args.smoke, branch=args.branch, steps=step,
             actual_audio_exposures=exposed, world_size=args.world, timing=timing(),
             planned_steps=total_steps, stopped_at_wall_budget=step < total_steps,
             base_frozen=True, optimizer_scheduler_continuous=True, feedback=feedback_metrics, test=test_metrics))
        event(out, 'complete', smoke=args.smoke, steps=step, timing=timing())
    dist.barrier()
    dist.destroy_process_group()


def supervise(args):
    protocol = read(args.protocol)
    out = Path(protocol['root']) / args.branch
    assert (out / 'preparation.json').exists()
    assert not (out / 'supervisor.json').exists(), 'Use a new directory; never overwrite an earlier execution.'
    if args.branch == 'mas':
        from therapist_harness.laicai_credentials import load_laicai_key
        load_laicai_key()
    gpus = args.gpus.split(',')
    assert len(gpus) in (1, 2, 4) and len(gpus) == len(set(gpus))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    processes, handles = [], []
    previous_sigterm = signal.signal(signal.SIGTERM, _terminate_supervisor)
    save(out / 'supervisor.json', dict(pid=os.getpid(), gpus=gpus, port=port, smoke=args.mode=='smoke', started_unix=time.time()))
    try:
        for rank, gpu in enumerate(gpus):
            handle = (out / f'rank-{rank}.log').open('w')
            handles.append(handle)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--protocol', args.protocol,
                       '--branch', args.branch, '--rank', str(rank), '--world', str(len(gpus)), '--port', str(port)]
            if args.mode == 'smoke':
                command.append('--smoke')
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
            processes.append(process)
            event(out, 'process_started', rank=rank, gpu=gpu, pid=process.pid)
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError('A training worker failed; stopping this execution.')
            time.sleep(2)
        assert all(process.returncode == 0 for process in processes)
        assert read(out / 'complete.json')['passed']
    except BaseException as error:
        save(out / 'failure.json', dict(error_type=type(error).__name__,
             worker_returncodes=[p.poll() for p in processes], detail='See rank logs; no completion was recorded.'))
        event(out, 'execution_failed', error_type=type(error).__name__)
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


def _terminate_supervisor(signum, frame):
    # Let the supervisor's finally block close its own GPU workers.
    raise SystemExit(128 + signum)


def report(args):
    """Use latest completed evaluations at a common time, never best test CER."""
    protocol = read(args.protocol)
    root = Path(protocol['root'])
    runs = {}
    for branch in ('uniform', 'mas'):
        directory = root / branch
        evaluations = [row for row in rows(directory / 'events.jsonl') if row['stage'] == 'evaluation_complete']
        runs[branch] = dict(evaluations=evaluations, initialization=read(directory / 'initialization.json'),
                            supervisor=read(directory / 'supervisor.json'))
    assert runs['uniform']['initialization'] == runs['mas']['initialization']
    assert len(runs['uniform']['supervisor']['gpus']) == len(runs['mas']['supervisor']['gpus'])
    common_seconds = min(run['evaluations'][-1]['timing']['wall_seconds'] for run in runs.values())
    if protocol.get('wall_time_budget_seconds'):
        common_seconds = min(common_seconds, protocol['wall_time_budget_seconds'])
    timed = {}
    for branch, run in runs.items():
        available = [row for row in run['evaluations'] if row['timing']['wall_seconds'] <= common_seconds]
        timed[branch] = available[-1] if available else None
    by_step = {branch: {row['step']: row for row in run['evaluations']} for branch, run in runs.items()}
    shared_steps = sorted(set(by_step['uniform']) & set(by_step['mas']))
    result = dict(comparison='fixed-step and fixed-wall-clock are reported separately',
         wall_time_rule='Latest completed evaluation at or before the common elapsed time; no CER interpolation, no best-test selection. Loading, evaluation and Agent waiting count.',
         common_elapsed_seconds=common_seconds, same_walltime=timed,
         same_steps=[dict(step=step, uniform=by_step['uniform'][step], mas=by_step['mas'][step]) for step in shared_steps],
         test_role=f"{protocol['test_audio']} real held-out recordings under Setting {protocol['setting']}; reporting only, never delivered to MAS",
         feedback_role='dev')
    save(root / 'comparison.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'cpu-smoke', 'smoke', 'supervise', 'worker', 'report'])
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--branch', choices=['uniform', 'mas'], required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=4)
    parser.add_argument('--port', type=int)
    parser.add_argument('--smoke', action='store_true')
    arguments = parser.parse_args()
    if arguments.mode == 'prepare':
        prepare(arguments)
    elif arguments.mode == 'cpu-smoke':
        cpu_smoke(arguments)
    elif arguments.mode == 'worker':
        worker(arguments)
    elif arguments.mode == 'report':
        report(arguments)
    else:
        supervise(arguments)
