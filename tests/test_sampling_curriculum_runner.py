import importlib.util
import json
import pytest
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


def runner():
    path = Path(__file__).resolve().parents[1] / 'examples/run_setting_e_sampling_curriculum_20260911.py'
    spec = importlib.util.spec_from_file_location('sampling_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_time_comparison_uses_latest_completed_result_not_best_test(tmp_path):
    module = runner()
    protocol = tmp_path / 'protocol.json'
    module.save(protocol, {'root': str(tmp_path), 'wall_time_budget_seconds': 100, 'test_audio': 372})
    # The better step-1 score must not replace the later step-2 model, and the
    # even better step-3 result completed outside the agreed time budget.
    for branch, delays in [('uniform', [10, 30, 70, 110]), ('mas', [15, 45, 90, 130])]:
        root = tmp_path / branch
        module.save(root / 'initialization.json', {'seed': 42, 'fresh_adapter_sha256': 'same', 'world_size': 2})
        module.save(root / 'supervisor.json', {'gpus': ['0', '1'] if branch == 'uniform' else ['2', '3']})
        records = [dict(stage='evaluation_complete', step=step, timing={'wall_seconds': elapsed},
                        test={'CER_percent': cer}) for step, elapsed, cer in zip(range(4), delays, [20, 8, 12, 3])]
        (root / 'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    module.report(SimpleNamespace(protocol=str(protocol)))
    result = module.read(tmp_path / 'comparison.json')
    assert result['common_elapsed_seconds'] == 100
    assert result['same_walltime']['uniform']['step'] == 2
    assert result['same_walltime']['mas']['step'] == 2
    assert [row['step'] for row in result['same_steps']] == [0, 1, 2, 3]


def test_dev_is_disjoint_from_train_and_test(tmp_path):
    module = runner()
    train = [dict(utt_id='a', speaker_id='p1', prompt='', clean_gt='南')]
    dev = [dict(utt_id='b', speaker_id='p2', prompt='', clean_gt='南')]
    test = [dict(utt_id='c', speaker_id='p3', prompt='', clean_gt='南')]
    for name, data in [('train', train), ('dev', dev), ('test', test)]:
        (tmp_path / f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in data))
    module.save(tmp_path / 'groups.json', {'group': {'train_ids': ['a']}})
    protocol = dict(train_manifest=str(tmp_path / 'train.jsonl'), feedback_manifest=str(tmp_path / 'dev.jsonl'),
                    test_manifest=str(tmp_path / 'test.jsonl'), groups_path=str(tmp_path / 'groups.json'),
                    setting='E', feedback_role='dev', seed=42, global_batch=32, stages=2, steps_per_stage=1)
    actual_train, feedback, actual_test, groups = module.load_inputs(protocol)
    assert actual_train == train and feedback == dev and actual_test == test
    for invalid_feedback in ('train', 'test'):
        with pytest.raises(AssertionError):
            module.load_inputs(dict(protocol, feedback_manifest=str(tmp_path / f'{invalid_feedback}.jsonl')))


def test_terminating_supervisor_reaps_its_worker(tmp_path):
    module = runner()
    module.save(tmp_path / 'protocol.json', {'root': str(tmp_path)})
    module.save(tmp_path / 'uniform/preparation.json', {})
    script = r'''
import importlib.util, json, pathlib, subprocess, sys
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('supervisor_under_test', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
root = pathlib.Path(sys.argv[2])
real_popen = subprocess.Popen
def dummy_worker(command, **kwargs):
    child = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
    (root/'worker_pid').write_text(str(child.pid))
    return child
m.subprocess.Popen = dummy_worker
m.supervise(SimpleNamespace(protocol=str(root/'protocol.json'), branch='uniform', gpus='0', mode='supervise'))
'''
    process = subprocess.Popen([sys.executable, '-c', script, module.__file__, str(tmp_path)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / 'worker_pid').exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / 'worker_pid').exists()
        child_pid = int((tmp_path / 'worker_pid').read_text())
        process.terminate()
        assert process.wait(timeout=20) == 143
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            child_pid = None
        assert child_pid is None, 'terminating the supervisor left its worker alive'
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_smoke_preserves_shared_texts_and_patient_disjointness():
    module = runner()
    targets = ["南", "開門", "幫我開門", "麻煩你幫我打開門"]
    parts = [[dict(utt_id=f"{split}-{t}-{patient}", speaker_id=f"{split}-p{patient}", clean_gt=target)
              for t, target in enumerate(targets) for patient in range(5)] for split in ('train', 'dev', 'test')]
    small = module.smoke_subsets(parts, 16)
    assert small == module.smoke_subsets([list(reversed(p)) for p in parts], 16)
    assert all(len(p) == len({r['utt_id'] for r in p}) == 16 for p in small)
    assert all({r['clean_gt'] for r in p} == set(targets) for p in small)
    assert not {r['speaker_id'] for r in small[0]} & {r['speaker_id'] for r in small[1]}
    assert all(len({r['speaker_id'] for r in p if r['clean_gt'] == t}) == 4 for p in small for t in targets)


def test_29_step_schedule_stops_at_590_with_a_ten_step_tail():
    from itertools import accumulate
    from therapist_harness.sampling_curriculum_runtime import ExposureSampler
    module = runner()
    protocol = dict(stages=21, steps_per_stage=29, total_optimizer_steps_limit=590)
    sizes = module.stage_lengths(protocol)
    assert sizes == [29] * 20 + [10]
    endpoints = list(accumulate(sizes))
    assert endpoints == list(range(29, 581, 29)) + [590]
    sampler = ExposureSampler([str(n) for n in range(1862)], {'all': [str(n) for n in range(1862)]})
    delivered = [uid for size in sizes for uid in sampler.draw(size * 32)]
    assert len(delivered) == 18880
    for start in range(0, 18620, 1862):
        assert len(set(delivered[start:start+1862])) == 1862
    assert module.stage_lengths(protocol, smoke=True) == [2, 2]
    with pytest.raises(AssertionError, match='stage count'):
        module.stage_lengths(dict(protocol, stages=20))


def test_scheduler_uses_590_steps_across_29_step_boundaries():
    import torch
    from transformers import get_linear_schedule_with_warmup
    module = runner()
    sizes = module.stage_lengths(dict(stages=21, steps_per_stage=29, total_optimizer_steps_limit=590))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=sum(sizes))
    step = 0
    for size in sizes:
        for _ in range(size):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
        assert scheduler.last_epoch == step
        assert int(optimizer.state[parameter]['step']) == step
    assert step == 590 and optimizer.param_groups[0]['lr'] == 0


def test_resume_preserves_optimizer_scheduler_sampling_and_dropout_rng():
    import copy
    import random
    import numpy as np
    import torch
    from transformers import get_linear_schedule_with_warmup
    from therapist_harness.sampling_curriculum_runtime import ExposureSampler
    module = runner()
    def build():
        parameter = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
        optimizer = torch.optim.AdamW([parameter], lr=2e-4)
        scheduler = get_linear_schedule_with_warmup(optimizer, 2, 10)
        sampler = ExposureSampler(['a','b','c'], {'one':['a','b'], 'two':['c']})
        return parameter, optimizer, scheduler, sampler
    def update(p, optimizer, scheduler, sampler):
        drawn = sampler.draw(5, {'one':.6, 'two':.4})
        loss = (p*torch.rand(3)).sum() * (random.random()+np.random.rand())
        loss.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
        return drawn, float(loss), p.detach().clone()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    p, opt, sched, sampler = build()
    for _ in range(3):
        update(p,opt,sched,sampler)
    saved_parameter = p.detach().clone()
    state = copy.deepcopy(dict(optimizer=opt.state_dict(), scheduler=sched.state_dict(),
        sampler=sampler.state_dict(), step=3, rng_per_rank=[dict(rank=0, python=random.getstate(),
        numpy=np.random.get_state(), cpu=torch.random.get_rng_state())]))
    expected = [update(p,opt,sched,sampler) for _ in range(3)]
    p,opt,sched,sampler = build()
    with torch.no_grad(): p.copy_(saved_parameter)
    module.restore_training_state(opt,sched,sampler,state,0)
    actual = [update(p,opt,sched,sampler) for _ in range(3)]
    for a,b in zip(actual,expected):
        assert a[:2] == b[:2] and torch.equal(a[2],b[2])


def test_replay_checks_restored_update_and_reports_later_numerical_drift():
    module = runner()
    original = {n: dict(loss=1., grad_norm=.8, learning_rate=.0001) for n in (204,205)}
    replayed = {n: dict(row) for n,row in original.items()}
    replayed[205]['loss'] += .006
    result = module.compare_replayed_losses(original,replayed,203,205)
    assert result['first_update_matches'] and not result['all_losses_bitwise_equal']
    assert result['maximum_loss_difference'] == pytest.approx(.006)
    replayed[204]['grad_norm'] += .01
    with pytest.raises(AssertionError,match='restored first update'):
        module.compare_replayed_losses(original,replayed,203,205)
