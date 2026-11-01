import hashlib
import json

import pytest

from examples.generate_reviewed_sg import run_attempts
from therapist_harness.generation_design import ConfiguredGenerator, load_generation_choices, role_delivery
from therapist_harness.transcript_review import ThreeWayReviewJudgment


def test_pending_quarantines_without_retry_or_invented_label():
    calls = []
    attempts, action = run_attempts(42, lambda *x: calls.append(x),
        lambda *_: ThreeWayReviewJudgment(verdict='pending', reason='疑似參考內容混入，需聽辨'))
    assert action == 'quarantine' and len(calls) == len(attempts) == 1


def test_instruction_control_preserves_actual_target_and_reference():
    calls = []
    class Model:
        sample_rate = 24000
        def inference_zero_shot(self, *args, **kwargs): calls.append(args)
        def inference_instruct2(self, *args, **kwargs): calls.append(args)
    ConfiguredGenerator(Model(), 'zero_shot', 'cantonese_en', '參考原文').inference_zero_shot('目標', 'ignored', '/reference.wav')
    assert calls[-1] == ('目標', 'You are a helpful assistant. Please express in Cantonese.<|endofprompt|>參考原文', '/reference.wav')
    ConfiguredGenerator(Model(), 'instruct2', 'default', '參考原文').inference_zero_shot('目標', 'ignored', '/reference.wav')
    assert '參考原文' not in calls[-1][1] and '广东话' in calls[-1][1]


def test_choices_bind_target_reference_and_identity(tmp_path):
    row = dict(target_text='目標', reference_wav_sha256='a'*64)
    choice = dict(sample_id='s1', mode='zero_shot', instruction_id='default',
                  reference_wav_sha256='a'*64, target_text_sha256=hashlib.sha256('目標'.encode()).hexdigest())
    p=tmp_path/'plan.json';p.write_text(json.dumps({'choices':[choice]}))
    assert load_generation_choices(p, {'s1':row})['s1'] == choice
    with pytest.raises(ValueError):load_generation_choices(p, {'s1':dict(row,target_text='另一目標')})


def test_delivery_counts_audio_not_pair_log_lines():
    assignments=[dict(sample_id='s1',role='target'),dict(sample_id='s2',role='maintenance')]
    reviews=[dict(sample_id='s1',verdict='pass'),dict(sample_id='s2',verdict='pending')]
    pairs=[dict(audio_id='s1'),dict(audio_id='s1')]
    event=dict(step=1,source='synthetic',audio_id='s1',occurrence=0)
    result=role_delivery(assignments,reviews,pairs,[event,event])
    assert result['target']['pairs']==2 and result['target']['exposures']==1
    assert result['maintenance']['exposures']==0 and result['maintenance']['pending']==1
    with pytest.raises(ValueError):role_delivery(assignments,reviews,pairs+[dict(audio_id='s2')],[event])
