import math
from types import SimpleNamespace

import pytest
import torch

from therapist_harness.dpo_training import AudioPreferenceCollator, dpo_losses, transcript_log_probs
from therapist_harness.preference_token_mask import collator_type, edit_eos_labels, edit_positions
from test_audio_dpo_training import FakeProcessor


@pytest.mark.parametrize('chosen,rejected,expected',[
    ([1,2,3],[1,4,3],([1],[1],1)),
    ([1,3],[1,2,3],([],[1],1)),
    ([1,2,3],[1,3],([1],[],1)),
    ([1,1],[1],([0],[],1)),
    ([1,2],[1,2],([],[],0)),
    ([],[1],([],[0],1)),
])
def test_edit_alignment_handles_substitution_insertion_deletion_and_repetition(chosen,rejected,expected):
    assert edit_positions(chosen,rejected)==expected


def test_batched_collator_preserves_inputs_full_mode_and_real_teacher_labels():
    processor=FakeProcessor()
    processor.tokenizer=SimpleNamespace(eos_token='§',eos_token_id=0,pad_token_id=0)
    data=[dict(source='synthetic',audio_id='a',audio='/a.wav',chosen='甲乙丙',rejected='甲丁丙'),
          dict(source='synthetic',audio_id='b',audio='/b.wav',chosen='甲丙',rejected='甲乙丙')]
    loader=lambda path:[1.,2.] if path=='/a.wav' else [3.,4.]
    original=AudioPreferenceCollator(processor,audio_loader=loader)
    full_audit={};edit_audit={}
    full=collator_type('full',full_audit)(processor,audio_loader=loader)
    edit=collator_type('edit_eos',edit_audit)(processor,audio_loader=loader)
    expected=original(data);unchanged=full(data);actual=edit(data)
    assert all(torch.equal(expected[k],unchanged[k]) for k in expected)
    assert all(torch.equal(expected[k],actual[k]) for k in expected if k!='labels')
    supervised=[x[x.ne(-100)].tolist() for x in actual['labels']]
    assert supervised==[[ord('乙'),0],[0],[ord('丁'),0],[ord('乙'),0]]
    assert actual['labels'][1,-1]==-100  # EOS == PAD does not unmask padding.
    real=[dict(source='real',audio_id='r',audio='/a.wav',chosen='甲乙丙')]
    a,b=original(real,include_rejected=False),edit(real,include_rejected=False)
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert len(edit_audit)==len(full_audit)==2
    edit(data)  # Repeated reference/policy scoring must retain exactly the same mask.


def test_masked_loss_has_no_score_gradient_on_shared_tokens_and_keeps_EOS():
    labels=torch.tensor([[-100,1,2,0],[-100,1,3,0]])
    before=labels.clone();masked,_=edit_eos_labels(labels,1,0)
    assert torch.equal(labels,before)
    logits=torch.zeros(2,4,5,requires_grad=True)
    logs,counts=transcript_log_probs(logits,masked)
    assert counts.tolist()==[2,2]
    reference=logs.detach().clone().requires_grad_()
    loss=dpo_losses(logs[:1],logs[1:],reference[:1],reference[1:],.1).mean()
    assert loss.item()==pytest.approx(math.log(2))
    loss.backward()
    assert logits.grad[:,0].count_nonzero()==0
    assert logits.grad[:,3].count_nonzero()==0
    assert logits.grad[:,1:3].count_nonzero()>0
    assert reference.grad is None
