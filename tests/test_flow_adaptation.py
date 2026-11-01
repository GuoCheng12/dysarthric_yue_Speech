import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

PATH = Path(__file__).resolve().parents[1] / 'synthesis/cosyvoice3_patient_sft/flow_adaptation.py'
spec = importlib.util.spec_from_file_location('flow_adaptation_test', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def features():
    out = {}
    for i, (name, length) in enumerate([('a', 3), ('b', 5), ('c', 2)]):
        out[name+'.codes'] = torch.arange(length, dtype=torch.int16)
        out[name+'.mel'] = torch.full((length*2, 80), float(i+1))
        out[name+'.speaker'] = torch.ones(192)
    return out


def test_reference_is_condition_not_supervised_and_padding_is_excluded():
    packed = m.pack([{'uid':'a','ref_uid':'b'}, {'uid':'b','ref_uid':'c'}], features(), 'cpu')
    token, x, valid, target, cond, _ = packed
    assert token.shape == (2, 8)
    assert valid.sum().item() == 30
    assert target.sum().item() == 16
    assert not torch.any(cond.masked_select(target.expand_as(cond)))
    assert torch.equal(cond[0,:,:10], x[0,:,:10])
    assert not target[1,:,14:].any()


def test_pairing_is_distinct_same_patient_and_keeps_every_target():
    rows=[{'uid':f'{s}{i}', 'speaker_id':s} for s in ('p','q') for i in range(3)]
    one=m.pair_epoch(rows,1)
    assert one == m.pair_epoch(rows,1)
    assert {r['uid'] for r in one} == {r['uid'] for r in rows}
    assert all(r['uid'] != r['ref_uid'] and r['uid'][0] == r['ref_uid'][0] for r in one)


def test_frame_normalized_accumulation_matches_effective_batch():
    chunks=[torch.arange(12.).reshape(3,4),torch.arange(20.).reshape(5,4)]
    def gradient(accumulated):
        p=torch.tensor(.3,requires_grad=True)
        denominator=sum(x.numel() for x in chunks)
        if accumulated:
            for x in chunks:
                ((p*x-1).square().sum()/denominator).backward()
        else:
            ((p*torch.cat(chunks)-1).square().sum()/denominator).backward()
        return p.grad
    torch.testing.assert_close(gradient(True),gradient(False))


def test_velocity_loss_ignores_reference_and_padded_outputs():
    class Estimator(torch.nn.Module):
        def forward(self, y, valid, mu, t, emb, cond, streaming=False):
            return torch.zeros_like(y)
    class IdentityLookahead(torch.nn.Module):
        def forward(self,x):
            return x
    flow=SimpleNamespace(input_embedding=torch.nn.Embedding(10,80),
        pre_lookahead_layer=IdentityLookahead(), spk_embed_affine_layer=torch.nn.Linear(192,80),
        decoder=SimpleNamespace(sigma_min=1e-6, estimator=Estimator()))
    packed=m.pack([{'uid':'a','ref_uid':'b'}],features(),'cpu')
    torch.manual_seed(10)
    n,d=m.velocity_error(flow,packed,False)
    torch.manual_seed(10)
    torch.rand(1,1,1)
    noise=torch.randn_like(packed[1])
    expected=((packed[1]-(1-1e-6)*noise).square()*packed[3]).sum()
    torch.testing.assert_close(n,expected)
    assert d==3*2*80


def test_epoch_selection_uses_reconstruction_not_text_score():
    import sys
    sys.path.insert(0,str(PATH.parents[2]/'examples'))
    from flow_adaptation_evaluate import select_epoch
    def metric(v):
        return {'reconstruction':{'overall':{'mcd_dtw_db':v}}}
    assert select_epoch({0:metric(5),1:metric(8),2:metric(7.995),3:metric(8.1)})==1
    assert select_epoch({0:metric(5),1:metric(8),2:metric(7.8),3:metric(7.7)})==3
