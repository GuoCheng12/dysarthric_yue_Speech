import sys
from pathlib import Path

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from tts_gradient_control import protect_real_gradient


def test_conflict_removed_real_preserved_and_synthetic_capped():
    r=[torch.tensor([2.,0.]),torch.tensor([0.])]
    s=[torch.tensor([-3.,4.]),torch.tensor([0.])]
    g,record=protect_real_gradient(r,s)
    torch.testing.assert_close(g[0],torch.tensor([2.,.5]))
    assert record['projection_coefficient']<0
    assert record['synthetic_to_real_norm_after']<=.25+1e-8


def test_small_aligned_gradient_is_unchanged():
    r=[torch.tensor([2.,0.])];s=[torch.tensor([.1,0.])]
    g,record=protect_real_gradient(r,s)
    torch.testing.assert_close(g[0],r[0]+s[0])
    assert record['synthetic_scale']==1


def test_exact_opposing_gradient_leaves_real_update():
    r=[torch.tensor([2.,0.])];s=[torch.tensor([-3.,0.])]
    g,_=protect_real_gradient(r,s)
    torch.testing.assert_close(g[0],r[0])
