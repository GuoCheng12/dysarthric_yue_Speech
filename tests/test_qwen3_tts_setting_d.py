import importlib.util
from pathlib import Path

import pytest
import torch

TRAINER = Path(__file__).parents[1] / 'synthesis/qwen3_tts_patient_sft/train_setting_d.py'
spec = importlib.util.spec_from_file_location('qwen_train', TRAINER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_target_prediction_positions_include_first_frame_and_eos():
    assert m.supervision_positions(10, 3).tolist() == [9, 10, 11, 12]
    # Reference positions 0..8 are not supervised. No second label shift.
    assert len(m.supervision_positions(10, 3)) == 4


def test_reference_pairing_and_complete_coverage():
    rows = [{'uid': f'{s}{i}', 'speaker_id': s} for s in ('a', 'b') for i in range(5)]
    p = m.train_pairs(rows, 1)
    assert {r['uid'] for r in p} == {r['uid'] for r in rows}
    assert all(r['ref_uid'] != r['uid'] and r['ref_uid'].startswith(r['speaker_id']) for r in p)
    assert p == m.train_pairs(rows, 1)
    assert p != m.train_pairs(rows, 2)


def test_effective_batch_normalizes_heads_separately():
    x = torch.tensor(2., requires_grad=True)
    a = {'first_sum': x * 4, 'residual_sum': x * 30}
    b = {'first_sum': x * 10, 'residual_sum': x * 90}
    value = m.normalized_loss(a, 7, 60) + m.normalized_loss(b, 7, 60)
    value.backward()
    assert x.grad.item() == pytest.approx(14 / 7 + .3 * 120 / 60)


def test_residual_labels_flatten_without_shifting_codebooks():
    labels = torch.arange(15)[None].repeat(2, 1)
    logits = torch.full((2, 15, 20), -20.)
    logits.scatter_(2, labels[..., None], 20.)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='sum')
    assert loss.item() == 0


def test_last_partial_batch_is_kept():
    sizes = [len(list(range(2108))[i:i+16]) for i in range(0, 2108, 16)]
    assert len(sizes) == 132 and sizes[-1] == 12 and sum(sizes) == 2108


def test_json_io_is_explicit_utf8(tmp_path):
    path = tmp_path / 'record.json'
    m.write(path, {'text': '我隻腳冇力'})
    assert m.read(path)['text'] == '我隻腳冇力'


def test_lora_scope_is_talker_and_residual_predictor_attention():
    import re
    assert re.fullmatch(m.TARGET_MODULES, 'talker.model.layers.0.self_attn.q_proj')
    assert re.fullmatch(m.TARGET_MODULES, 'talker.code_predictor.model.layers.4.self_attn.o_proj')
    assert not re.fullmatch(m.TARGET_MODULES, 'speaker_encoder.q_proj')
