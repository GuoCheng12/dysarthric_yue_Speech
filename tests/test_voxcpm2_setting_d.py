import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'synthesis/voxcpm2_patient_sft'))
from train_setting_d import CONFIG, ROOT, CachedPacker, read, write, train_pairs, weighted_loss


def test_cantonese_json_with_ascii_process_locale(tmp_path):
    import locale
    previous = locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE, 'C')
        path = tmp_path / 'receipt.json'
        write(path, {'text': '唔該晒你呀'})
        assert read(path) == {'text': '唔該晒你呀'}
        assert '唔該晒你呀'.encode('utf-8') in path.read_bytes()
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)


def test_distinct_same_patient_epoch_pairing_and_full_coverage():
    rows = [{'uid': f'{spk}_{i}', 'speaker_id': spk} for spk in ('a', 'b') for i in range(7)]
    p1 = train_pairs(rows, 1)
    assert p1 == train_pairs(rows, 1)
    assert p1 != train_pairs(rows, 2)
    assert {r['uid'] for r in p1} == {r['uid'] for r in rows}
    assert all(r['ref_uid'] != r['uid'] and r['ref_uid'].split('_')[0] == r['speaker_id'] for r in p1)


def test_official_reference_mask_and_stop_position():
    packer = CachedPacker(1, 100, 4, 64, SimpleNamespace(hop_length=640))
    ref, target = torch.ones(3, 4, 64), torch.full((5, 4, 64), 2.)
    tokens, features, text_mask, audio_mask, loss_mask, labels, _, _ = packer.process_tts_data_with_ref(
        ref, target, torch.tensor([700, 701], dtype=torch.int32))
    assert tokens.tolist() == [103, 0, 0, 0, 104, 700, 701, 101, 0, 0, 0, 0, 0, 102]
    assert loss_mask.sum() == 5 and loss_mask[:8].sum() == 0
    assert audio_mask.sum() == 8 and text_mask.sum() == 6
    assert labels.sum() == 1 and labels[-2] == 1
    torch.testing.assert_close(features[1:4], ref)
    torch.testing.assert_close(features[8:13], target)


def test_accumulation_matches_full_target_frame_mean_including_partial_group():
    x = torch.arange(1., 12.)
    full_parameter = torch.tensor(.3, requires_grad=True)
    ((full_parameter * x - 1) ** 2).mean().backward()
    accumulated = torch.tensor(.3, requires_grad=True)
    for chunk in (x[:2], x[2:9], x[9:]):
        result = {'loss/diff': ((accumulated * chunk - 1) ** 2).mean(), 'loss/stop': torch.tensor(0.)}
        weighted_loss(result, len(chunk), len(x)).backward()
    torch.testing.assert_close(accumulated.grad, full_parameter.grad)


def test_setting_d_matches_speech_generator_and_merged_monitoring():
    prep = read(ROOT / 'preparation.json')
    assert prep['train']['utterances'] == 2108 and prep['train']['patients'] == 57
    assert prep['merged_test']['utterances'] == 511
    assert prep['merged_test']['original_splits'] == {'dev': 258, 'test': 253}
    assert not any(prep['train_test_overlap'].values())
    rows = read(ROOT / 'data.json')
    train = {r['uid']: r for r in rows if r['partition'] == 'train'}
    for row in rows:
        if row['partition'] == 'merged_test':
            ref = train[row['ref_uid']]
            assert ref['speaker_id'] == row['speaker_id'] and ref['uid'] != row['uid']
    assert CONFIG['input_sample_rate'] == 16000 and CONFIG['output_sample_rate'] == 48000
    assert CONFIG['lora']['enable_lm'] and CONFIG['lora']['enable_dit'] and not CONFIG['lora']['enable_proj']


def test_frozen_feature_cache_receipt_and_size():
    receipt = read(ROOT / 'cache_receipt.json')
    assert receipt['utterances'] == 2619
    assert receipt['raw_packer_matches_inference_encoder']
    assert receipt['encoder_max_abs_error'] < 1e-5
    assert receipt['bytes'] < 200 * 1024**2
