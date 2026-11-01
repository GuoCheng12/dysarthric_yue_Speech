from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from investigate_confusion_curriculum import paired_events, pair_summary, signature_map
from therapist_harness.evidence import _read_snapshot


def snapshot(pairs):
    rows = [dict(utt_id=f'u{i}', speaker_id=f'p{i}', prompt_id=f'c{i}',
                 zero_shot_bucket=bucket, clean_gt=gt, prediction=hyp)
            for i, (gt, hyp, bucket) in enumerate(pairs)]
    return _read_snapshot(rows, role='fixture', prediction_column='prediction',
                          critical_cer_threshold=0.5)


def test_reciprocal_actual_recovery_and_regression():
    before = snapshot([('我買包', '我買飽', 'hard'), ('你買飽', '你買飽', 'easy')])
    after = snapshot([('我買包', '我買包', 'hard'), ('你買飽', '你買包', 'easy')])
    events, coverage, _ = paired_events(before, after)
    result = pair_summary(events, ('tone', '1', '2'))
    assert result['hard_net_recovery'] == 1
    assert result['easy_medium_reverse_net_harm'] == 1
    assert result['hard_recovered']['positions'] == 1
    assert coverage['raw_gt_positions'] == 6


@pytest.mark.parametrize('new_hyp', ['我買', '我買飯'])
def test_error_migration_is_not_recovery(new_hyp):
    before = snapshot([('我買包', '我買飽', 'hard')])
    after = snapshot([('我買包', new_hyp, 'hard')])
    events, _, _ = paired_events(before, after)
    assert not events[('tone', '1', '2')].get('recovered')
    assert len(events[('tone', '1', '2')]['migrated_to_other_error']) == 1


def test_multiple_component_change_not_primary():
    row = snapshot([('我買包', '我買飯', 'hard')])[0]
    assert not signature_map(row, single_component=True)
    assert len(signature_map(row, single_component=False)[2]) > 1


def test_cer_equivalent_variant_does_not_create_tone_error():
    before = snapshot([('我冇錢', '我冇錢', 'easy')])
    after = snapshot([('我冇錢', '我無錢', 'easy')])
    events, _, _ = paired_events(before, after)
    assert not events


def test_ambiguous_alignment_not_recovery():
    before = snapshot([('包包', '飽包', 'hard')])
    after = snapshot([('包包', '包', 'hard')])
    events, coverage, _ = paired_events(before, after)
    assert not events[('tone', '1', '2')].get('recovered')
    assert events[('tone', '1', '2')]['after_unresolved']
    assert coverage['incorrect__unresolved'] == 1


def test_metadata_mismatch_fails():
    before = snapshot([('我買包', '我買飽', 'hard')])
    after = snapshot([('我買飽', '我買飽', 'hard')])
    with pytest.raises(ValueError, match='metadata mismatch'):
        paired_events(before, after)
