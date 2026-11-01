from therapist_harness.material_sampling_mas import augment_material_outcome
from therapist_harness.material_selection_offline import build_index


def test_group_delivery_does_not_double_count_overlapping_focuses():
    train = [dict(utt_id=uid, audio=f'/{uid}.wav', speaker_id=f'train-{uid}',
                  clean_gt='木', duration=2, split='train') for uid in ['a', 'b']]
    index = build_index(train, [], {})
    focuses = [dict(focus_id=name, purpose=purpose, level='syllable', target_unit='muk6',
                   reason='检查读音', evidence_ids=['dev-one'])
               for name, purpose in [('focus-a', 'target'), ('focus-b', 'maintenance')]]
    outcome = dict(checkpoint_step=2, observation_ids=[], delivery={}, interpretation='',
        evidence={'feedback_role': 'dev', 'phonology': {}},
        decision=dict(diagnosis=dict(producer='diagnosis_agent', round_index=1,
            source_evidence_sha256='0'*64, summary='检查', focuses=focuses),
            selected_purposes={'b': ['focus-a', 'focus-b']}))
    dev = [dict(utt_id='dev-one', speaker_id='heldout', clean_gt='木')]
    before = [dict(utt_id='dev-one', prediction='北')]
    after = [dict(utt_id='dev-one', prediction='木')]
    result = augment_material_outcome(outcome, before, after, feedback_rows=dev,
        index=index, channels=['group_pool', 'selected', 'group_pool'], observed=['a', 'b', 'b'],
        groups={'easy': {'train_ids': ['a']}, 'hard': {'train_ids': ['b']}})
    assert result['delivery']['group_counts'] == {'easy': 1, 'hard': 2}
    assert result['delivery']['selected_group_counts'] == {'hard': 1}
    assert result['delivery']['channels'] == {'group_pool': 2, 'selected': 1}
    assert result['delivery']['actual_selected_unique'] == 1
    assert [f['matching_selected_ids_exposures'] for f in result['focus_outcomes']] == [2, 2]
    assert all(f['current']['cohort']['CER_percent'] == 0 for f in result['focus_outcomes'])
