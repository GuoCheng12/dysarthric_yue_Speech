import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from train_mas576_hardref import select_alternatives


def row(uid, text, seconds, cer):
    return {'utt_id':uid,'speaker_id':'patient','seconds':seconds,
            'duration_bin':min(2,int((seconds-3)//3)), 'zero_shot_cer':cer,
            'prompt_normalized_sha16':text,'prompt_jyutping_sha16':text}


def test_excludes_old_text_and_prioritizes_matched_duration_over_cer():
    old=row('old','same',5,0)
    candidates=[row('old_duplicate','same',5,2),row('far','different_far',10,3),
                row('low','different_low',4,.1),row('high','different_high',5,.8)]
    assert [x['utt_id'] for x in select_alternatives(candidates,old)]==['high','low']


def test_sparse_patient_uses_existing_nearest_bin_without_repeating_reference():
    old=row('old','same',7,0)
    only=row('only','different',5,.2)
    assert select_alternatives([old,only],old)==[only]


def test_no_alternative_is_reported_instead_of_reusing_original():
    old=row('old','same',5,0)
    with pytest.raises(ValueError,match='No alternate train reference'):
        select_alternatives([old],old)
