from collections import Counter

import pytest

from therapist_harness.training_allocation import compile_role_allocation


def test_role_budget_uses_eligible_audio_and_accounts_for_pairless_maintenance():
    sidecar={'samples':[dict(sample_id='a',role='target'),dict(sample_id='b',role='target'),
                        dict(sample_id='c',role='protection'),dict(sample_id='d',role='maintenance')]}
    pairs=[dict(audio_id='a',source='synthetic'),dict(audio_id='a',source='synthetic'),
           dict(audio_id='c',source='synthetic')]
    allocation=compile_role_allocation(sidecar,pairs,steps=3,per_step=4,seed=42)
    assert allocation['role_exposures']=={'target':8,'protection':4}
    assert allocation['unavailable_roles']==['maintenance']
    assert Counter(x for step in allocation['steps'] for x in step)=={'a':8,'c':4}
    assert all(len(step)==4 for step in allocation['steps'])


def test_no_pairs_cannot_manufacture_delivery():
    with pytest.raises(ValueError):compile_role_allocation({'samples':[]},[],1,2,42)
