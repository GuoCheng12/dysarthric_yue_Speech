from types import SimpleNamespace

import pytest

from therapist_harness.checkpoint_promotion import accepts, accepted_hashes
from therapist_harness.schema import EvidencePack, canonical_sha256
from therapist_harness.evidence_pack import build_evidence_pack
from test_therapist_reflection import mixed_round as mixed_round
from test_therapist_outcome import _prediction_rows
from test_therapist_memory import _summary


@pytest.mark.parametrize('overall,macro,hard,expected', [
    (-.001,0,0,True),(0,0,0,False),(.001,-.001,0,False),
    (-.001,.002,0,False),(-.001,0,.006,False),(-.001,.0005,.004,True)])
def test_frozen_acceptance_rule(overall,macro,hard,expected):
    delta=SimpleNamespace(overall_pooled_cer=overall,patient_macro_cer=macro,hard_pooled_cer=hard)
    assert accepts(delta)==expected


def test_rejected_candidate_keeps_experience_but_next_evidence_uses_parent(mixed_round):
    item=mixed_round
    summary=_summary(item)
    data=summary.model_dump(mode='json',exclude={'ledger_sha256'})
    data.update(accepted_checkpoint_sha256=item.protocol.initial_adapter.sha256,
                accepted_predictions_sha256=item.protocol.initial_adapter_dev_predictions.sha256)
    summary=type(summary).model_validate(dict(data,ledger_sha256=canonical_sha256(data)))
    before=_prediction_rows(after=False)
    p=item.protocol
    pack=build_evidence_pack(p,round_index=2,current_rows=before,current_prediction_column='before',
        current_checkpoint_sha256=p.initial_adapter.sha256,current_predictions_sha256=p.initial_adapter_dev_predictions.sha256,
        m0_rows=before,m0_prediction_column='before',m0_predictions_sha256=p.initial_adapter_dev_predictions.sha256,
        zero_shot_rows=before,zero_shot_prediction_column='before',zero_shot_predictions_sha256=p.zero_shot_dev_predictions.sha256,
        previous_rows=before,previous_prediction_column='before',previous_checkpoint_sha256=p.initial_adapter.sha256,
        previous_predictions_sha256=p.initial_adapter_dev_predictions.sha256,intervention_history=[summary])
    assert pack.snapshots.current.checkpoint_sha256==p.initial_adapter.sha256
    assert pack.comparisons.current_vs_previous.metric_delta.overall_pooled_cer==0
    assert pack.intervention_history[0].new_checkpoint_sha256==summary.new_checkpoint_sha256
    assert pack.intervention_history[0].episode_memory==summary.episode_memory


def test_legacy_ledger_serialization_unchanged(mixed_round):
    summary=_summary(mixed_round)
    assert 'accepted_checkpoint_sha256' not in summary.model_dump(mode='json')
