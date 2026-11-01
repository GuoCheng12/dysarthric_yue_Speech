"""Exploratory checkpoint acceptance, separate from preserving failed experience."""

from pathlib import Path

from .store import sha256_file, sha256_directory

RULE = dict(rule_id="real_dev_v1", pooled_cer="strict_decrease",
            patient_macro_max_regression=0.001, hard_max_regression=0.005,
            uncertainty="report paired variability; no significance gate for exploratory continuation")


def accepts(delta):
    return (delta.overall_pooled_cer < 0
            and delta.patient_macro_cer <= RULE["patient_macro_max_regression"]
            and delta.hard_pooled_cer <= RULE["hard_max_regression"])


def initial_gate(pack, protocol):
    from .evidence import evaluate_snapshot
    from .evidence_pack import load_prediction_rows

    ref = protocol.initial_adapter_dev_predictions
    path = Path(ref.path)
    if sha256_file(path) != ref.sha256:
        raise ValueError('initial R-only evaluation changed')
    initial = evaluate_snapshot(load_prediction_rows(path),
        prediction_column=protocol.initial_adapter_prediction_column,
        critical_cer_threshold=protocol.evaluation.critical_cer_threshold)
    after = pack.response.after_metrics
    delta = dict(overall_pooled_cer=after.overall.pooled_cer - initial.overall.pooled_cer,
        patient_macro_cer=after.overall.patient_macro_cer - initial.overall.patient_macro_cer,
        hard_pooled_cer=after.hard.pooled_cer - initial.hard.pooled_cer)
    return dict(predictions_sha256=ref.sha256, delta=delta,
        accepted=delta['overall_pooled_cer'] < 0 and delta['patient_macro_cer'] <= .001 and delta['hard_pooled_cer'] <= .005)


def accepted_hashes(pack, *, protocol=None):
    p = pack.provenance
    accepted = accepts(pack.response.paired_delta)
    if protocol is not None and protocol.preference_loop.checkpoint_acceptance == 'real_dev_initial_parent_v2':
        accepted = accepted and initial_gate(pack, protocol)['accepted']
    if accepted:
        return p.new_checkpoint_directory_sha256, p.after_predictions_file_sha256
    return p.parent_checkpoint_directory_sha256, p.before_predictions_file_sha256


def make_decision(pack, *, parent, before, candidate, after, reference_sha256, protocol=None):
    accepted = accepts(pack.response.paired_delta)
    initial = None
    if protocol is not None and protocol.preference_loop.checkpoint_acceptance == 'real_dev_initial_parent_v2':
        initial = initial_gate(pack, protocol); accepted = accepted and initial['accepted']
    selected, predictions = (candidate, after) if accepted else (parent, before)
    p = pack.provenance
    selected_hash, prediction_hash = ((p.new_checkpoint_directory_sha256, p.after_predictions_file_sha256)
        if accepted else (p.parent_checkpoint_directory_sha256, p.before_predictions_file_sha256))
    if sha256_directory(Path(selected)) != selected_hash or sha256_file(Path(predictions)) != prediction_hash:
        raise ValueError("accepted endpoint differs from measured provenance")
    return dict(schema_version="checkpoint_promotion_v1" if initial is None else 'checkpoint_promotion_v2',
        rule=RULE if initial is None else dict(RULE, rule_id='real_dev_initial_parent_v2', compared_to='both initial R-only and current parent'), accepted=accepted,
        candidate_checkpoint=str(candidate), candidate_checkpoint_sha256=pack.provenance.new_checkpoint_directory_sha256,
        candidate_predictions=str(after), candidate_predictions_sha256=pack.provenance.after_predictions_file_sha256,
        parent_checkpoint=str(parent), parent_checkpoint_sha256=pack.provenance.parent_checkpoint_directory_sha256,
        before_predictions=str(before), before_predictions_sha256=pack.provenance.before_predictions_file_sha256,
        accepted_checkpoint=str(selected), accepted_checkpoint_sha256=selected_hash,
        accepted_predictions=str(predictions), accepted_predictions_sha256=prediction_hash,
        frozen_reference_sha256=reference_sha256, candidate_delta=pack.response.paired_delta.model_dump(mode="json"),
        experience_retained=True, **({'initial_R_only_gate': initial} if initial is not None else {}))
