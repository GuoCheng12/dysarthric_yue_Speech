import pytest

from therapist_harness.sampling_curriculum_data import (
    make_sampling_groups, build_sampling_evidence, build_sampling_outcome,
)


def examples():
    rows = [dict(utt_id=f"audio-{i}", speaker_id=f"patient-{i % 2}",
                 clean_gt="幫我開門" if i % 2 else "麻煩你幫我閂埋個窗呀",
                 zero_shot_bucket=level, duration=2.5)
            for i, level in enumerate(("easy", "easy", "medium", "medium", "hard", "hard"))]
    dev = [dict(r, utt_id="dev-"+r["utt_id"], speaker_id="dev-"+r["speaker_id"]) for r in rows]
    predictions = [dict(utt_id=r["utt_id"], prediction=r["clean_gt"]) for r in dev]
    return rows, dev, predictions, make_sampling_groups(rows)


def test_partition_exposures_and_group_links():
    rows, dev, predictions, groups = examples()
    exposure = {rows[0]["utt_id"]: 32}
    evidence = build_sampling_evidence(rows, predictions, groups, feedback_rows=dev, round_index=1,
                                      checkpoint_step=1, exposures=exposure)
    assert sum(g["train_count"] for g in groups.values()) == len(rows)
    assert sum(g["actual_exposures"] for g in evidence["groups"]) == 32
    assert all("actual_exposures" not in p for p in evidence["patient_observations"])
    assert sum(sum(c["training_practice"]["group_counts"].values()) for c in evidence["content_observations"]) == len(rows)
    assert evidence["feedback"]["current"]["CER_percent"] == 0
    assert evidence["feedback_role"] == "dev"


def test_reject_test_recording_and_duplicate_prediction():
    rows, dev, predictions, groups = examples()
    predictions[-1] = dict(utt_id="test-audio", prediction="幫我開門")
    with pytest.raises(ValueError, match="exactly the dev pool"):
        build_sampling_evidence(rows, predictions, groups, feedback_rows=dev, round_index=1, checkpoint_step=1)
    predictions[-1] = predictions[0]
    with pytest.raises(ValueError, match="exactly the dev pool"):
        build_sampling_evidence(rows, predictions, groups, feedback_rows=dev, round_index=1, checkpoint_step=1)


def test_outcome_measures_actual_changes_and_keeps_exposure():
    rows, dev, before, groups = examples()
    after = [dict(p) for p in before]
    after[0]["prediction"] = ""
    outcome = build_sampling_outcome(rows, before, after, groups, feedback_rows=dev, decision={}, round_index=2,
                                    checkpoint_step=2, exposures={rows[0]["utt_id"]: 32})
    evidence = outcome["evidence"]
    assert evidence["feedback"]["recording_transitions"]["worsened"] == 1
    assert evidence["feedback"]["previous"]["CER_percent"] == 0
    assert evidence["feedback"]["current"]["CER_percent"] > 0
    assert outcome["delivery"]["recording_exposures"] == 32


def test_training_predictions_and_training_patients_rejected():
    rows, dev, predictions, groups = examples()
    with pytest.raises(ValueError, match="exactly the dev pool"):
        build_sampling_evidence(rows, [dict(r, prediction=r["clean_gt"]) for r in rows], groups,
                                feedback_rows=dev, round_index=1, checkpoint_step=0)
    with pytest.raises(ValueError, match="patient-disjoint"):
        build_sampling_evidence(rows, predictions, groups, feedback_rows=rows, round_index=1, checkpoint_step=0)


def test_dev_phone_patterns_map_to_training_carriers_and_keep_improvements():
    rows, dev, predictions, _ = examples()
    for r in rows + dev:
        r["clean_gt"] = "南媽"
    before = [dict(p, prediction="藍馬") for p in predictions]
    after = [dict(p, prediction="南媽") for p in predictions]
    groups = make_sampling_groups(rows)
    outcome = build_sampling_outcome(rows, before, after, groups, feedback_rows=dev, decision={},
                                    round_index=1, checkpoint_step=1)
    patterns = outcome["evidence"]["phonology"]["patterns"]
    onset = next(p for p in patterns if p["level"] == "onset")
    assert (onset["target_unit"], onset["recognized_unit"]) == ("n", "l")
    assert onset["previous"]["events"] == 6 and onset["current"]["events"] == 0
    assert onset["training_practice"]["recordings"] == 6
    assert onset["training_practice"]["target_texts"] == ["南媽"]
    assert outcome["evidence"]["feedback"]["current"]["CER_percent"] == 0


def test_homophones_aliases_and_deletions_are_not_phone_confusions():
    from therapist_harness.sampling_phonology import transcript_contrasts
    events, homophones, _ = transcript_contrasts("詩", "師")
    assert not events and homophones == 1
    assert not transcript_contrasts("冇南", "無藍")[0][0].reference_character == "冇"
    assert not transcript_contrasts("南媽", "南")[0]
