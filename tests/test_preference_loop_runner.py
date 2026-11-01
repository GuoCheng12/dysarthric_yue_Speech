"""CPU registry tests with generated fixture WAVs and explicitly fake reviews.

audio_inspected=True here exercises receipt validation only. It is not live
listening evidence, and no test fixture is registered in an experiment RunStore.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from therapist_harness.preference_contract import PreferenceDelivery, PreferenceLoopConfig, PreferenceSampleDelivery
from therapist_harness.preference_data import build_preference_pairs, validate_sampling_inputs
from therapist_harness.preference_loop import (
    build_branch_manifests, validate_registered_preferences, validate_registered_preference_training,
)
from therapist_harness.store import sha256_file


PARENT = "a" * 64


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False), "utf-8")


def _write_rows(path, values):
    Path(path).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), "utf-8")


def _read_rows(path):
    return [json.loads(x) for x in Path(path).read_text("utf-8").splitlines()]


def _wav(tmp_path, name, frequency):
    path = tmp_path / f"{name}.wav"
    t = np.arange(1600) / 16000
    sf.write(path, .1 * np.sin(2 * np.pi * frequency * t), 16000)
    return str(path)


@pytest.fixture
def registered(tmp_path):
    real = [
        {"audio": _wav(tmp_path, "r_correct", 300), "utt_id": "r_correct", "text": "今日食飯"},
        {"audio": _wav(tmp_path, "r_error", 400), "utt_id": "r_error", "text": "聽日去街"},
    ]
    feedback = [{"audio": _wav(tmp_path, "feedback", 500), "utt_id": "feedback", "text": "杯水好熱"}]
    for filename, data in (("real.jsonl", real), ("feedback.jsonl", feedback)):
        _write_rows(tmp_path / filename, data)
    assignments, reviews, synthetic = [], [], []
    for index, (sample_id, text, verdict) in enumerate([
        ("s_error", "姐姐買衫", "pass"),
        ("s_correct", "哥哥開門", "pass"),
        ("s_failed", "慢慢坐低", "obvious_failure"),
    ]):
        wav = _wav(tmp_path, sample_id, 650 + 100 * index)
        review = {
            "verdict": verdict, "reason": "TEST FIXTURE ONLY: coarse decision, no CER assessment.",
            "audio_inspected": True, "candidate_audio_sha256": sha256_file(Path(wav)),
            "target_text": text, "reference_audio_sha256": sha256_file(Path(real[0]["audio"])),
            "reference_text": real[0]["text"], "model": "mock-not-a-live-audio-review",
        }
        reviews.append(review)
        assignments.append({"sample_id": sample_id, "target_text": text, "output_wav": wav,
                            "reference_wav": real[0]["audio"],
                            "reference_wav_sha256": review["reference_audio_sha256"],
                            "reference_text": real[0]["text"]})
        if verdict == "pass":
            synthetic.append({"audio_id": sample_id, "audio": wav, "target_text": text,
                              "source": "synthetic", "reference_audio": real[0]["audio"],
                              "reference_text": real[0]["text"]})
    inputs = validate_sampling_inputs(
        [{**r, "source": "real"} for r in real] + synthetic, real, feedback, reviews
    )
    predictions = {
        "r_correct": "今日食飯", "r_error": "聽日去山",
        "s_correct": "哥哥開門", "s_error": "完全唔關事嘅另外一句說話",
    }
    candidates = [{"audio_id": row["audio_id"], "audio_sha256": row["audio_sha256"],
                   "current_asr_sha256": PARENT, "sample_index": index,
                   "prediction": predictions[row["audio_id"]]} for row in inputs for index in range(4)]
    pairs, audit = build_preference_pairs(inputs, candidates, PARENT)
    paths = {key: tmp_path / name for key, name in (
        ("preference_inputs", "inputs.jsonl"), ("preference_candidates", "candidates.jsonl"),
        ("preference_reviews", "reviews.jsonl"), ("preference_sampling_receipt", "sampling_receipt.json"),
    )}
    _write_rows(paths["preference_inputs"], inputs)
    _write_rows(paths["preference_candidates"], candidates)
    _write_rows(paths["preference_reviews"], reviews)
    real_ref = SimpleNamespace(path=str(tmp_path / "real.jsonl"), sha256=sha256_file(tmp_path / "real.jsonl"))
    feedback_ref = SimpleNamespace(path=str(tmp_path / "feedback.jsonl"), sha256=sha256_file(tmp_path / "feedback.jsonl"))
    protocol = SimpleNamespace(real_train=real_ref, feedback_split=feedback_ref,
        real_replay_count_per_round=2,
        preference_loop=PreferenceLoopConfig(audio_review_model="unused", audio_review_base_url="https://example.invalid"))
    receipt = {"status": "complete", "current_asr_sha256": PARENT,
               "candidates_sha256": sha256_file(paths["preference_candidates"]),
               "validated_inputs_sha256": sha256_file(paths["preference_inputs"]),
               "real_train_sha256": real_ref.sha256, "feedback_sha256": feedback_ref.sha256,
               "samples_per_audio": 4}
    _write_json(paths["preference_sampling_receipt"], receipt)
    return SimpleNamespace(protocol=protocol, paths=paths, pairs=pairs, inputs=inputs,
                           candidates=candidates, assignments=assignments, reviews=reviews,
                           receipt=receipt, audit=audit, real=real, feedback=feedback)


def _validate(item, *, pairs=None, assignments=None):
    return validate_registered_preferences(item.protocol, item.paths,
        item.pairs if pairs is None else pairs,
        item.assignments if assignments is None else assignments, PARENT)


def test_four_branches_keep_all_correct_audio_for_sft_and_skip_only_dpo_pairs(registered):
    item = registered
    _validate(item)
    arms = build_branch_manifests(item.inputs, item.pairs)
    assert list(arms) == ["r_sft", "r_dpo", "mas_sft", "mas_dpo"]
    assert [x["audio_id"] for x in arms["r_sft"]] == ["r_correct", "r_error"]
    assert [x["audio_id"] for x in arms["mas_sft"]] == ["r_correct", "r_error", "s_error", "s_correct"]
    assert [x["audio_id"] for x in arms["r_dpo"]] == ["r_error"]
    assert [x["audio_id"] for x in arms["mas_dpo"]] == ["r_error", "s_error"]
    assert arms["r_sft"] == [x for x in arms["mas_sft"] if x["source"] == "real"]
    assert arms["r_dpo"] == [x for x in arms["mas_dpo"] if x["source"] == "real"]
    assert item.audit["all_correct_audio_count"] == 2
    assert item.audit["pair_count"] == 2  # Four copies of the same error remain one pair.
    assert all("s_failed" != x["audio_id"] for rows in arms.values() for x in rows)


def test_large_current_asr_error_is_used_as_rejected_not_as_sg_quality_filter(registered):
    _validate(registered)
    pair = next(x for x in registered.pairs if x["audio_id"] == "s_error")
    assert pair["chosen"] == "姐姐買衫"
    assert pair["rejected"] == "完全唔關事嘅另外一句說話"
    assert pair["sample_indices"] == [0, 1, 2, 3]
    assert pair["pair_weight"] == 1.0
    assert "not_word_verified" in pair["chosen_status"]


@pytest.mark.parametrize("mutation", ["status", "parent", "candidate_hash", "input_hash", "real_scope", "feedback_scope", "sample_count"])
def test_sampling_receipt_is_bound_to_current_parent_artifacts_and_scope(registered, mutation):
    item = registered
    changes = {
        "status": ("status", "incomplete"), "parent": ("current_asr_sha256", "b" * 64),
        "candidate_hash": ("candidates_sha256", "b" * 64),
        "input_hash": ("validated_inputs_sha256", "b" * 64),
        "real_scope": ("real_train_sha256", "b" * 64),
        "feedback_scope": ("feedback_sha256", "b" * 64), "sample_count": ("samples_per_audio", 8),
    }
    key, value = changes[mutation]
    _write_json(item.paths["preference_sampling_receipt"], {**item.receipt, key: value})
    with pytest.raises(ValueError):
        _validate(item)


def test_changed_actual_samples_cannot_keep_old_or_invented_preferences(registered):
    item = registered
    candidates = copy.deepcopy(item.candidates)
    next(x for x in candidates if x["audio_id"] == "s_error")["prediction"] = "姐姐買衫"
    _write_rows(item.paths["preference_candidates"], candidates)
    _write_json(item.paths["preference_sampling_receipt"], {
        **item.receipt, "candidates_sha256": sha256_file(item.paths["preference_candidates"])
    })
    with pytest.raises(ValueError, match="actual deduplicated errors"):
        _validate(item)
    rebuilt, _ = build_preference_pairs(item.inputs, candidates, PARENT)
    _validate(item, pairs=rebuilt)
    invented = copy.deepcopy(rebuilt)
    invented[-1]["rejected"] = "捏造嘅錯誤"
    with pytest.raises(ValueError, match="actual deduplicated errors"):
        _validate(item, pairs=invented)


@pytest.mark.parametrize("mutation", ["missing_failed_review", "failed_audio_uninspected", "failed_target", "failed_reference", "failed_reference_text", "invalid_verdict", "pass_to_fail", "fail_to_pass"])
def test_every_generated_assignment_needs_bound_review_and_admission_matches(registered, mutation):
    item = registered
    reviews = copy.deepcopy(item.reviews)
    if mutation == "missing_failed_review":
        reviews.pop()
    elif mutation == "failed_audio_uninspected":
        reviews[-1]["audio_inspected"] = False
    elif mutation == "failed_target":
        reviews[-1]["target_text"] = "其他句子"
    elif mutation == "failed_reference":
        reviews[-1]["reference_audio_sha256"] = "b" * 64
    elif mutation == "failed_reference_text":
        reviews[-1]["reference_text"] = "另外一段参考"
    elif mutation == "invalid_verdict":
        reviews[-1]["verdict"] = "unreviewed"
    elif mutation == "pass_to_fail":
        reviews[0]["verdict"] = "obvious_failure"
    else:
        reviews[-1]["verdict"] = "pass"
    _write_rows(item.paths["preference_reviews"], reviews)
    with pytest.raises(ValueError):
        _validate(item)


def test_feedback_content_and_reference_never_become_training_inputs(registered):
    item = registered
    inputs = copy.deepcopy(item.inputs)
    inputs[-1]["audio"] = item.feedback[0]["audio"]
    inputs[-1].pop("audio_sha256")
    with pytest.raises(ValueError, match="feedback audio"):
        validate_sampling_inputs(inputs, item.real, item.feedback, item.reviews)
    inputs = copy.deepcopy(item.inputs)
    inputs[-1]["reference_audio"] = item.feedback[0]["audio"]
    inputs[-1]["reference_text"] = item.feedback[0]["text"]
    with pytest.raises(ValueError, match="reference must belong"):
        validate_sampling_inputs(inputs, item.real, item.feedback, item.reviews)


def test_replay_selection_and_required_sources_are_not_optional(registered):
    item = registered
    item.protocol.real_replay_count_per_round = 3
    with pytest.raises(ValueError, match="real preference sampling count"):
        _validate(item)
    item.protocol.real_replay_count_per_round = 2
    item.paths.pop("preference_reviews")
    with pytest.raises(ValueError, match="lacks reviewed-audio"):
        _validate(item)


@pytest.fixture
def registered_training(registered, tmp_path):
    """Explicit trainer-artifact fixture; no model update or real cache is claimed."""
    from therapist_harness.dpo_training import preference_key
    item = registered
    p = item.protocol
    p.base_model_dir = str(tmp_path / "model")
    p.initial_adapter = SimpleNamespace(path=str(tmp_path / "initial_r"), sha256="f" * 64)
    p.lora = SimpleNamespace(optimizer_steps_per_round=2, learning_rate=1e-5, seed=42)
    p.preference_loop = p.preference_loop.model_copy(update={"real_per_step": 1, "synthetic_per_step": 1})
    for key, filename in (
        ("preference_training_protocol", "run_protocol.json"),
        ("preference_training_result", "training_result.json"),
        ("preference_exposures", "training_exposures.jsonl"),
        ("reference_log_probs", "reference_log_probs.json"),
        ("train_manifest", "train.jsonl"), ("synthesis_manifest", "assignments.jsonl"),
    ):
        item.paths[key] = tmp_path / filename
    item.paths["checkpoint"] = tmp_path / "checkpoint-2"
    item.paths["checkpoint"].mkdir()
    _write_rows(item.paths["train_manifest"], item.pairs)
    _write_rows(item.paths["synthesis_manifest"], item.assignments)
    # One paired audio in each source gives a transparent R,S schedule at each
    # step; pairless r_correct/s_correct and excluded s_failed never appear.
    exposures = [{"step": step, "source": pair["source"], "audio_id": pair["audio_id"],
                  "occurrence": 0, "preference_key": preference_key(pair),
                  "loss_weight": 1.0 if pair["source"] == "real" else 0.05}
                 for step in (1, 2) for pair in item.pairs]
    _write_rows(item.paths["preference_exposures"], exposures)
    run = {"arguments": {
        "objective": "dpo", "model_path": p.base_model_dir,
        "reference_adapter_path": p.initial_adapter.path, "max_steps": 2,
        "real_per_step": 1, "synthetic_per_step": 1, "synthetic_weight": 0.05,
        "beta": 0.1, "lr": 1e-5, "seed": 42,
    }, "reference_asr_sha256": p.initial_adapter.sha256, "policy_parent_sha256": PARENT,
        "training_file_sha256": sha256_file(item.paths["train_manifest"])}
    _write_json(item.paths["preference_training_protocol"], run)
    run_digest = sha256_file(item.paths["preference_training_protocol"])
    _write_json(item.paths["reference_log_probs"], {"protocol_sha256": run_digest, "reference_unchanged": True})
    _write_json(item.paths["preference_training_result"], {
        "protocol_sha256": run_digest, "reference_files_unchanged": True,
        "status": "complete", "checkpoint": str(item.paths["checkpoint"]),
    })
    delivery = PreferenceDelivery(
        current_asr_sha256=PARENT, frozen_reference_asr_sha256=p.initial_adapter.sha256,
        reviews_sha256=sha256_file(item.paths["preference_reviews"]),
        candidates_sha256=sha256_file(item.paths["preference_candidates"]),
        pair_manifest_sha256=sha256_file(item.paths["train_manifest"]),
        exposure_log_sha256=sha256_file(item.paths["preference_exposures"]),
        samples_per_audio=4, real_audio_considered=2, real_pair_audio_count=1,
        real_pair_count=1, real_training_exposures=2,
        synthetic_samples=[PreferenceSampleDelivery(
            sample_id=x["sample_id"], review_verdict=r["verdict"],
            pair_count=int(x["sample_id"] == "s_error"),
            training_exposures=2 if x["sample_id"] == "s_error" else 0,
        ) for x, r in zip(item.assignments, item.reviews)],
    )
    item.training = SimpleNamespace(parent_checkpoint_directory_sha256=PARENT,
        train_manifest_file_sha256=sha256_file(item.paths["train_manifest"]), preference_delivery=delivery)
    item.training_run = run
    return item


def test_trainer_artifacts_bind_exposure_dose_and_frozen_reference(registered_training):
    item = registered_training
    validate_registered_preference_training(item.protocol, item.paths, item.training)
    assert item.training.preference_delivery.real_training_exposures == 2
    assert [s.training_exposures for s in item.training.preference_delivery.synthetic_samples] == [2, 0, 0]


def test_executed_reference_cannot_follow_current_parent(registered_training):
    item = registered_training
    _write_json(item.paths["preference_training_protocol"], {
        **item.training_run, "reference_asr_sha256": PARENT,
    })
    with pytest.raises(ValueError, match="reference or parent"):
        validate_registered_preference_training(item.protocol, item.paths, item.training)


def test_rehashed_exposure_tamper_cannot_change_training_weight(registered_training):
    item = registered_training
    changed = _read_rows(item.paths["preference_exposures"])
    changed[1]["loss_weight"] = 1.0  # Inflate S weight beyond the frozen 0.05.
    _write_rows(item.paths["preference_exposures"], changed)
    item.training.preference_delivery = item.training.preference_delivery.model_copy(update={
        "exposure_log_sha256": sha256_file(item.paths["preference_exposures"])
    })
    with pytest.raises(ValueError, match="exposure sequence/weights"):
        validate_registered_preference_training(item.protocol, item.paths, item.training)
