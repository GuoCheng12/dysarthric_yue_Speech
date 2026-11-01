from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from therapist_harness.preference_contract import PreferenceLoopConfig
from therapist_harness.preference_data import (
    build_preference_pairs, sample_seed, sft_rows, target_text, validate_sampling_inputs,
)
from therapist_harness.store import sha256_file


PARENT = "a" * 64


@pytest.fixture
def corpus(tmp_path):
    def row(name, text, source="real"):
        audio = tmp_path / f"{name}.wav"
        audio.write_bytes(name.encode())
        return {"utt_id": name, "audio": str(audio), "text": text, "source": source}
    real = [row("real", "language Cantonese<asr_text>我想飲水")]
    feedback = [row("dev", "佢去公園")]
    synthetic = row("sg", "今日落雨", "synthetic")
    synthetic.update(reference_audio=real[0]["audio"], reference_text="我想飲水")
    review = {"verdict": "pass", "reason": "沒有非常明顯的生成失敗；這不代表逐字保真。",
              "audio_inspected": True, "candidate_audio_sha256": sha256_file(Path(synthetic["audio"])),
              "target_text": "今日落雨", "reference_audio_sha256": sha256_file(Path(real[0]["audio"])),
              "reference_text": "我想飲水", "model": "audio-review-test", "response_id": "fixture"}
    synthetic["review"] = review
    return real, feedback, synthetic


def draws(row, predictions):
    return [{"audio_id": row["audio_id"], "audio_sha256": row["audio_sha256"],
             "current_asr_sha256": PARENT, "sample_index": i, "prediction": prediction,
             "sample_id": f"{row['audio_id']}::{i}", "raw_output": prediction,
             "seed": sample_seed(42, row["audio_id"], i)} for i, prediction in enumerate(predictions)]


def test_all_correct_skips_dpo_but_keeps_sft(corpus):
    real, feedback, _ = corpus
    rows = validate_sampling_inputs(real, real, feedback)
    candidates = draws(rows[0], ["我想飲水", "我想饮水。", "我 想 飲 水", "我想飲水！"])
    pairs, audit = build_preference_pairs(rows, candidates, PARENT)
    assert pairs == []
    assert audit["all_correct_audio_count"] == 1
    assert sft_rows(rows)[0]["text"] == "language Cantonese<asr_text>我想飲水"


@pytest.mark.parametrize('normal,origin',[('我想飲水','stochastic_only'),('我想食飯','greedy_error')])
def test_greedy_is_a_separate_error_source_not_a_fifth_draw(corpus,normal,origin):
    real,feedback,_=corpus
    rows=validate_sampling_inputs(real,real,feedback)
    candidates=draws(rows[0],['我想食飯']*4)
    greedy=dict(audio_id=rows[0]['audio_id'],audio_sha256=rows[0]['audio_sha256'],
        current_asr_sha256=PARENT,do_sample=False,num_beams=1,language='Cantonese',prompt='',prediction=normal)
    pairs,audit=build_preference_pairs(rows,candidates,PARENT,greedy_candidates=[greedy])
    assert len(pairs)==1 and pairs[0]['pair_weight']==1
    assert audit['audio_results'][0]['error_origin']==origin
    assert audit['sample_count']==4
    with pytest.raises(ValueError):
        build_preference_pairs(rows,candidates,PARENT,greedy_candidates=[dict(greedy,current_asr_sha256='b'*64)])


def test_repeated_same_error_yields_one_pair_with_original_text(corpus):
    real, feedback, _ = corpus
    rows = validate_sampling_inputs(real, real, feedback)
    pairs, audit = build_preference_pairs(rows, draws(rows[0], ["我想食飯"] * 4), PARENT)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "我想飲水"
    assert pairs[0]["rejected"] == "我想食飯"
    assert pairs[0]["sample_indices"] == [0, 1, 2, 3]
    assert pairs[0]["pair_weight"] == 1
    assert audit["audio_results"][0]["correct_samples"] == 0


def test_normalized_error_deduplication_does_not_multiply_audio_weight(corpus):
    real, feedback, synthetic = corpus
    rows = validate_sampling_inputs([*real, synthetic], real, feedback)
    candidates = draws(rows[0], ["我想食饭。", "我想食飯", "我想訓覺", "我想飲水"])
    candidates += draws(rows[1], ["今日落雨"] * 4)
    pairs, audit = build_preference_pairs(rows, candidates, PARENT)
    assert len(pairs) == 2
    assert [p["pair_weight"] for p in pairs] == [0.5, 0.5]
    assert pairs[0]["rejected"] == "我想食饭。"  # No normalized labels substituted.
    assert pairs[0]["sample_indices"] == [0, 1]
    assert audit["source_audio_counts"] == {"real": 1, "synthetic": 1}
    assert len(sft_rows(rows)) == 2


def test_empty_actual_sample_is_valid_negative(corpus):
    real, feedback, _ = corpus
    rows = validate_sampling_inputs(real, real, feedback)
    pairs, _ = build_preference_pairs(rows, draws(rows[0], [""] * 4), PARENT)
    assert len(pairs) == 1 and pairs[0]["rejected"] == ""


def test_coarse_review_does_not_filter_hard_asr_or_claim_word_fidelity(corpus):
    real, feedback, synthetic = corpus
    rows = validate_sampling_inputs([synthetic], real, feedback)
    pairs, _ = build_preference_pairs(rows, draws(rows[0], ["完全不同"] * 4), PARENT)
    assert len(pairs) == 1
    assert pairs[0]["chosen_status"] == "intended_target_after_coarse_review_not_word_verified"
    assert pairs[0]["metadata"]["review"]["audio_inspected"] is True


@pytest.mark.parametrize("change", [
    {"verdict": "obvious_failure"}, {"audio_inspected": False},
    {"candidate_audio_sha256": "b" * 64}, {"target_text": "其他內容"},
    {"reference_audio_sha256": "b" * 64}, {"reference_text": "其他參考"},
])
def test_synthetic_requires_review_bound_to_audio_target_and_reference(corpus, change):
    real, feedback, synthetic = corpus
    synthetic["review"].update(change)
    with pytest.raises(ValueError):
        validate_sampling_inputs([synthetic], real, feedback)


def test_feedback_cannot_enter_sampling_even_with_renamed_id(corpus):
    real, feedback, _ = corpus
    forbidden = {**feedback[0], "utt_id": "renamed", "text": "我想飲水"}
    with pytest.raises(ValueError, match="feedback audio"):
        validate_sampling_inputs([forbidden], real, feedback)


def test_real_gt_must_match_authoritative_pool(corpus):
    real, feedback, _ = corpus
    altered = {**real[0], "text": "其他文字"}
    with pytest.raises(ValueError, match="not in the training pool"):
        validate_sampling_inputs([altered], real, feedback)


def test_synthetic_cannot_use_feedback_content_or_reference(corpus):
    real, feedback, synthetic = corpus
    synthetic["text"] = feedback[0]["text"]
    with pytest.raises(ValueError, match="feedback content"):
        validate_sampling_inputs([synthetic], real, feedback)
    synthetic["text"] = "今日落雨"
    synthetic["reference_audio"] = feedback[0]["audio"]
    synthetic["reference_text"] = feedback[0]["text"]
    with pytest.raises(ValueError, match="reference must belong"):
        validate_sampling_inputs([synthetic], real, feedback)


@pytest.mark.parametrize("mutation", ["parent", "audio", "count", "duplicate_index", "prompt"])
def test_candidate_lineage_and_complete_draw_count_are_enforced(corpus, mutation):
    real, feedback, _ = corpus
    rows = validate_sampling_inputs(real, real, feedback)
    candidates = draws(rows[0], ["錯誤"] * 4)
    if mutation == "parent": candidates[0]["current_asr_sha256"] = "b" * 64
    elif mutation == "audio": candidates[0]["audio_sha256"] = "b" * 64
    elif mutation == "count": candidates.pop()
    elif mutation == "duplicate_index": candidates[0]["sample_index"] = 1
    elif mutation == "prompt": candidates[0]["prompt"] = "The target is 我想飲水"
    with pytest.raises(ValueError):
        build_preference_pairs(rows, candidates, PARENT)


def test_seeds_stable_per_audio_and_distinct_per_draw():
    assert sample_seed(42, "audioA", 0) == sample_seed(42, "audioA", 0)
    assert len({sample_seed(42, a, i) for a in ("audioA", "audioB") for i in range(4)}) == 8


def test_cpu_pair_cli_rebuilds_only_completed_unchanged_samples(tmp_path, corpus):
    real, feedback, synthetic = corpus
    raw = [*real, synthetic]
    validated = validate_sampling_inputs(raw, real, feedback)
    candidates = draws(validated[0], ["錯誤"] * 4) + draws(validated[1], ["今日落雨"] * 4)
    from examples.sample_asr_preferences import write_jsonl
    for filename, records in [("inputs", raw), ("real", real), ("feedback", feedback),
                              ("validated", validated), ("candidates", candidates)]:
        write_jsonl(tmp_path / f"{filename}.jsonl", records)
    receipt = {"status": "complete", "candidates_sha256": sha256_file(tmp_path / "candidates.jsonl"),
               "validated_inputs_sha256": sha256_file(tmp_path / "validated.jsonl"),
               "samples_per_audio": 4, "current_asr_sha256": PARENT}
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    script = Path(__file__).resolve().parents[1] / "examples/sample_asr_preferences.py"
    result = subprocess.run([sys.executable, str(script), "pairs", "--inputs", str(tmp_path / "inputs.jsonl"),
                             "--real-train", str(tmp_path / "real.jsonl"), "--feedback", str(tmp_path / "feedback.jsonl"),
                             "--out-dir", str(tmp_path / "out"), "--candidates", str(tmp_path / "candidates.jsonl"),
                             "--sampling-receipt", str(tmp_path / "receipt.json")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "out/preference_audit.json").read_text())["pair_count"] == 1
    assert len((tmp_path / "out/sft.jsonl").read_text().splitlines()) == 2


def test_formatted_asr_target_is_stripped_once():
    assert target_text({"text": "language Cantonese<asr_text>我想飲水"}) == "我想飲水"


def review_config(**updates):
    return PreferenceLoopConfig(audio_review_model="text-review-fixture",
                                audio_review_base_url="https://example.invalid", **updates)


def test_default_audio_review_preserves_existing_config_serialization():
    config = review_config()
    payload = config.model_dump(mode="json")
    assert config.review_mode == "audio"
    assert "review_mode" not in payload
    assert not any(key.startswith("review_asr_") for key in payload)
    assert PreferenceLoopConfig.model_validate(payload).model_dump(mode="json") == payload


@pytest.mark.parametrize("missing", ["review_asr_adapter", "review_asr_adapter_sha256", "review_asr_base_model_sha256"])
def test_transcript_review_config_requires_frozen_transcriber_identity(missing):
    fields = {"review_mode": "asr_transcript", "review_asr_adapter": "/fixture/adapter",
              "review_asr_adapter_sha256": "a" * 64, "review_asr_base_model_sha256": "b" * 64}
    fields.pop(missing)
    with pytest.raises(ValidationError, match="frozen review adapter"):
        review_config(**fields)


def test_transcript_review_config_records_mode_and_both_model_hashes():
    fields = {"review_mode": "asr_transcript", "review_asr_adapter": "/fixture/adapter",
              "review_asr_adapter_sha256": "a" * 64, "review_asr_base_model_sha256": "b" * 64}
    payload = review_config(**fields).model_dump(mode="json")
    assert {key: payload[key] for key in fields} == fields
    with pytest.raises(ValidationError):
        review_config(**{**fields, "review_asr_adapter_sha256": "not-a-digest"})


def test_unknown_review_mode_cannot_bypass_audio_inspection(corpus):
    real, feedback, synthetic = corpus
    synthetic["review"].update(review_mode="text_only_unbound", audio_inspected=False)
    with pytest.raises(ValueError, match="audio inspection"):
        validate_sampling_inputs([synthetic], real, feedback)


@pytest.fixture
def transcript_corpus(corpus, tmp_path):
    real, feedback, synthetic = corpus
    path = tmp_path / "transcription.json"
    payload = {
        "status": "complete", "actual_transcription": True,
        "audio": synthetic["audio"], "audio_sha256": sha256_file(Path(synthetic["audio"])),
        "prediction": "完全不同", "raw_output": "language Cantonese<asr_text>完全不同",
        "model": "/fixture/base", "base_model_sha256": "b" * 64, "adapter_sha256": PARENT,
        "decode_config": {"language": "Cantonese", "context": "", "do_sample": False, "num_beams": 1},
        "ended_with_eos": True, "reached_max_new_tokens": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False))
    # Explicit counterfactual receipt for boundary tests. This fixture never
    # performs ASR or calls a provider, and is not an observed review result.
    synthetic["review"].update(
        review_mode="asr_transcript", audio_inspected=False, transcript_inspected=True,
        asr_transcription=payload["prediction"], asr_raw_output=payload["raw_output"],
        transcription_model=payload["model"], transcription_base_model_sha256="b" * 64,
        transcription_adapter_sha256=PARENT, transcription_receipt_path=str(path),
        transcription_receipt_sha256=sha256_file(path), prompt_sha256="c" * 64,
    )
    return real, feedback, synthetic, path, payload


@pytest.mark.parametrize("empty_transcript", [False, True])
def test_bound_transcript_review_admits_patient_errors_without_claiming_listening(
    transcript_corpus, empty_transcript,
):
    real, feedback, synthetic, path, payload = transcript_corpus
    if empty_transcript:
        payload.update(prediction="", raw_output="")
        path.write_text(json.dumps(payload))
        synthetic["review"].update(asr_transcription="", asr_raw_output="",
                                   transcription_receipt_sha256=sha256_file(path))
    rows = validate_sampling_inputs([synthetic], real, feedback)
    assert rows[0]["review"]["audio_inspected"] is False
    assert rows[0]["review"]["review_mode"] == "asr_transcript"
    assert rows[0]["chosen_status"] == "intended_target_after_coarse_review_not_word_verified"
    pairs, _ = build_preference_pairs(rows, draws(rows[0], ["另一個錯誤"] * 4), PARENT)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "今日落雨"
    assert pairs[0]["metadata"]["review"]["asr_transcription"] == payload["prediction"]


@pytest.mark.parametrize("change", [
    {"verdict": "obvious_failure"}, {"transcript_inspected": False}, {"audio_inspected": True},
    {"review_mode": "audio"}, {"target_text": "改過目標"}, {"reference_text": "改過參考"},
    {"asr_transcription": "冒充轉寫"}, {"transcription_adapter_sha256": "d" * 64},
])
def test_transcript_admission_still_requires_review_and_exact_artifact_bindings(transcript_corpus, change):
    real, feedback, synthetic, _path, _payload = transcript_corpus
    synthetic["review"].update(change)
    with pytest.raises(ValueError):
        validate_sampling_inputs([synthetic], real, feedback)


def test_sampling_reopens_transcription_receipt_instead_of_trusting_inline_review(transcript_corpus):
    real, feedback, synthetic, path, payload = transcript_corpus
    path.write_text(json.dumps({**payload, "prediction": "被改寫的轉寫"}))
    with pytest.raises(ValueError, match="receipt hash changed"):
        validate_sampling_inputs([synthetic], real, feedback)
