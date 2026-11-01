import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import tts_reference_policy as experiment


def test_reference_selection_on_real_split():
    selection = experiment.read(experiment.SOURCE / "selection.json")
    candidates = [{**r, "seconds": float(r["duration"]),
                   "duration_bin": experiment.duration_bin(float(r["duration"]))}
                  for r in selection["R"] if 3 <= float(r["duration"]) <= 12]
    targets = selection["Q"]
    chosen, audit = experiment.choose_references(candidates, targets)
    assert (chosen, audit) == experiment.choose_references(list(reversed(candidates)), list(reversed(targets)))
    assert len(chosen) == 576
    allowed = {r["utt_id"] for r in selection["R"]}
    for target in targets:
        a, b = (chosen[target["utt_id"]][arm] for arm in experiment.ARMS)
        for ref in (a, b):
            assert ref["utt_id"] in allowed
            assert ref["speaker_id"] == target["speaker_id"]
            assert not experiment.same_text(ref, target)
        assert a["duration_bin"] == b["duration_bin"]
    assert all(2 <= len(x["slots"]) <= 3 for x in audit.values())
    for speaker, detail in audit.items():
        assert sum(s["uses"] for s in detail["slots"]) == sum(q["speaker_id"] == speaker for q in targets)
    other, _ = experiment.choose_references(candidates, targets, seed=123)
    assert other != chosen


def test_two_reference_pool_cannot_cross_exclusions():
    def row(uid, text, cer):
        return {"utt_id": uid, "speaker_id": "patient", "prompt_normalized_sha16": text,
                "prompt_jyutping_sha16": text, "seconds": 5., "duration_bin": 0, "zero_shot_cer": cer}
    candidates = [row("r1", "a", 0), row("r2", "b", 1)]
    targets = [row("q1", "a", 0), row("q2", "b", 0)]
    chosen, audit = experiment.choose_references(candidates, targets)
    assert chosen["q1"]["random_multi"]["utt_id"] == chosen["q1"]["hard_multi"]["utt_id"] == "r2"
    assert audit["patient"]["different_reference_targets"] == 0
    with pytest.raises(ValueError, match="two distinct"):
        experiment.choose_references(candidates[:1], targets)


def test_audio_checks_separate_signal_validity_from_asr(tmp_path):
    path = tmp_path / "audio.wav"
    for wave, reason in [(np.zeros(16000), "near_silent"),
                         (np.ones(16000), "excessive_dc"),
                         (np.tile([1., -1.], 8000), "clipped"),
                         (np.array([.2]), "too_short")]:
        sf.write(path, wave, 16000, subtype="FLOAT")
        assert experiment.audio_features(path)["rejection"] == reason
    wave = .1 * np.sin(2 * np.pi * 150 * np.arange(16000) / 16000)
    sf.write(path, wave, 16000, subtype="FLOAT")
    features = experiment.audio_features(path)
    assert features["rejection"] is None and features["seconds"] == 1


def fake_pair():
    return {"sample_id": "q0000", "target": {"clean_gt": "目標"}, "seed": 42, "probe": True,
        "references": {a: {"reference_text": a, "reference_wav": "/reference.wav",
                           "reference_wav_sha256": "0" * 64}
                       for a in (*experiment.ARMS, "old_fixed")}}


class FakeTTS:
    ManifestSample = SimpleNamespace

    def __init__(self, failure):
        self.calls = []
        self.failure = failure

    def _stage_audio_once(self, sample, model):
        self.calls.append((sample.reference_text, sample.seed))
        if sample.reference_text == "random_multi" and sample.seed == 42:
            raise RuntimeError(self.failure)
        temporary = sample.output_wav.with_suffix(".tmp")
        temporary.write_bytes(f"{sample.reference_text}:{sample.seed}".encode())
        record = {**vars(sample), "reference_wav": str(sample.reference_wav),
                  "output_wav": str(sample.output_wav), "output_wav_sha256": experiment.sha256_file(temporary)}
        return record, temporary


def test_paired_retries_and_completed_resume(tmp_path):
    tts = FakeTTS("Speech Generator returned near-silent audio: test")
    pair = fake_pair()
    experiment.synthesize_pair(tmp_path, pair, None, tts)
    selected = experiment.read(tmp_path / "selected/q0000.json")
    assert selected["attempt"] == 1
    assert [s for a, s in tts.calls] == [42] * 3 + [10042] * 3
    assert {r["seed"] for r in selected["records"].values()} == {10042}
    experiment.synthesize_pair(tmp_path, pair, None, tts)
    assert len(tts.calls) == 6
    corrupt = copy.deepcopy(selected)
    corrupt["records"]["hard_multi"]["seed"] = 42
    with pytest.raises(AssertionError):
        experiment.validate_selected(pair, corrupt)
    Path(selected["records"]["hard_multi"]["output_wav"]).write_bytes(b"changed")
    with pytest.raises(AssertionError):
        experiment.validate_selected(pair, selected)


def test_cuda_failure_is_not_resampled(tmp_path):
    tts = FakeTTS("CUDA out of memory")
    with pytest.raises(RuntimeError, match="CUDA"):
        experiment.synthesize_pair(tmp_path, fake_pair(), None, tts)
    assert len(tts.calls) == 1
    assert not (tmp_path / "selected/q0000.json").exists()


def test_report_rejects_bad_metrics_and_produces_zero_paired_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "verify", lambda root: None)
    source_train = experiment.SOURCE / "patient_tts/train.jsonl"
    for arm in experiment.ARMS:
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "train.jsonl").symlink_to(source_train)
        for seed in experiment.SEEDS:
            source = experiment.SOURCE / "patient_tts" / f"seed_{seed}"
            dest = tmp_path / arm / f"seed_{seed}"
            dest.mkdir()
            result = experiment.read(source / "result.json")
            result["arm"] = arm
            experiment.save(dest / "result.json", result)
            experiment.save(dest / "execution.json", {"actual_stop_step": 50,
                "train_sha256": experiment.sha256_file(source_train)})
            (dest / "evaluation").symlink_to(source / "evaluation", target_is_directory=True)
    path = tmp_path / "hard_multi/seed_42/result.json"
    good = path.read_text()
    broken = json.loads(good)
    broken["metrics"]["hard"]["mean_utterance_cer"] += .01
    path.write_text(json.dumps(broken))
    with pytest.raises(AssertionError):
        experiment.report(tmp_path)
    assert not (tmp_path / "acceptance.json").exists()
    path.write_text(good)
    experiment.report(tmp_path)
    results = experiment.read(tmp_path / "results.json")
    assert results["comparisons"]["hard_multi_minus_random_multi"]["overall"]["seed_deltas_pp"] == [0, 0, 0]
    assert experiment.read(tmp_path / "acceptance.json")["new_runs"] == 6
