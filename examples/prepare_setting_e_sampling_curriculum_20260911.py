"""Prepare the real-only comparison and exercise the actual laicai Agent calls."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from therapist_harness.sampling_curriculum_data import make_sampling_groups, build_sampling_evidence
from therapist_harness.store import sha256_file, write_immutable_json, write_immutable_bytes

DATA = Path("/data/qwen3-asr/finetune/data_setting_e_v1")
DEFAULT_ROOT = Path("/data/qwen3-asr/records/therapist_mas_setting_e_v1/sampling_curriculum_dev_phonology_20260911_v2")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(root):
    train_path, dev_path, test_path = [DATA / f"setting_e_{name}.jsonl" for name in ("train", "dev", "test")]
    train, dev, test = [read_rows(path) for path in (train_path, dev_path, test_path)]
    if (len(train), len(dev), len(test)) != (1862, 394, 372):
        raise ValueError("expected canonical Setting E train1862/dev394/test372")
    parts = (train, dev, test)
    patient_sets = [{r["speaker_id"] for r in part} for part in parts]
    if any(patient_sets[i] & patient_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Setting E requires three patient-disjoint splits")
    if not {r["clean_gt"] for r in train} == {r["clean_gt"] for r in dev} == {r["clean_gt"] for r in test}:
        raise ValueError("Setting E requires the same target-text coverage")
    manifest_hashes = json.loads((DATA / "protocol.json").read_text())["output_manifests_sha256"]
    if any(sha256_file(p) != manifest_hashes[str(p)] for p in (train_path, dev_path, test_path)):
        raise ValueError("canonical Setting E data changed")
    expected_audio = json.loads((DATA / "audio_sha256.json").read_text())
    audio_hashes = {r["audio"]: sha256_file(Path(r["audio"])) for part in parts for r in part}
    if any(audio_hashes[path] != expected_audio[path] for path in audio_hashes):
        raise ValueError("audio file changed after the Setting E split")
    hashes = [{audio_hashes[r["audio"]] for r in part} for part in parts]
    if any(hashes[i] & hashes[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("splits share an audio hash")
    groups = make_sampling_groups(train)
    write_immutable_json(root / "groups.json", groups)
    write_immutable_json(root / "audio_sha256.json", audio_hashes)
    config = {
        "provider": "OpenAI", "base_url": "https://api.laicaishare.xyz",
        "model": "gpt-5.6-sol", "reasoning_effort": "xhigh", "store": False,
        "max_retries": 0, "timeout_seconds": 900, "max_output_tokens": 8192,
        "api_key_env": "OPENAI_API_KEY",
    }
    protocol = {
        "version": "setting_e_dev_phonology_sampling_v2", "setting": "E", "root": str(root.resolve()),
        "train_manifest": str(train_path), "feedback_manifest": str(dev_path),
        "test_manifest": str(test_path), "groups_path": str((root / "groups.json").resolve()),
        "feedback_role": "dev", "train_audio": 1862, "synthetic_audio": 0,
        "dev_audio": 394, "test_audio": 372, "train_patients": 46, "dev_patients": 10, "test_patients": 10, "shared_texts": 60,
        "seed": 42, "global_batch": 32, "stages": 5, "steps_per_stage": 118,
        "total_optimizer_steps_limit": 590, "maximum_audio_exposures": 18880,
        "equivalent_full_training_pool_exposures": 18880 / 1862,
        "wall_time_budget_seconds": 1800, "intended_gpus_per_branch": 2,
        "base_model": "/data/qwen3-asr/models/Qwen3-ASR-1.7B", "initial_adapter": None,
        "training": {"objective": "utterance-mean cross entropy", "audio_lora_only": True,
                     "lora_rank": 16, "lora_alpha": 4, "lora_dropout": 0.05,
                     "learning_rate": 2e-4, "warmup_steps": 5, "schedule": "linear",
                     "optimizer": "AdamW", "weight_decay": 0.0, "microbatch": 4},
        "agent": config,
        "only_agent_action": "probability distribution over the frozen groups; uniform recording sampling within each group",
        "uniform_control": "continuous shuffled traversal without replacement; full global batches may cross traversal boundaries",
        "mas_sampling": "with replacement; actual recording, patient and group exposure counts retained",
        "evaluation": "greedy Cantonese, empty context, cap256, original batches of8, BF16 merged adapters with exact base restore",
        "test_use": "report only; no test transcripts, scores or patient metadata are included in MAS requests or experience",
        "checkpoint_acceptance": "fixed training-step/time endpoints; latest completed result at each endpoint, not best test CER",
        "time_comparison": "same GPU count; include model loading, training, evaluation, Agent waiting and checkpoint time; compare latest completed evaluation within the common elapsed budget",
        "historical_test_exposure": "the 372 test recordings have been examined in earlier experiments; not a new blind test",
        "input_sha256": {str(p): sha256_file(p) for p in (train_path, dev_path, test_path, DATA / "protocol.json", root / "groups.json", root / "audio_sha256.json")},
    }
    write_immutable_json(root / "protocol.json", protocol)
    zero_path = Path("/data/qwen3-asr/inference/outputs/setting_d_seen_patient_unseen_content_v1_qc_frozen/setting_d_manifest.csv")
    with zero_path.open() as handle:
        zero = {r["utt_id"]: r for r in csv.DictReader(handle)}
    predictions = []
    for row in dev:
        z = zero[row["utt_id"]]
        if z["clean_gt"] != row["clean_gt"] or z["speaker_id"] != row["speaker_id"]:
            raise ValueError("historical base prediction identity mismatch")
        predictions.append({"utt_id": row["utt_id"], "speaker_id": row["speaker_id"], "prediction": z["raw_pred"]})
    evidence = build_sampling_evidence(train, predictions, groups, feedback_rows=dev, round_index=0, checkpoint_step=0)
    evidence["measurement_source"] = "历史原始模型的独立dev录音转写，仅用于API接入smoke；正式训练会重新转写当前模型。"
    write_immutable_json(root / "api_smoke/evidence.json", evidence)
    write_immutable_json(root / "api_smoke/source.json", {
        "scope": "Agent integration only; no ASR update or performance conclusion",
        "prediction_source": str(zero_path), "source_sha256": sha256_file(zero_path),
        "dev_records_selected": len(predictions), "training_predictions_included": 0, "test_records_included": 0})
    summary = {"train": len(train), "dev": len(dev), "feedback": "dev only", "test": len(test),
               "groups": {gid: g["train_count"] for gid, g in groups.items()},
               "historical_difficulty": dict(Counter(r["zero_shot_bucket"] for r in train)),
               "audio_integrity_verified": True, "formal_training_started": False}
    write_immutable_json(root / "preparation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def api_smoke(root):
    from therapist_harness.laicai_credentials import load_laicai_key
    from therapist_harness.real_sampling_mas import choose_sampling
    from therapist_harness.schema import AgentConfig
    credential = load_laicai_key()
    protocol = json.loads((root / "protocol.json").read_text())
    evidence = json.loads((root / "api_smoke/evidence.json").read_text())
    selected = choose_sampling(AgentConfig.model_validate(protocol["agent"]), evidence,
                               directory=root / "api_smoke/prescription")
    write_immutable_json(root / "api_smoke/result.json", {"passed": True, "actual_api": True,
        "asr_training_run": False, "group_weights": selected["group_weights"],
        "seconds": selected["seconds"], "credential": credential, "sources": selected["sources"]})
    print(json.dumps({"passed": True, "actual_api": True, "seconds": selected["seconds"],
                      "group_weights": selected["group_weights"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "api-smoke"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    {"prepare": prepare, "api-smoke": api_smoke}[args.mode](args.root)
