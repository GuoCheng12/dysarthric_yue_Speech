#!/usr/bin/env python3
"""Sample a composed residual ASR while keeping initial R-only independently bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from therapist_harness.preference_data import (  # noqa: E402
    build_preference_pairs, sample_seed, sft_rows, validate_sampling_inputs,
)
from therapist_harness.store import (  # noqa: E402
    sha256_directory, sha256_file, write_immutable_bytes, write_immutable_json,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_immutable_bytes(path, ("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)).encode())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sample", "pairs"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--real-train", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, help="JSONL reviews keyed by candidate_audio_sha256; alternatively embed row.review")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--current-adapter", type=Path)
    parser.add_argument("--initial-r-only-adapter", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--include-greedy", action="store_true", help="Separate normal decode; never replaces the four stochastic draws")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates", type=Path, help="pairs stage: candidates.jsonl produced by sample")
    parser.add_argument("--sampling-receipt", type=Path, help="pairs stage: sampling_receipt.json produced by sample")
    return parser.parse_args(argv)


def sample_current_asr(inputs: list[dict], args, output_dir: Path) -> tuple[list[dict], str]:
    """Use the maintained audio processor/prompt/parser with stochastic HF generation."""
    import torch
    sys.path.insert(0, str(REPO / "finetune/scripts"))
    from qwen3_asr_residual_dpo import load_merged_parent, add_residual
    from qwen_asr.inference.utils import normalize_audio_input, parse_asr_output
    from transformers import set_seed

    if args.base_model is None or args.current_adapter is None:
        raise ValueError("sampling requires --base-model and --current-adapter")
    if args.samples < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("stochastic sampling requires samples>0, temperature>0, and 0<top_p<=1")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    parent_hash = sha256_directory(args.current_adapter)
    if args.expected_parent_sha256 and parent_hash != args.expected_parent_sha256:
        raise ValueError("current parent differs from the frozen expected SHA256")
    base_hash = sha256_directory(args.base_model)
    initial_hash = sha256_directory(args.initial_r_only_adapter)
    warm = args.current_adapter.resolve() != args.initial_r_only_adapter.resolve()
    if warm:
        composition = json.loads((args.current_adapter / 'residual_composition.json').read_text())
        assert composition['parent_adapter_sha256'] == initial_hash
        assert Path(composition['base_model']).resolve() == args.base_model.resolve()
    runtime_source = REPO / 'finetune/scripts/qwen3_asr_residual_dpo.py'
    runtime_hash = sha256_file(runtime_source)
    config = {"schema_version": "current_asr_sampling_v1", "current_asr_sha256": parent_hash,
              "current_adapter": str(args.current_adapter.resolve()), "base_model": str(args.base_model.resolve()),
              "base_model_sha256": base_hash, "samples_per_audio": args.samples, "seed": args.seed,
              "temperature": args.temperature, "top_p": args.top_p, "top_k": 0,
              "max_new_tokens": args.max_new_tokens, "language": "Cantonese", "prompt": "",
              "do_sample": True, "sampling_scope": "frozen current parent; DPO reference model is separately frozen at round zero",
              "input_manifest_sha256": sha256_file(args.inputs), "real_train_sha256": sha256_file(args.real_train),
              "feedback_sha256": sha256_file(args.feedback), "source_sha256": sha256_file(Path(__file__)),
              "preference_code_sha256": sha256_file(REPO / "src/therapist_harness/preference_data.py")}
    config.update(schema_version='current_asr_sampling_residual_v1',
        initial_R_only_adapter=str(args.initial_r_only_adapter.resolve()), initial_R_only_sha256=initial_hash,
        initial_R_only_merged=True, current_residual_merged=False, learned_residual_loaded=warm,
        composition_source_sha256=runtime_hash)
    write_immutable_json(output_dir / "sampling_config.json", config)
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    wrapper = load_merged_parent(str(args.base_model), str(args.initial_r_only_adapter),
                                args.device, dtype, batch_size=1, max_tokens=args.max_new_tokens)
    wrapper.model = add_residual(wrapper.model, str(args.initial_r_only_adapter), args.current_adapter if warm else None)
    wrapper.model.eval()
    wrapper.model.requires_grad_(False)
    candidates = []
    greedy = []
    with (output_dir / "candidates.jsonl").open("x", encoding="utf-8") as handle:
        for row in inputs:
            wav = normalize_audio_input(row["audio"])
            prompt = wrapper._build_text_prompt(context="", force_language="Cantonese")
            features = wrapper.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
            features = features.to(wrapper.model.device).to(wrapper.model.dtype)
            prefix_length = features["input_ids"].shape[1]
            if getattr(args, "include_greedy", False):
                with torch.inference_mode():
                    normal = wrapper.model.generate(**features, do_sample=False, num_beams=1,
                                                    max_new_tokens=args.max_new_tokens)
                normal_tokens = normal.sequences[0, prefix_length:].tolist()
                normal_raw = wrapper.processor.batch_decode([normal_tokens], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)[0]
                _, normal_text = parse_asr_output(normal_raw, user_language="Cantonese")
                greedy.append(dict(audio_id=row["audio_id"], audio_sha256=row["audio_sha256"],
                    current_asr_sha256=parent_hash, prediction=normal_text, raw_output=normal_raw,
                    language="Cantonese", prompt="", generated_token_ids=normal_tokens,
                    ended_with_eos=bool(normal_tokens and normal_tokens[-1] in {151645,151643}),
                    reached_max_new_tokens=len(normal_tokens)>=args.max_new_tokens,
                    do_sample=False, num_beams=1))
            for index in range(args.samples):
                seed = sample_seed(args.seed, row["audio_id"], index)
                set_seed(seed)
                with torch.inference_mode():
                    output = wrapper.model.generate(**features, do_sample=True, temperature=args.temperature,
                                                    top_p=args.top_p, top_k=0, max_new_tokens=args.max_new_tokens)
                tokens = output.sequences[0, prefix_length:].tolist()
                raw = wrapper.processor.batch_decode([tokens], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)[0]
                language, prediction = parse_asr_output(raw, user_language="Cantonese")
                # Qwen's outer generate returns a structured object itself and
                # stops on either chat-end or text-end, not only tokenizer EOS.
                eos_ids = {151645, 151643}
                candidate = {"schema_version": "asr_sample_v1", "audio_id": row["audio_id"],
                             "sample_id": f"{row['audio_id']}::{index}", "sample_index": index,
                             "seed": seed, "audio_sha256": row["audio_sha256"],
                             "current_asr_sha256": parent_hash, "raw_output": raw,
                             "prediction": prediction, "language": "Cantonese", "detected_language": language,
                             "prompt": "", "generated_token_ids": tokens,
                             "ended_with_eos": bool(tokens and tokens[-1] in eos_ids),
                             "reached_max_new_tokens": len(tokens) >= args.max_new_tokens}
                candidates.append(candidate)
                handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                handle.flush()
            print(json.dumps({"sampled_audio": len(candidates) // args.samples, "total_audio": len(inputs)}), flush=True)
    if sha256_directory(args.current_adapter) != parent_hash:
        raise ValueError("current ASR adapter files changed during sampling")
    assert sha256_directory(args.initial_r_only_adapter) == initial_hash
    assert sha256_file(runtime_source) == runtime_hash
    greedy_receipt = {}
    if getattr(args, "include_greedy", False):
        write_jsonl(output_dir / "greedy.jsonl", greedy)
        greedy_receipt = dict(greedy_sha256=sha256_file(output_dir / "greedy.jsonl"),
                              greedy_count=len(greedy), greedy_config=dict(do_sample=False,num_beams=1,
                              language="Cantonese",prompt="",max_new_tokens=args.max_new_tokens))
    write_immutable_json(output_dir / "sampling_receipt.json", {
        **config, "status": "complete", "candidate_count": len(candidates),
        "candidates_sha256": sha256_file(output_dir / "candidates.jsonl"),
        "validated_inputs_sha256": sha256_file(output_dir / "validated_inputs.jsonl"),
        "parent_files_unchanged": True,
        **greedy_receipt,
    })
    return candidates, parent_hash


def main(argv=None):
    args = parse_args(argv)
    raw_inputs = read_jsonl(args.inputs)
    inputs = validate_sampling_inputs(raw_inputs, read_jsonl(args.real_train),
                                      read_jsonl(args.feedback), read_jsonl(args.reviews) if args.reviews else ()) if raw_inputs else []
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out_dir / "validated_inputs.jsonl", inputs)
    if args.stage == "sample":
        candidates, parent_hash = sample_current_asr(inputs, args, args.out_dir)
    else:
        if args.candidates is None or args.sampling_receipt is None:
            raise ValueError("pairs stage requires candidates and their sampling receipt")
        receipt = json.loads(args.sampling_receipt.read_text())
        if receipt.get("status") != "complete" or receipt["candidates_sha256"] != sha256_file(args.candidates):
            raise ValueError("incomplete or changed sampling artifact")
        if receipt["validated_inputs_sha256"] != sha256_file(args.out_dir / "validated_inputs.jsonl"):
            raise ValueError("pair construction inputs differ from sampled inputs")
        if receipt["samples_per_audio"] != args.samples:
            raise ValueError("samples differs from the completed sampling receipt")
        parent_hash = receipt["current_asr_sha256"]
        if args.expected_parent_sha256 and args.expected_parent_sha256 != parent_hash:
            raise ValueError("samples came from a different current ASR parent")
        candidates = read_jsonl(args.candidates)
    if inputs:
        greedy = None
        receipt_path = args.out_dir / "sampling_receipt.json" if args.stage == "sample" else args.sampling_receipt
        sampling_receipt = json.loads(receipt_path.read_text())
        if "greedy_sha256" in sampling_receipt:
            greedy_path = receipt_path.parent / "greedy.jsonl"
            if sha256_file(greedy_path) != sampling_receipt["greedy_sha256"]:
                raise ValueError("normal decoding receipt changed")
            greedy = read_jsonl(greedy_path)
        pairs, audit = build_preference_pairs(inputs, candidates, parent_hash, expected_samples=args.samples,
                                             greedy_candidates=greedy)
    else:
        if candidates:
            raise ValueError("empty admission cannot have sampled candidates")
        pairs, audit = [], {"schema_version": "asr_preference_audit_v1", "current_asr_sha256": parent_hash,
            "audio_count": 0, "sample_count": 0, "pair_count": 0, "source_audio_counts": {},
            "paired_audio_count": 0, "all_correct_audio_count": 0, "audio_results": [],
            "scope": "no admitted audio; no ASR generation or preference pair"}
    write_jsonl(args.out_dir / "preferences.jsonl", pairs)
    write_jsonl(args.out_dir / "sft.jsonl", sft_rows(inputs))
    write_immutable_json(args.out_dir / "preference_audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "audio_results"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
