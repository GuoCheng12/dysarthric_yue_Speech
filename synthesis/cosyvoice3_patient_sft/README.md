# Patient Speech Generator

The maintained synthesizer is the Setting D CosyVoice3 LLM LoRA checkpoint selected at
epoch 8. Flow and HiFT remain frozen. Its training provenance and aggregate dev result
are recorded in `SPEECH_GENERATOR.md` and `conf/speech_generator_epoch8.yaml`.

`synthesize_manifest.py` is the only maintained inference entrypoint. It accepts a
frozen JSONL manifest whose rows contain:

```json
{"sample_id":"r01-s01","target_text":"...","reference_text":"...","reference_wav":"/absolute/reference.wav","reference_wav_sha256":"...","output_wav":"/absolute/output.wav","seed":42}
```

The entrypoint fixes hashes for the Speech Generator, CosyVoice model components,
Transformers overlay, CosyVoice source, references, and local text frontend. It validates
the entire manifest before loading the model, calls CosyVoice once per row, applies
fail-fast technical audio checks, and publishes the batch only after every row succeeds.
There is no base-model mode, retry, refill, alternate reference, or silent fallback.

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python \
  synthesis/cosyvoice3_patient_sft/synthesize_manifest.py \
  --manifest /absolute/frozen_synthesis_manifest.jsonl \
  --summary /absolute/synthesis_summary.json
```

Manifest construction belongs to `therapist_harness`, where proposal evidence,
budgets, patient profiles, references, and content isolation are checked before this
boundary is called.
