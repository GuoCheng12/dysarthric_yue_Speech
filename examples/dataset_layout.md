# Private Setting D layout

The scripts expect private data to live outside the git repository.

One working layout is:

```text
/data/qwen3-asr/
  models/
    Qwen3-ASR-1.7B/
  datasets/
    Speech_data/
      vlink_data_raw/
        SPEAKER_ID/
          transcript.txt
          *.wav
  finetune/
    data_setting_d_v1/
      setting_d_train.jsonl
      setting_d_dev.jsonl
      setting_d_test.jsonl
  records/
    cosyvoice3_reference_sft_setting_d_v1/
      speech_generator.pt
      therapist_reference_manifest_v1.json
```

The raw dataset, frozen Setting D rows, patient references, generated audio, and model
checkpoints are private artifacts outside this Git repository. Their identities and
hashes are registered in `data/registry/`, `artifacts/registry/`, and the frozen run
protocol.

Each Setting D JSONL row consumed by the maintained ASR evaluator provides:

- `utt_id`
- `speaker_id`
- `disease_tag`
- `audio` (absolute path)
- `clean_gt`
- `prompt_id`
- `zero_shot_bucket` (`easy`, `medium`, or `hard`)

The Therapist Agent never receives these private rows. The Harness reduces real dev
predictions to an anonymized aggregate `EvidencePack` before the Agent boundary.
