# Private Dataset Layout

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
  inference/
    outputs/
  finetune/
    data/
    e2-pooled-sft-3epoch/
```

The raw dataset and generated per-utterance outputs are not included in this repository.

For Experiment 2, the private cleaning manifest should provide at least:

- `utt_id`
- `speaker_id`
- `disease_tag`
- `audio_path`
- `raw_gt`
- `clean_gt`
- `task_type`
- `duration`
- `zero_shot_cer`
- `zero_shot_critical`
- `split`
