# 当前实验记录

## 主线结论

当前正式数据是 Setting D：`2117 / 258 / 253`，train/dev/test content 为
`48 / 6 / 6`，患者在训练中可见、评测 content 全局未见。Setting D 去除了
`vlink201`、`vlink203`、`vlink225`、`vlink243` 和一条已确认的 vlink211 噪声样本。

| 方法 | test overall CER | easy | medium | hard |
|---|---:|---:|---:|---:|
| Qwen3-ASR zero-shot | **24.39%** | **5.60%** | **28.89%** | 69.59% |
| Real-only LoRA | 24.93% | 8.39% | 29.68% | **63.44%** |
| Real + static Speech Generator 1440 | 25.21% | 8.16% | 30.21% | 64.73% |

真实数据微调和静态增强都呈现“hard 改善、easy/medium 退化”。因此当前问题不是
继续堆合成数据，而是让 Therapist Agent 根据真实语音上的 ASR 错误和上一轮干预
结果，决定下一轮有限预算的数据处方。

## 保留记录

- `historical_settings_summary.md`: Setting A/B/C 的最小聚合背景。
- `setting_d_lora_baseline_step_v1/`: Setting D zero-shot 与 real-only LoRA。
- `setting_d_static_augmentation_v1/`: 正式 Full-1440 与 matched patient/base TTS
  对照。
- `speech_generator_new60_design_v1/`: 当前高信息粤语刺激句设计。
- `rejected_approaches.md`: 已删除的 rulebook、Gate、verifier、旧 ASR/TTS 分支
  的压缩结论。

旧 E 系列 probe、逐患者表、逐条 transcript/prediction、患者语音、模型权重与
checkpoint 不进入 Git；需要追溯时使用仓库外清理归档。
