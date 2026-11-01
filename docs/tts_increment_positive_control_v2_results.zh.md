# 患者 TTS 真实增量正对照：结果与验收

2026-09-06。四组 × 三个训练 seed，12 次训练与真实语音评测全部完成，验收通过。

## 结论

**当前 Speech Generator 比未微调的 CosyVoice3 更适合患者 ASR，但没有产生超出真实数据重复采样的稳定 overall 收益，不能当作有效的真实录音增量替代品。**

这个实验已经区分了两个问题：

1. 同样的训练方法能够利用新增真实录音：真实 Q 在三个 seed 上都改善 overall、easy 和 hard。
2. 将这些录音替换成同患者、同 GT 的合成语音，overall 收益基本消失。问题不能仅归为“这套训练完全吸收不了新增数据”。

## 实验设计

不采集新真实语音。从现有 2117 条训练数据中划分共同集 R=1541 和真实增量 Q=576。

- replay：R + 从 R 重复采样的 576 条；逐条匹配 Q 的患者和难度。
- real_q：R + 真实 Q，共 2117 条真实录音。
- patient_tts：R + Speech Generator 按 Q 患者/GT 合成的 576 条。
- base_tts：R + 原始 CosyVoice3 按同样患者/GT 合成的 576 条。

两种 TTS 使用同一 reference 和 seed；reference 全部来自 R，不使用 Q 录音作为 reference。Q 覆盖 46 位患者，包含评测集全部 36 位患者。所有组训练清单为 2117 条，48 个既有 train content；评测固定为 merged real 511 条、12 个不重叠 content。

现有 Speech Generator 本身在原 2117 条上微调过（包括 Q）；本实验检验的是已有生成器的信息向 ASR 转移，不是 TTS 未见 Q 的泛化。

所有 ASR 从原始 Qwen3-ASR-1.7B 初始化新 LoRA，batch 8 × grad accumulation 4，LR 2e-4，LoRA r16/alpha4/dropout0.05，已修复梯度归一化。沿用 140-step linear schedule，但所有模型固定在 step 50 停止并评测，不选择 best checkpoint。训练 seed 为 42、43、44。

## 三 seed 平均结果

主指标为 mean utterance CER，以下单位均为 %。

| 训练增量（都加在 R 上） | Overall | Easy | Medium | Hard |
| --- | ---: | ---: | ---: | ---: |
| 576 条真实重复采样 | 22.0719 | 7.3445 | 24.5126 | 57.2093 |
| 576 条新增真实 Q | **21.6815** | **7.1811** | **24.3487** | **55.8979** |
| 576 条患者 TTS | 22.1111 | 7.2944 | 24.8663 | 57.0312 |
| 576 条基础 CosyVoice | 24.7050 | 8.6752 | 27.0916 | 63.3372 |

### 真实增量有用，而且方向稳定

真实 Q 相对 replay：overall -0.3904 pp、hard -1.3114 pp，均为 3/3 seed 改善。三个 seed 的 overall 差值分别为 -0.4905、-0.3349、-0.3459 pp。

### 患者合成增量没有复现真实增量的价值

患者 TTS 相对 replay：overall +0.0392 pp，基本持平，只有 1/3 seed 改善；三次差值为 +0.2980、-0.2621、+0.0817 pp。

hard 平均只下降 0.1781 pp（2/3 seed 改善），medium 上升 0.3537 pp。不能将这一结果描述为有效且稳定的 ASR 增强。

相对真实 Q，患者 TTS 的 overall 高 0.4296 pp，三个 seed 都更差；hard 平均高 1.1333 pp。

### 患者生成器微调确实改变了下游效用

患者 TTS 相对基础 CosyVoice：overall -2.5939 pp、hard -6.3060 pp，所有难度组均为 3/3 seed 改善。

因此不能说微调毫无作用。但“优于基础 TTS”和“优于重复采样真实数据”是不同的门槛：当前通过前者，没有通过后者。这个差异本身也不等同于定位到了具体的病理构音机制。

## 当前行动建议

先暂停以“扩大合成数量或继续增加 MAS 轮数”为主要投入。当前生成器的患者化是有效的，但 ASR 数据增量价值尚未建立。下一步应围绕真实 Q 与其合成版本的差异定位有效信息的损失，而不是继续用更多处方掩盖这个问题。

## 验收记录

- 两组各 576 条，全部在第一对 seed 成功，无重采样、删样本或 ASR 质量筛选。
- 患者 TTS 3802.16 秒（63.37 分钟）；基础 TTS 4591.68 秒（76.53 分钟）；真实 Q 3924.099 秒（65.40 分钟）。
- 12 个 checkpoint 均精确停在 step 50。
- 12 次评测均覆盖同一批 511 条真实语音，无重复或遗漏。
- 源码 SHA256、训练清单和合成音频 hash 核对通过；原始数据和 reference 清单 hash 未改变。
- 独立的两行动态规划编辑距离实现复核了 12 次 × 4 组 CER，与原 evaluator 一致。
- 启动前相关测试 9 项通过；所有训练进程正常退出，GPU 已释放。

逐 seed 指标与 patient macro / pooled CER：[完整数据报告](/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_increment_positive_control_v2/report.zh.md)。

机器可读验收：[acceptance.json](/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_increment_positive_control_v2/acceptance.json)。

运行目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_increment_positive_control_v2`。
