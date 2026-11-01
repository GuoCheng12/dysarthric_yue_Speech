# 48 条配对重建诊断：快速实验方案

目的：在不训练 ASR、不新建 MAS/Gate 的条件下，定位真实 Q 与文本条件合成之间的差异主要发生在哪一段。

从已冻结的 Q=576 中按既有难度抽取 48 条，easy/medium/hard 各 16 条。每个难度先覆盖不同患者，固定 seed=20260906；最终覆盖 30 位患者。最长真实音频 9.44 秒。选择不使用此次 ASR 输出或重建质量。

五种配对条件：

1. real：真实 Q。
2. patient_tts：上一实验已有的 Speech Generator 合成。
3. base_tts：上一实验已有的基础 CosyVoice 合成。
4. token_reconstruction：真实 Q → 原 tokenizer → Flow → vocoder，不运行 LLM。
5. vocoder_reconstruction：真实 Q → 官方前端 mel → vocoder，不运行 LLM/Flow。

真实 token 重建使用与正常 TTS 相同的 `frontend_zero_shot` reference 条件和 token/mel 长度对齐，而不是缺少这一步的旧 `frontend_vc`。复用同一条 R 内 reference，固定随机 seed，完整保留输出，不截断、不筛选、不重试。运行时禁止 LLM inference，确认确实走旁路。这里只使用 Q 原声做诊断，不将重建语音称为新内容增强。

解码器固定为正对照实验的 `replay/seed_42/checkpoint-50`，只训练过 R 及其重复采样，没有见过 Q 录音。沿用现有 Qwen3-ASR evaluator，batch=8、Cantonese、max_new_tokens=256。一次解码 240 条。这里的 CER 是输入音频上的识别误差，不是增强训练后的测试 CER。

自动输出：

- 五组 overall/easy/medium/hard mean CER 与 pooled CER。
- 每条原声和各版本的完整 ASR 转录。
- 原声正确/错误与各版本正确/错误的四格统计。
- 原声转录与重建转录的编辑距离，按 GT 长度归一化。
- 原声与各版本的时长比、静音代理比例变化、F0 中位数差（半音）。
- 每条音频 100 个归一化时间段的有声 F0 中位数，作为句级粗轮廓保存。

静音代理：16 kHz 单声道，25 ms RMS 窗、10 ms 帧移，阈值为本句 RMS 第95百分位下35 dB。它不是临床停顿检测或强制音节对齐。F0 使用 WORLD DIO+StoneMask，60–500 Hz；无声段记为缺测，不把零频率当作音高。句级轮廓不能直接作为音节声调错误结论。

判断原则：

- 不把更低 CER 自动当成更好的患者模仿；它可能表示原有识别困难被正常化。
- 如果真实 token 重建在错误位置和声学指标上明显更接近原声，而文本 TTS 偏离，优先调查文本到 token 的预测。
- 如果真实 token 重建已明显偏离，而 vocoder 重建接近原声，优先调查 tokenizer+Flow 联合路径，不能单独归罪于其中一个。
- 如果 vocoder 重建已偏离，先调查 mel/vocoder。
- 本次只定位候选差异，不把声学差异直接等同于 ASR 训练效用；效用来自上一轮12次正对照，后续单因素实验才能检验具体机制。

代码：`examples/tts_reconstruction_probe.py`。

运行目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_reconstruction_probe48_v1`。`protocol.json` 冻结方案，`selection.json` 冻结48条，逐条 `receipt.json` 保存重建路径/长度/hash，`results.json` 与 `report.zh.md` 汇总结果。
