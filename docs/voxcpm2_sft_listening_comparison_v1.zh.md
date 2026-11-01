# VoxCPM2 微调与旧 Speech Generator：固定六患者试听

日期：2026-09-06。结果目录：
`/data/qwen3-asr/records/voxcpm2_reference_sft_setting_d_v1/comparison_cosyvoice_v1/`。

## 配对条件

复用 VoxCPM2 checkpoint 监控时预先固定的六患者：vlink234、vlink210、vlink209（easy）；vlink106_SCA、vlink104_SCA、vlink103_SCA（hard）。每位一条目标，共六条、五种唯一 GT。所有版本使用同一目标 GT、同患者同一 train reference、seed 42；不重试、不筛选、不裁切生成结果。

两套 TTS 微调均使用相同的 2108 条 Setting D 真实训练语音。六条目标的音频、GT、Jyutping 均不与 TTS train 重叠。这里是已被训练监控使用的 merged-test 固定试听子集，不是新增独立测试集。

- CosyVoice：当前仍注册为 Speech Generator 的 epoch 8，LoRA r8/alpha16，LLM q/k/v/o；原生 reference 音频 + reference transcript 条件。固定 Transformers 4.51.3 overlay。
- VoxCPM2：原始 step 0，微调 step 132/330/396；reference-only，不传 reference transcript，不剪 VAD；10 inference steps，cfg 2，max_len 512。复用对应点首次生成的完整 WAV。
- 六条 CosyVoice 合成是本次新生成。共 30 条合成 + 6 条患者目标原声，统一由原始 Qwen3-ASR-1.7B 回译，无 ASR LoRA，language=Cantonese，batch 8，max_new_tokens 256。
- 两套 TTS 均不加显式香港粤语 instruction。相同 reference/GT/seed 不意味着两种架构的内部条件机制相同。

## 实测结果

CER 按字符汇总；声学指标为六条平均（一条/患者）。相似度对照的是同 GT 的真实目标语音，不是输入 reference。

| 版本 | CER ↓ | MCD-DTW dB ↓ | CAMPPlus ↑ | 时长绝对相对误差 ↓ | 低能量占比差 pp ↓ | F0 中位数差/半音 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 患者原声 | 27.03% | 0 | 1 | 0% | 0 | 0 |
| CosyVoice Speech Generator | 22.97% | 9.19 | 0.749 | 15.6% | 7.19 | 0.72 |
| VoxCPM2 原始 | 54.05% | 10.70 | 0.530 | 44.0% | 12.95 | 2.25 |
| VoxCPM2 step 132 | 29.73% | 9.86 | 0.674 | 34.5% | 12.21 | 0.78 |
| VoxCPM2 step 330 | 18.92% | 9.80 | 0.651 | 22.7% | 10.62 | 1.12 |
| VoxCPM2 step 396 | 18.92% | 9.89 | 0.640 | 29.6% | 12.29 | 0.87 |

三位 hard 患者的 corpus CER：原声 44.74%，SG 26.32%，Vox 原始 50.00%，step132 36.84%，step330 23.68%，step396 26.32%。hard 音色相似度：SG 0.738；Vox132/330/396 为 0.643/0.624/0.618。

结论：VoxCPM2 微调确实优于自身 zero-shot；后期 checkpoint 的内容回译最好。但它尚未在这些配对声学指标上超过旧 SG。旧 SG 的平均 MCD、CAMPPlus、时长误差、低能量占比差和 F0 中位数差均优于本次三个 Vox 微调点。用户目标是患者风格保留，因此不能因 Vox 的 CER 更低就直接替换旧 SG。

Vox 内部，step330 比最终 step396 在平均 MCD、音色相似度和时长匹配上更好，CER 相同；值得重点对听这两个点。这里没有把几种指标拼成一个人为总分。

## 指标口径

MCD-DTW：统一 16kHz；WORLD CheapTrick 功率谱包络，SPTK mel-cepstrum order24/alpha0.42，取 c1–c24 排除音量相关 c0；10ms 帧移。排除 25ms RMS 低于自身 95 分位数 35dB 的低能量帧（阈值下限 1e-7）。以欧氏距离做全局 DTW，沿路径平均后乘 `10*sqrt(2)/ln(10)`。不是将 MFCC 欧氏距离误称 MCD。[作者实现](https://github.com/MattShannon/mcd) · [SPTK 定义](https://pysptk.readthedocs.io/en/latest/generated/pysptk.conversion.sp2mc.html)

MCD 比较对齐后声学谱形；DTW 会消除部分节奏差异，因此同时报告全音频时长误差。低能量占比包括首尾静音，不声称是语言学停顿率。F0 使用 WORLD DIO+StoneMask（60–500Hz），比较有声帧 F0 中位数的绝对半音差。本批所有六条均有有效 F0。

CAMPPlus 衡量音色/说话人相似度；它同时用于 CosyVoice conditioning，不作为独立病理判官。CER 衡量回译内容，不是越低就越像患者。MCD 同样不是经过临床验证的 impairment 指标。采用分项而非单一总分，与构音障碍克隆研究同时关注可懂度、音色和韵律的思路一致。[相关研究](https://arxiv.org/html/2508.05102v1)

## 运行与验收

入口：`examples/voxcpm2_sft_comparison.py`，阶段 `prepare / cosy / assemble / acoustics / summarize / report`。
ASR 使用现有 `finetune/scripts/evaluate_asr_checkpoint.py`。所有输入、参考、音频哈希、模型运行时信息和逐句分数均保存；原训练代码、旧 SG 注册和基座环境不改。

声学度量专用新增依赖仅 `pysptk==1.0.1`，位于隔离目录 `/data/qwen3-asr/overlays/tts-comparison-metrics`，通过 PYTHONPATH 使用，没有安装进 ASR 虚拟环境。

五项单测通过：MCD identity、已知距离/对称单帧、DTW 时间重复与时长误差、固定版本列表、排除 c0 后全局增益不变性。HTML 提供 42 个全长播放器（6 reference + 36 对照），无音频预览截取，音频资产使用本地符号链接，不复制已有大文件。

HTTP 验收完成：页面 200；42 个播放器逐一通过完整 Content-Length、SHA256 和 WAV 帧数/时长检查，总计读回 35,189,104 bytes。新增六条 CosyVoice WAV 仅 4,412,640 bytes（4.21 MiB）。本地服务入口为 `http://127.0.0.1:8936/voxcpm2_sft/index.html`，复用之前的私有监听服务，未发布至外网。
