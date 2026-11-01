# VoxCPM2：Setting D 患者 reference-conditioned LoRA

## 目的与数据

从原始 `openbmb/VoxCPM2` 开始，学习同患者 reference 条件下的目标内容与患者发音。
不沿用 CosyVoice3 权重，不改 AudioVAE，不进行 ASR 微调。

复用已经下载并验证的基座 revision `32279effe8c19989596f05d353d1447f51d9e915`，路径 `/data/qwen3-asr/models/tts/comparison24_v1/voxcpm2`。

- 训练：与已选 Speech Generator 完全相同的 **2108 条、57 患者、4.05978 小时**。
- Setting D 原始 ASR train 有 2117 条；其中 9 位单录音患者不能提供不同 reference，旧 SG 已将这 9 条排除，本次保持一致。
- 监控：遵循用户后来的 train/test 口径，旧 dev 258 + test 253 = **511 条、36 患者**。
- 所有 reference 都来自上述训练集；训练每个 epoch 同患者随机重配且排除自身；评估沿用 SG 的固定 reference manifest。
- 与旧 SG 相同，这里按不同录音而非不同 GT 配对；3 轮分别有 16/13/13 对 reference 与目标同文，占比不到 1%，不涉及监控 content。
- train 与合并监控集的 content ID、归一化文本、Jyutping、GT、音频 SHA256 交集均为 0。
- vlink201/203/225/243 及指定 vlink211 噪声录音没有进入本次训练/评估。
- 合并集会反复监控并影响选点，不标作 untouched final test。

数据清单、输入哈希和配对在 `/data/qwen3-asr/records/voxcpm2_reference_sft_setting_d_v1/` 留档；原 WAV 不复制、不改写。

## 调研后采用的方法

1. **LoRA 起步**：官方示例是 LM（含 residual LM）与 DiT 的 q/k/v/o 投影同时加 LoRA，rank/alpha=32/32、dropout=0、LR=1e-4、weight decay=0.01、clip=1，投影桥接层不调。本次采用这些设置。[官方配置](https://github.com/OpenBMB/VoxCPM/blob/main/conf/voxcpm_v2/voxcpm_finetune_lora.yaml)
2. **16kHz 输入 / 48kHz 输出**：编码器采样率不能按输出采样率填写；本次加载时校验，音频统一重采样到 16kHz 后才进入冻结 AudioVAE。[官方微调指南](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)
3. **保留条件 dropout 与权重衰减，早点看生成结果**：官方 FAQ 提醒，小数据可能很早出现“记住训练音频但忽略文本”。本次保留 training_cfg_rate=0.1；不能仅因 train loss 降低便认定有效。[官方 FAQ](https://voxcpm.readthedocs.io/en/latest/finetuning/faq.html)
4. **香港粤语社区配置仅作实现参考**：该社区项目第一阶段同样用 rank 32 / LR 1e-4，第二阶段改为 rank 64 并使用合成训练数据；不能据此认定已验证病理发音保留。本次不采用它的二阶段合成训练。[社区项目](https://github.com/johnsonhk88/voxCPM2-hk-cantonese-fine-tuning) · [第二阶段配置](https://github.com/johnsonhk88/voxCPM2-hk-cantonese-fine-tuning/blob/main/configs/lora_phase2.yaml)

有意不照搬的通用 TTS 建议：

- 官方建议混合 30–50% reference 样本以兼顾多种生成模式；本任务只研究患者 reference-conditioned 生成，因此 **100% 带 reference**。
- 官方普通 TTS 数据建议限制长尾静音/过短片段；本次先保留 SG 原始语料与病理停顿，不自动去噪、VAD 剪切、按 CER 删除困难样本。

## 输入和 loss

直接复用官方 `process_tts_data_with_ref`：

```
[103, 同患者另一条 reference 音频特征, 104,
 目标粤语 GT 文本, 101, 目标真实音频特征, 102]
```

只在目标音频段计算 diffusion 与 stop loss。reference 是条件，不是训练标签；这一模式不需要 reference transcript。
推理使用相同的 reference-only 模式，隐式推断粤语，不增加香港话 instruction。
因此不能把本次试听与此前 ref+continuation 模式的 zero-shot 分数直接混作同一实验；这里另存相同 reference-only 模式的 step 0 对照。

## 首轮训练配置

- 3 epochs，effective batch=16，**132 optimizer steps/epoch，共 396 steps**。
- micro-batch=4，gradient accumulation=4；同患者最长 reference 配对实测峰值 allocated 27.31GiB / reserved 28.70GiB。batch 2 的对应峰值为 20.16/20.91GiB；预热后同等有效 batch 的预计计算时间由 2.77s 降到 2.53s（约快 9%）。没有改变有效 batch。
- 每条目标每个 epoch 恰好使用一次；最后不足 16 条的 batch 也训练，不丢掉最后 12 条。
- warmup=40 steps，之后 cosine LR；BF16 autocast、FP32 master weights、AdamW。
- AudioVAE 及所有非 LoRA 权重冻结。缓存的只是冻结 VAE 输出，不缓存可训练模块输出。
- 可训练参数 **18,087,936**；全部是 LoRA 参数。
- 梯度累积按整个 effective batch 的目标音频 patch 数加权，不简单叠加多个 micro-batch mean。
- 每 66 steps 完整评估 511 条。固定顺序、reference、随机噪声 seed 和 diffusion progress；报告 diff/stop/total loss。
- step 0 与每个保存点，固定 3 位 easy + 3 位 hard 患者生成试听。保存首次 seed=42 的完整 WAV；不做挑选重试，达到长度上限明确记录。
- 最多跑完 3 epochs；连续两次完整评估均比历史最佳 loss 高 15% 时停止。这个条件是 loss 监控停止规则，不等同于病理保留已得到验证。
- 保存所有这 6 个 LoRA 增量点，仅保留一份最新 optimizer；不复制基座。磁盘可用空间低于 20GiB 时停止继续写 checkpoint。

## 实现与验证

入口：`synthesis/voxcpm2_patient_sft/train_setting_d.py`，试听：同目录 `listen.py`。
模型与 packer 源码复用官方 checkout `f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69`；不是另写一个 VoxCPM 模型。

相对官方示例训练循环的必要调整：完整且固定的评估、不同 reference 配对、明确的 optimizer step/epoch、累计 loss 正确加权、reference-conditioned 试听、限定存储。
没有加入重试替代样本、静默跳过错误、自动改配置等 fallback。

验证记录：

- 单测：同患者非自身配对与每轮覆盖；官方 reference/target mask 与 stop label；不等长累积归一化；Setting D 数据与 merged-test 口径。
- 缓存：实际第一条音频的官方训练 encoder 输出与官方推理 encoder 输出进行数值一致性检查（rtol/atol=1e-5，记录实际最大误差）；AudioVAE 使用 FP32、关闭 cuDNN TF32；全部缓存保留哈希。
- 启动前 benchmark：最长目标 + 同患者最长不同 reference，连续三次真实 forward/backward/optimizer，检查有限 loss/grad 和峰值显存；其更新不用于正式训练。
- 正式训练从基座重新初始化 LoRA，日志中保留每 step loss、LR、grad norm、耗时、seen targets 与显存。

该实验仍未把新模型注册为 `speech generator`；旧 epoch 8 命名不变，等新 checkpoint 试听后再决定是否替换。

## 启动记录

训练进程于 2026-09-06 启动，当前 PID **3893374**。配置、runtime、`events.jsonl`、`status.json`、`train.log` 均在上述实验目录。
6 项单测通过（含 ASCII locale 下的粤语 JSON 读写）。2619 条 AudioVAE 特征缓存耗时 57.59s，占 118,034,688 bytes；启动时磁盘剩余约 36.1GiB。
第一次启动在保存 step 0 试听的中文 JSON 时因默认文件编码为 ASCII 停止，尚未执行任何 optimizer update。现已显式指定 UTF-8，并完整保留该次日志和 6 条 WAV 于 `startup_attempt_001/`；不是跳过失败样本，正式流程从零重新开始。
训练前 merged-test 511 条：diff loss **1.060155**，stop loss **0.208812**，total **1.268967**。这些是 VoxCPM2 训练目标，不是 CER，也不与 CosyVoice3 的 token loss 横比。

## 首个 checkpoint 的端到端验收

step 66（0.50095 epoch）已完成保存、重新加载、511 条完整评估与 6 位患者的固定试听；随后继续训练。

| 监控量 | Step 0 | Step 66 |
| --- | ---: | ---: |
| diffusion loss | 1.060155 | 1.055381 |
| stop loss | 0.208812 | 0.066574 |
| total loss | 1.268967 | 1.121955 |

本阶段下降主要来自 stop loss。检查 LoRA_B 后，base LM、residual LM、DiT 三部分均有非零更新；不存在只配置了模块但没有实际训练的情况。

6 条保存后的音频均可读取，帧数与 receipt 一致，没有达到 512-patch 长度上限。**但 vlink209 /「蠍子用強力鉗箍住呀嘛」出现异常拉长：45.28s，原声 6.53s（约 6.93 倍）**。这不是“完整 WAV 验收通过”就能掩盖的生成质量问题；保留首次 seed 输出，后续点继续观察，不重试替换。

首个 LoRA 权重约 70MiB，保存后磁盘剩约 35.94GiB。实际训练平均约 1.2s/step；完整评估约 23s，固定试听另需模型加载与生成时间。

第 1 个完整 epoch（step 132）已结束：diff **1.051478**、stop **0.059881**、total **1.111360**，511 条全部评估，当前最低监控 loss 位于 step 132。该点的固定试听正在生成，之后按配置进入第 2/3 epoch。已消耗新增存储约 419MiB，磁盘剩余约 **35.72GiB**。
