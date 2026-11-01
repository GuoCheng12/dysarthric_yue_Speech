# Qwen3-TTS：Setting D 完整 reference-conditioned LoRA

用户于 2026-09-06 确认：直接微调，每个保留 checkpoint 监控 MCD 和相似度。当前 CosyVoice epoch8 Speech Generator 不替换。

## 模型与数据

- 本地 `Qwen3-TTS-12Hz-1.7B-Base`，revision `fd4b254389122332181a7c3db7f27e918eec64e3`；不下载/复制第二份基座。
- 官方源码 `/data/qwen3-asr/third_party/Qwen3-TTS-probe`，commit `022e286b98fbec7e1e916cb940cdf532cd9f488e`。
- 运行环境 `/data/qwen3-asr/venvs/tts-model-probe`：Torch2.5.1，Transformers4.57.3，PEFT0.19.1；原 ASR 环境不改。
- 与旧 SG 和 Vox 实验相同的 **2108 条 / 57 患者 / 4.05978 小时** TTS train。不是 ASR 原始2117条全部用于TTS；9位只有一条录音的患者继续排除。
- 遵循用户的 train/test 协议：旧 dev258 + test253 = **511 条监控集**；没有重新引入独立 dev。它用于反复监控，不描述为 untouched final test。
- train 与监控集的 content ID、归一化文本、Jyutping、GT、音频 SHA256 交集为零。
- 每个 epoch 每条目标用一次，随机选同患者另一条 train 录音作 reference；监控 reference 使用旧 SG 固定 policy。
- 复用 Vox 实验预先固定的六条目标：3 easy / 3 hard，六患者五种唯一GT。不根据 Qwen 输出重新选人。

## 训练与推理必须一致

采用 Qwen Base **完整 ICL**：reference transcript、reference 的16组codec tokens、speaker embedding，加上目标 transcript。保持 `language=Auto`、`x_vector_only_mode=False`、`non_streaming_mode=False`，与现有 Qwen 原生克隆调用一致。这里的 non-streaming 参数控制文字/音频双轨排布，不表示向用户输出截断音频。

不直接运行官方单说话人 `sft_12hz.py`，也不把模型导出为固定 speaker 的 CustomVoice。官方该示例使用参考音频的 speaker embedding，但没有本实验所需的完整 reference codec/text 上下文。

本实现重用官方 `generate_icl_prompt`、Talker、Code Predictor 及冻结 speech tokenizer / speaker encoder。训练前拦截原生 `model.generate → talker.generate` 的实际输入，对比训练 prefix、trailing text、pad embedding；覆盖实际目标以及 reference 极短的边界情况，要求逐值完全相同。

监督位置明确：

- reference 最后一个位置的 hidden 预测目标的第一个 codebook0 token。
- 目标每帧预测下一帧；目标最后一帧的 hidden 预测 EOS。
- 对目标每帧，Code Predictor 的15个输出分别监督该帧的 codebooks1–15；不在 codebook 维度再做一次 causal label shift。
- reference、文本前缀和 padding 不计算监督 loss。

采用显式对齐后的 cross-entropy，避免把已经对齐的标签再次交给默认 causal LM loss 做隐式 shift。

## 首轮训练配置

| 参数 | 值 |
| --- | --- |
| LoRA 范围 | Talker 28层 + Code Predictor 5层的 attention q/k/v/o |
| rank / alpha / dropout | 16 / 32 / 0 |
| 冻结 | 非LoRA全部参数，含speaker encoder、codec、text projection、embedding及输出heads |
| LR | 2e-5，warmup20 steps，之后cosine |
| 优化器 | AdamW，weight decay .01，clip norm 1 |
| batch | micro4 × accumulation4 = effective16 |
| 训练上限 | 3 epochs，132 steps/epoch，共396 steps |
| 尾batch | 保留最后12条，不丢数据 |
| loss | codebook0+EOS CE + 0.3 × residual-code CE |
| 累积归一化 | 两项分别按整个effective batch实际有效token数归一化 |
| 精度 | BF16冻结基座，FP32 LoRA及Adam状态，BF16 autocast |
| 内存 | SDPA，non-reentrant gradient checkpointing |
| 保存和监控 | step0及66/132/198/264/330/396；每个保存点完整511条loss＋固定6条生成MCD/CAMPPlus |

MCD和相似度分别记录最佳step，不强行合并成一个加权分数。连续两次保存点同时满足 `MCD > 历史最低MCD × 1.05` 且 `CAMPPlus < 历史最高相似度 − 0.03` 时停止，记录为持续声学退化。这是本次预先固定的监控停止规则，不是临床定义的overfitting。没有自动换模型、换数据或重试替换生成结果。

## 生成与声学监控

- 保存后由独立进程重新载入原始Base＋该LoRA，检查加载参数集合完全匹配。
- 每条固定seed42，保留官方接口输出的完整FLOAT WAV。max_new_tokens768，记录触顶，不重试、不VAD裁切、不按CER过滤。
- MCD-DTW、CAMPPlus与上一轮 Vox/Cosy 对比完全共用 `examples/voxcpm2_sft_comparison.py::acoustics`，在原ASR环境＋隔离pysptk overlay的CPU进程计算。
- MCD：16kHz、WORLD CheapTrick、SPTK order24/alpha0.42、c1–c24排除c0、10ms、全局DTW；低能量阈值与前一轮一致。
- CAMPPlus：对照同GT的真实患者原声，不是输入reference。另报时长绝对相对误差，overall/easy/hard分开。
- 这次不新增ASR回译，不用CER选择TTS checkpoint。
- 每个点的六条评分完成后更新本地HTML，展示原声、固定reference、旧SG、Qwen原始版及所有保存点完整音频。未完成点不会显示为已测。

## 文件与存储

运行根目录：`/data/qwen3-asr/records/qwen3_tts_reference_sft_setting_d_v1`。

- `config.json / preparation.json / data.json / samples.json`：固定配置和数据。
- `features.safetensors / tokens.json`：冻结codec与speaker特征、文本token缓存；2619条cache18,631,536 bytes，127.09秒。
- `native_alignment.json / benchmark.json`：原生输入一致性、最长样本真实梯度与显存检查。
- `events.jsonl / status.json / train.log`：每step进展和错误。
- `checkpoints/step_*/lora_weights.safetensors`：只保存增量权重，不复制基座。
- `latest_optimizer.pt`：仅保留一个最新优化器状态。
- `listening/step_*/receipts.json / acoustics.json / metrics.json`：每点完整生成和客观分数。
- `listening/index.html`：持续更新的本地试听监控页。

存储低于20GiB保留线就停止继续保存，不自动删除任何已有模型、checkpoint或原始语音。

入口：`synthesis/qwen3_tts_patient_sft/train_setting_d.py`（prepare/cache/benchmark/train）、`monitor.py`。12项相关单测已通过，包含监督位置、codebook对齐、累积归一化、配对、尾batch、UTF-8以及共用声学指标的数值检验。

## 正式启动

原生推理输入逐值对齐已通过8个实际模型检查，prefix和trailing text最大绝对误差均为0。最长配对micro4的三次真实训练耗时1.10/0.74/0.92秒，峰值allocated7.08GiB、reserved8.19GiB；Talker和Code Predictor的LoRA_B均得到非零梯度。benchmark更新全部丢弃，正式LoRA从原始Base重新初始化。

可训练参数7,241,728。正式训练以独立会话PID3896848启动，启动信息写入`launch.json`；训练内同步调用每点生成和声学评分，不依赖聊天轮询才能推进。

初始同条件6条：MCD9.2538、CAMPPlus0.6487。vlink234首条seed42达到767帧/61.36秒上限，保留完整失败音频；另行用完全不注入LoRA的原始Base复核，也达到61.36秒，问题不是LoRA注入产生。该诊断未替换任何评测样本。

step66已保存，511条监控loss由5.4421降至4.0418；对应生成声学指标由流水线继续计算，不据此loss预判声音质量。

## 第1个epoch实测进展

step132完成全部2108条训练曝光，包含最后12条尾batch；随后进入第2个epoch。已完成点的全6条评分如下（均值，同GT真实目标作为对照）：

| 版本 | MCD-DTW ↓ | CAMPPlus ↑ | 时长绝对误差 ↓ | 触顶/6 |
| --- | ---: | ---: | ---: | ---: |
| 旧CosyVoice SG | 9.1887 | 0.7488 | 15.64% | — |
| Qwen step0 | 9.2538 | 0.6487 | 123.76% | 1 |
| Qwen step66 | 9.5994 | 0.6628 | 24.05% | 0 |
| Qwen step132 | 9.3404 | 0.6717 | 38.11% | 0 |

step132的511条监控loss为3.4320。hard患者相似度由0.5954升至0.6256，但仍低于旧SG的0.7376；hard MCD由12.0714变为12.8754。当前不能因loss下降便认定病理声音保留成功。继续执行预定3epoch上限和每点声学监控，未触发持续联合退化停止规则。

持续更新的本地页面：`http://127.0.0.1:8936/qwen3_tts_sft/index.html`，复用已有私有服务。初始24个全长播放器已通过HTTP长度、WAV时长和SHA256核验；评分更新会加入后续保存点。
