# M0 纯真实数据续训对照

## 结论

去掉全部 24 条合成语音后，同样的续训仍将 511 条 test 的 CER 从 **21.96% 推到 34.76%**；原增强组终点为 **32.51%**。

这说明本轮大幅退化不需要合成数据也能发生。优先应检查的是“已适配的 M0 再按当前更新方案继续训练”，不能直接把失败归因于 Therapist 处方或 TTS 质量。增强组在终点相对纯 real 续训好 **2.24 个百分点**，但仍明显差于 M0，因此也不能据此宣称 MAS 已改善 ASR。

## 对齐条件

- 相同起点：新 8×4 real-only M0 的 checkpoint-50。加载同一 adapter 权重，不恢复 Trainer 状态；两组都重置 optimizer/scheduler。
- 对照：2117 real；原干预：同样 2117 real + 24 synthetic。
- 相同训练配置：batch 8 × gradient accumulation 4；67 个新增 optimizer steps；lr=2e-4、linear schedule、warmup_ratio=0.02；seed=42；LoRA r=16、alpha=4、dropout=0.05，q/k/v/o/gate/up/down；相同 utterance-mean 梯度归一化。
- 实际保存的训练参数、软件版本、LoRA 模块和参数量核对一致；参数差异仅 train_file/output_dir。对照训练清单逐条等于原干预清单前 2117 条。
- Test：511 条、36 位患者、12 个 content；easy/medium/hard 为 265/145/101。按同一解码配置每 10 步及终点评测，不选择 best、不提前停止。
- 原 dev/test 文件仅作为两个解码分片，保持与已完成增强组相同的 batch 边界；合并逐条预测后重新计算 511 条指标，无独立 dev 判定。
- 数据量不同会改变随机 shuffle 后的具体 batch；相同 seed 不意味着逐步训练样本完全相同。这是当前两种训练数据方案的单种子对照，不是单条合成语音的独立效应估计。

## 完整轨迹

以下为 **mean utterance CER（%）**，并非 pooled CER。step 是从 M0 开始的新增更新步数。

| 新增 step | Real-only 续训 | Real + 24 synthetic | 增强组减对照（百分点） |
|---|---:|---:|---:|
| 0（M0） | 21.96 | 21.96 | 0.00 |
| 10 | 23.34 | 24.07 | +0.73 |
| 20 | 25.42 | 25.30 | −0.12 |
| 30 | 27.64 | 27.31 | −0.33 |
| 40 | 28.10 | 28.88 | +0.78 |
| 50 | 32.72 | 31.55 | −1.17 |
| 60 | 34.71 | 32.19 | −2.52 |
| 67 | 34.76 | 32.51 | −2.24 |

| 模型 | Overall | Easy | Medium | Hard | Patient macro |
|---|---:|---:|---:|---:|---:|
| M0 | 21.96 | 7.40 | 24.76 | 56.16 | 19.12 |
| Real-only 续训 +67 | 34.76 | 18.80 | 39.10 | 70.38 | 30.86 |
| Real + 24 synthetic +67 | 32.51 | 16.84 | 36.33 | 68.15 | 28.55 |

纯 real 续训终点相对 M0 退化 **12.79 个百分点**。三个难度组均退化，并非仅 easy/medium 的权衡。增强组相对该对照的三个难度组终点都更好，但仍全部差于 M0。

按每位患者的平均句子 CER 配对：纯 real 续训相对 M0，36 位患者全部变差；增强组相对纯 real 续训，25 位改善、1 位持平、10 位变差。

## Train loss 与运行核验

对照完成 67 步、epoch=1.0，Trainer 耗时约 **21.6 分钟**，包含七次完整解码评测。

- 5 步窗口 loss：step 5 为 0.2809，step 10 为 0.2314，step 30 为 0.1425，step 50 为 0.1389，step 65 为 0.1332。
- 全程 train_loss=0.167414，是全程平均，不是终点单步 loss。
- 记录的梯度范数约 0.24–0.39，没有 NaN 或梯度爆炸迹象。
- 训练目标继续拟合，同时真实未见 content 的识别退化，符合过拟合/有害继续适配的表现。此实验没有单独拆解学习率、调度重启、optimizer 重置和步数的作用。
- 七个 checkpoint 各 511 条预测均核验唯一性、GT、患者、content、难度标签，并从逐条结果重新计算指标，与汇总完全一致。
- 原 M0、训练源文件、最终 checkpoint 哈希核验通过；未替换 M0、未启动后续 MAS 轮次。

## 与旧静态增强实验的关系

此前 Random-480 / Full-1440 从原始 Qwen3-ASR 初始化新的 LoRA；本轮是从已经适配的 M0 出发，再重置调度器进行 67 步更新。两者峰值 lr 都是 2e-4，但模型起点、调度长度、数据比例与内容覆盖不同，不能把旧实验的稳定表现直接外推到当前续训方案。

下一步建议先重新确定 M0 上的单轮更新强度（更少更新步数或更低续训学习率），再判断处方的增量价值。当前只完成本对照，不自动改配置或开始新实验。

## 可追溯记录

运行根目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/m0_real_replay_control_b8g4_v1`。

- `run_config.json`：完整配置、父模型和源文件哈希。
- `run_summary.json`、`comparison.csv`：七个评测点的指标与对照。
- `evaluation/merged_test_step_*/predictions.jsonl`：逐条预测。
- `training/checkpoint-67`：终点；目录 SHA-256 为 `b4fcbc0076442330ad372973d3ac63b3f75d5921656ba205ccb0577e707c14aa`。
- 训练入口：`finetune/scripts/train_m0_real_replay_control.py`。
- 增强组原记录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/therapist_m0_b8g4_pilot_v1/merged_test_reassessment_v1`。
