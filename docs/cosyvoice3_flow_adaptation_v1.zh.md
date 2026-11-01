# SG Flow-only 患者适配：单配置实验

运行目录：`/data/qwen3-asr/records/cosyvoice3_flow_sft_setting_d_v1`。

## 问题与固定条件

当前 SG 只训练了 LLM LoRA；这里固定 epoch8 LLM、speech tokenizer 和 HiFT/vocoder，
只适配 Flow 全部参数。检验冻结 tokenizer 输出中的患者信息能否更好地还原成声学特征，
不把“合成语音更容易被 ASR 识别”当作成功。

与 SG 完全相同的 2108 条训练录音、57 位患者；merged511 仅用于既有反馈。
train 和反馈的音频哈希、GT、content、Jyutping 身份不重叠。
24 条声学监控样本覆盖 24 位患者，按原始 utterance 难度 easy/medium/hard 各8条，
不按任何生成结果挑选。Reference 为既有同患者训练录音；训练 reference 每 epoch
同患者随机选不同录音，target 全量遍历、不裁切、不替换失败结果。

## 训练

- 一个配置，3 epochs，264 optimizer steps/epoch，共792步。
- Adam，LR 1e-5 constant，weight decay 0，clip norm 5。
- microbatch 2，effective batch 8；最后不足8条的真实样本保留。
- FP32 参数/optimizer，BF16 autocast，DiT 非重入 gradient checkpointing。
- token embedding、lookahead、speaker projection、DiT 都属于 Flow 并更新；其他模块不更新。
- 复用官方 CFM 的线性 noise/target 插值和 velocity MSE、0.2 classifier-free dropout。
- 与官方默认 forward 的差别：固定非流式；reference 为独立同患者录音，而非 target 自身前缀；
  拼接 reference token/mel 与 target token/mel，只有 target 有 loss，reference 仅条件。
- 梯度归一化按整个 effective batch 的有效 target mel 元素总数计算，不逐 microbatch 均值累加。
- epoch0 与每个 epoch 终点均评测，原 optimizer 在各 epoch 间保持，不重启。
- 每 epoch 仅存 Flow 权重，不复制其他模型、不存 optimizer。可用存储保持20GiB以上。

已存在同一冻结 tokenizer/CAMPPlus 的 train/dev/test cache；对24条选定录音用官方
frontend 重新提取，核验 token 完全一致、embedding 最大误差小于1e-5 后复用，记录原文件哈希。
mel 重新提取。训练 token/mel 按官方2倍时间率取共同完整 token 覆盖，所有对齐长度有记录；
评测音频始终保留原完整输出，不将训练端帧对齐当作试听裁剪。

## 评测与决定

每保存点生成24条真实 token 重建与24条文本生成。文本生成在 epoch0 使用原 SG 原生接口，
记录实际送进 Flow 的 token 序列；以后完全复用该序列。这样对比只改变 Flow。
reference、GT、seed42、原生10步采样/CFG 均保持不变。

MCD-DTW、CAMPPlus、时长误差、静音比例和 F0 中位数差复用此前对比工具，按原声逐句配对。
选择三个训练点中重建 MCD 最低者；差不足0.01dB优先较早点，epoch0仍保留为对照。
若所选点的重建和文本生成 MCD 都低于 epoch0，且两种模式的 CAMPPlus 均下降不超过0.01，
进入已有576条真实增量对照的 ASR 检验；这是缩减后续计算的声学进展规则，不是临床指标，
不宣称由此能预测 ASR 收益。逐患者/难度结果同样全部报告。

条件满足后只新增“R1541 + 新Flow SG合成Q576”一组，沿用此前三训练seed、固定step50，
对照旧SG、real replay、真实Q和原始CosyVoice已有结果；不替换原SG、不修改原数据。
否则结束该单配置实验，说明哪个环节未改善，不自动扩大搜索。

## 启动前检查

单元测试覆盖 reference/target/padding mask、同患者不同录音配对、梯度累积归一化、
CFM velocity公式、checkpoint选择规则。真实模型 smoke 对比自定义训练条件与原生
Flow.inference 的 mu/speaker/cond，要求逐元素最大误差为0；验证非零有限梯度、实际参数更新。
Smoke 的更新丢弃，正式训练重新加载原 Flow。

入口：`synthesis/cosyvoice3_patient_sft/flow_adaptation.py`。
评测与完整试听：`examples/flow_adaptation_evaluate.py`。
