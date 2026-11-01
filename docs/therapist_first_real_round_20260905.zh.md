# 新 M0 的第一轮真实 Therapist 干预

> 评测口径已更正：用户确认当前为 2117 train / 511 merged test。本页的 dev 数字仅保留为历史执行记录；当前结果以 [合并 test 更正报告](therapist_first_round_merged_test_20260905.zh.md) 为准。

运行目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/therapist_m0_b8g4_pilot_v1`。

## 固定设置

- Setting D：2117 real train / 258 dev / 253 sealed test。
- 父模型：`real_only_m0_b8g4_utterance_mean_v1/training/checkpoint-50`，不是旧 1×32 M0。
- 新 M0 的真实 dev 输出重建 Evidence Pack；不复用旧 smoke 处方或反事实结果。
- Therapist：`gpt-5.6-sol`，xhigh；6 个独特句子，共 24 条语音。
- Speech Generator：Setting D epoch 8；固定 reference 分配，隐式粤语。
- 训练：2117 real replay + 本轮 24 synthetic，8×4；LoRA r16 / alpha4 / dropout0.05。
- 父 adapter 仅载入权重；optimizer / scheduler 重置。lr=2e-4，linear，warmup=0.02，seed42。
- 一次 manifest pass，共 **67 个新增 optimizer steps**；采用修复后的 utterance-mean 梯度路径和右侧 padding。
- 每 5 步记录训练日志；每 10 步及最后第 67 步保存、解码 dev。中间结果只用于监督，不选择中间 checkpoint 替换终点。
- 本轮不评测 test，不设训练前 Gate，不作回滚；真实终点结果进入 Reflection 和 ledger。
- 只运行首轮小规模闭环；不启动草案中的 4×480 大实验。

## 执行入口

`examples/therapist_first_real_round.py` 的顺序为：

1. `prepare`：冻结新协议、构建真实 Evidence Pack、保存本轮 prompts 和 Skill 快照。
2. `prescribe`：一次在线处方调用；验证后冻结 EvaluationPlan 并分配合成任务。
3. `synthesize`：运行维护中的 TTS 入口；用真实输出生成 receipt 和训练清单。
4. `finetune/scripts/train_therapist_round.py RUN_DIR`：固定预算训练与 dev 监督。
5. `reassess`：验证实际训练步数与 checkpoint，编译 RoundResult 和 ReflectionPack。
6. `reflect`：一次在线反思调用，验证后记录真实 intervention ledger。

所有阶段失败直接停止，不自动重试或替换数据。密钥只通过无回显输入进入当前进程，不写入记录。

## 解释边界

24 条合成语音约占训练条数 1.12%。相对 M0 的变化是 real replay、合成数据与继续优化的共同结果；这一轮没有 real-replay-only 对照，不能将变化单独归因于合成处方。

真实首轮已完成并入账。事件链以 `committed` 结束，不是需要最终 test 授权的全局 `completed` 状态。验收以运行目录中的训练记录、dev 输出、Reflection 与 ledger 为准。

## 已完成的真实训练与 dev 结果

完成 24/24 条语音，覆盖 20 位患者；reference 分组 easy/medium/hard 为 10/8/6。合成总时长 184.56 秒。逐条验证完整 WAV 帧数、有限数值、GT 对应关系；2117 条 real replay 的 audio/text/prompt 与原清单一致。

ASR 实际执行 67 个新增更新步，trainer state 的 epoch=1.0。训练及 7 次完整 dev 解码共 726.95 秒。进程正常退出，GPU 已释放。下面 CER 为 **mean utterance CER（%）**；正式 Reflection 另明确使用 pooled CER 等指标，不混用两种口径。

| 新增 step | Overall | Easy | Medium | Hard |
|---|---:|---:|---:|---:|
| 0（M0） | 19.34 | 6.61 | 19.10 | 51.26 |
| 10 | 20.10 | 6.96 | 19.80 | 53.10 |
| 20 | 22.74 | 8.06 | 24.89 | 56.39 |
| 30 | 24.73 | 10.26 | 26.77 | 58.00 |
| 40 | 26.29 | 13.06 | 26.91 | 58.33 |
| 50 | 28.39 | 14.62 | 29.20 | 61.53 |
| 60 | 29.25 | 15.72 | 29.49 | 62.52 |
| 67（固定终点） | 29.38 | 16.65 | 28.86 | 61.65 |

终点 overall pooled CER 为 28.86%（M0 为 19.01%）；patient-macro CER 为 27.69%（M0 为 17.53%）。36 位患者中 1 位改善、1 位持平、34 位退化。

### 处方目标的真实响应

主目标是声调 1→6 的确认字符替换归因，不是声学声调测量。

- 1→6 签名次数：19→17。但原 19 个签名错误中仅 2 个变正确，6 个转为其他替换、3 个变为 unresolved、8 个仍是原签名；另有新进入该签名的错误。因此不能将净减少 2 次等同于恢复能力。
- 1→3 对照签名：9→15。
- 6→4 保护签名：11→14。
- 4 个受监控的能力锚点均出现正确位置流失；其中包含转为 unresolved 的位置，不把它们全部说成已确认声学错误。
- 9 项预先冻结的 expected outcomes 全部 `not_met`，包括主目标的配对恢复条件。

### 当前结论

此次干预没有改善 ASR。训练继续拟合而 dev 持续退化，与过拟合/能力退化现象一致。它不证明 Therapist 方法无效，也不支持把本轮算作有效处方。

需要优先复核的是 **M0 后续更新强度**：本轮只加入约 1.12% 的合成条目，却以重置的 2e-4 学习率继续执行整整一遍 real replay。学习率/轮长过强是待验证解释，不是已经通过独立对照确认的原因。本轮未自动改变训练合同，也未启动下一轮。

### 验证记录

- 98 项相关单元测试通过；新增入口 Ruff 检查通过。
- 监督回调的 67 步保存调度通过单独检查，不触发 best-step selection 或提前终止。
- 训练源码 SHA 在执行前后保持一致；父模型、清单、生成语音和终点 checkpoint 均与事件链 receipt 绑定。
- 本轮仅评测 dev；未执行 sealed test 评测。

## Reflection 与首轮验收

在线 Therapist Reflection 一次调用完成，验证通过，结论为 `hypothesis_verdict=refuted`。处方调用耗时 375.56 秒，Reflection 调用耗时 293.95 秒；均为 gpt-5.6-sol / xhigh，无自动重试。

Agent 的定性建议是重新评估唯一主要目标、减少原来的 primary/contrast 副本，并在固定预算内增加合资格 protection 和 maintenance。它明确区分：新快照仍正确的位置才能称作 maintenance，已流失的位置必须由新 Evidence Pack 判断是否可成为回归修复目标。

这仅是下一轮建议，不是已经执行的新处方。Agent 没有改变学习率、训练步数或固定总预算。人类研究者需另行讨论更新强度问题，不能指望 Agent 通过改句子自动修复不合适的训练合同。

验收状态：

- 完整真实链路：`assessed → proposed → materialized → synthesized → trained → reassessed → reflected → committed`。
- 操作与可追溯性验收通过；ASR 改善目标未达到。
- 主目标签名减少但未通过配对恢复条件，已正确避免误报成功。
- Reflection 与真实 ledger 均已保存，Skill 保持 v000，未执行低频 Skill Optimizer。
- 原 M0 不变；本轮固定终点独立保存。无下一轮后台训练。
