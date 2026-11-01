# 首轮 MAS：按 train / merged test 口径更正评测

用户确认当前只有 train / test，原 dev 并入 test。此前报告误用了三分法；本报告取代此前 dev-only 表格作为当前 checkpoint 对比口径。

## 数据与模型

- Train：2117 条 real，48 个 content，未改变。
- Test：511 条 = 原 dev 258 + 原 test 253；36 位患者、12 个 content。
- Test 难度：easy 265、medium 145、hard 101。
- 清单：`/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl`。
- 清单 SHA-256：`189d749e7e65f14a0440d7f6707a0e1dcfa1a5a62c6c0e5f362b08c1c53e2f98`。
- M0 仍为新 8×4 real-only 训练的 checkpoint-50。本次不重训、不重新选择 M0。
- 干预仍是从该 M0 开始的 2117 real + 24 synthetic、67 个新增更新步；本次没有新增训练。

## 更正后的完整结果

以下均为 **511 条合并 test 的 mean utterance CER（%）**，step 是从 M0 开始的本轮新增步数。

| 模型 / 新增 step | Overall | Easy | Medium | Hard |
|---|---:|---:|---:|---:|
| Zero-shot | 25.70 | 6.41 | 28.78 | 71.89 |
| M0（本轮起点） | 21.96 | 7.40 | 24.76 | 56.16 |
| +10 | 24.07 | 9.06 | 25.60 | 61.27 |
| +20 | 25.30 | 9.42 | 28.88 | 61.79 |
| +30 | 27.31 | 11.20 | 30.80 | 64.58 |
| +40 | 28.88 | 13.49 | 32.04 | 64.73 |
| +50 | 31.55 | 15.59 | 35.24 | 68.10 |
| +60 | 32.19 | 16.16 | 37.28 | 66.92 |
| +67（固定终点） | 32.51 | 16.84 | 36.33 | 68.15 |

终点相对 M0 的 overall CER 退化 **10.55 个百分点**。已评测的所有续训 checkpoint 均差于 M0；这句话不涵盖未保存/未评测的第 1～9 步等位置。

Patient-macro CER：zero-shot **22.63%**，M0 **19.12%**，本轮终点 **28.55%**。终点相对 M0，36 位患者全部退化，而不是此前原 dev 子集上的 34 位退化。

## 目标错误的合并集配对检查

- 声调 1→6 签名：**24→21**。原 24 个错误中，3 个变正确、6 个转为其他替换、4 个变为 unresolved，11 个仍是同一签名。净次数下降仍不能证明目标恢复；配对恢复判定为 inconclusive。
- 声调 1→3 对照签名：**24→28**。
- 声调 6→4 保护签名：**20→23**。

这些是文字输出归因的确认签名，不是声学病理测量。原先的“19→17、只有 2 个恢复”等数值只描述原 dev 子集，不用于合并集结论。

## 计算与核验

复用 zero-shot、M0 和各 checkpoint 已有的原 dev 逐条预测，以及 M0 已有的原 test 预测；为七个干预 checkpoint 补跑缺失的原 test 253 条。解码使用同一维护入口、Cantonese、batch=8、max_new_tokens=256。

通过 utt_id 验证 511 条不重复且恰好覆盖两部分；逐条核验 GT、患者、content 与难度标签。统一重新归一化并计算 CER，不直接平均旧 split 百分比。Patient-macro CER 从 511 条重新聚合患者后计算。

预测源文件及 checkpoint 哈希保存在各模型的 summary.json。终点配对结果与模型汇总一致，并通过序列化往返、分组样本数和加权均值检查。中途汇总程序的 dataclass 序列化错误已修正；未修改预测，也未重复生成已完成的终点预测。

完整数据：

- `/data/qwen3-asr/records/therapist_mas_setting_d_v1/therapist_m0_b8g4_pilot_v1/merged_test_reassessment_v1/comparison.csv`
- 同目录 `results.json`：所有指标及来源哈希。
- 同目录 `paired_m0_vs_step67.json`：511 条上的配对错误变化。
- 同目录每个模型的 `predictions.jsonl`：完整合并集逐条预测。

## 本次更正的边界

这次纠正的是评测范围与报告，不是重跑历史决策。M0 原本按旧 dev 选择，第一轮处方及 Reflection 也实际只读取了 258 条证据；原始冻结记录保留，不伪装成基于 511 条执行过的闭环。

下一轮开始前，需将 Harness 协议、Evidence Pack 和反馈输入一并改为用户确认的 511 条口径。当前没有启动新一轮训练或新的 Agent 调用。
