# 已接受处方的真实 TTS smoke

## 结果

2026-09-05：将 `prescription_prompt_audit_smoke_v1` 中用户接受的 6 句处方，按原 reference 分配和 seed，交给 epoch 8 Speech Generator。

- 完成 24/24 条，每句 4 条；覆盖 20 位患者。
- Reference 难度分组：easy 10、medium 8、hard 6 条。
- 合成总时长 151.48 秒；成功执行耗时 232.42 秒（含加载）。
- 每条仅调用一次推理，没有替换 reference、改写句子、补生成或剪辑。
- WAV 哈希、完整帧数读取、采样率及有限数值检查通过；这不等于人工核实了读音和内容。
- 全项目测试 197 passed，相关文件 Ruff 检查通过。

输出目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/prescription_synthesis_smoke_v1/`。

- `synthesis_plan.json`：来源及哈希。
- `synthesis_manifest.jsonl`：24 条真实推理输入。
- `synthesis_summary_attempt2.json`：成功运行、环境和逐条 WAV 记录。
- `audio_check.json`：独立文件检查结果。
- `index.html`：全部目标语音和 reference 原样内嵌的独立试听页。

## 本次发现并修复的执行问题

第一次在加载模型前退出，未生成任何音频。`source_tree_sha256` 拒绝了 CosyVoice 原仓库自带的文件符号链接。

诊断发现，按链接实际内容计算得到的源码哈希恰好等于既有锁定值 `2396a65ff94698e85c5624c8c5bc46bfa199e8542d2706c25e58b8281269f46e`。因此删除错误的“一律拒绝符号链接”判断，保留原哈希校验，不修改任何锁定值或第三方源码；增加回归测试，确认链接内容计入哈希，目标文件改变会改变哈希。

初次失败保留在 `synthesis_summary.failure.json`。修复后的运行另写成功记录，不覆盖失败历史。

## 与 ASR 闭环的关系

这次接通的是：已接受处方 → 真实患者条件 TTS → 技术检查 → 试听。

没有进行 ASR 训练、dev 复评或真实 Reflection。源处方来自接口 smoke，仍保持原 artifact mode，不将其改标成真实训练历史。正式四轮协议仍为 draft。

后续真实干预应从 Real-only M0 checkpoint-50 开始，使用 real replay 和合成数据更新 ASR，再将真实 dev 评测结果送入 Therapist 的 Reflection 调用。现有 `round_training.py` 只构建训练命令，不自动执行。

本次新增的 `examples/therapist_synthesis_smoke.py` 准备输入；`examples/therapist_synthesis_listening.py` 验证成功输出并生成试听页。模型推理仍使用唯一维护入口 `synthesis/cosyvoice3_patient_sft/synthesize_manifest.py`。
