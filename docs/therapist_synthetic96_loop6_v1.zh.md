# 六轮纯 synthetic 连续干预

运行目录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/therapist_synthetic96_loop6_v1`。

- 从 real-only M0 checkpoint-50 重新开始，不接之前 probe 第三轮。
- 每轮实际调用 Therapist（gpt-5.6-sol / xhigh），提出六个新文本；每个文本由同一组 16 位患者 reference 条件合成，共 96 条。
- 使用 Setting D epoch 8 Speech Generator，隐式粤语；reference 来自冻结的训练池。患者按 easy/medium/hard 分层选取，额外名额逐轮轮换。
- ASR 每轮仅使用当轮 96 条 synthetic，零 real replay，不累积旧 synthetic。
- 每轮两遍：microbatch 8 × grad_acc 4，6 optimizer steps；六轮累计新增 36 steps。lr 2e-5 constant，warmup 0；LoRA r16、alpha4、dropout0.05，七类 projection modules。
- 每轮承接上一轮终点 adapter 权重，优化器和 scheduler 重新初始化，与先前 probe 一致。
- 只保存 step 6 终点模型，不存 optimizer；不根据 CER 换 checkpoint，不因 CER 退化提前终止。技术错误写入 failure.json 并停止，不自动重试或伪造输出。
- 每轮记录 dropout-off synthetic 训练 loss（step 0 / 6），并在合并后的 511 条真实语音上 decoded CER 评测，再调用 Therapist Reflection，将真实结果和建议送到下一轮。
- 此处真实语音是 feedback，不是独立密封 test。Skill v000 固定，不运行 Skill Optimizer。

## 准备与状态

启动前原有测试 220 项通过；新增零 replay protocol、ActionContract、TrainingReceipt 正反例测试 2 项通过。
零 replay 显式允许为 0，负数仍拒绝。提示词同步说明零 real 梯度输入，并明确至少一条 target，避免 Reflection 建议无法执行的零 target 处方。

启动 PID 3854265。run.log 显示第 1 轮已完成 assess、进入 prescribe；此记录不代表训练已完成。
每轮完成会写 completed_round_NNN.json，全部完成写 run_summary.json；错误写 failure.json。
入口：`examples/therapist_synthetic96_loop6.py`。密钥仅进入进程环境，不写入实验文件。
