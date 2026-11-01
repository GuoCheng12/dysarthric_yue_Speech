# 三轮患者交叉合成 + 训练剂量实验

状态：三轮两臂训练、24份真实评测和3次实际Reflection全部完成并committed，主进程正常退出；执行与数据验收通过，另记录一项后续动作契约冲突。运行目录：
`/data/qwen3-asr/records/therapist_mas_setting_d_v1/therapist_crossed96_dose64_loop3_v1`。

完整剂量曲线、loss、Reflection和验收见[监督报告](/data/qwen3-asr/records/therapist_mas_setting_d_v1/therapist_crossed96_dose64_loop3_v1/supervision.md)。以下均为同一合并511条真实反馈集上的逐句平均CER（%）：

| 固定终点 | Real-only shadow | MAS | MAS easy | MAS medium | MAS hard |
|---|---:|---:|---:|---:|---:|
| 原M0 | 21.96 | 21.96 | 7.40 | 24.76 | 56.16 |
| 第1轮step64 | 23.54 | 22.53 | 7.66 | 25.15 | 57.77 |
| 第2轮step64 | 26.21 | 23.92 | 9.19 | 26.33 | 59.11 |
| 第3轮step64 | 27.42 | 25.09 | 10.06 | 27.87 | 60.53 |

MAS每轮最佳观测点分别为step8/16/16（21.65/21.75/22.97%），不是下一轮实际起点。MAS相对纯real续训减轻累计退化，但固定64-step闭环未超过M0。三轮synthetic loss均持续降低，终点约0.11–0.12；不支持继续以“没训够”为由增加重复遍数。优先讨论较小的固定每轮剂量，尚未运行新的小剂量轨迹，不拼接中途checkpoint充当结果。

验收：288个WAV、18个文本、28.2567分钟；引用均来自train，每句覆盖同一组16人；real跨轮无重复（MAS480/对照768），每个batch32严格20real+12synthetic。24份汇总重新计算共12264条预测完全一致；六个分支均64步、每臂累计192步，父权重与源文件hash通过；最终RunStore round3 committed。Harness测试189项通过。

第二轮曾因历史ID重复前缀停止。经用户授权修复后复用了未经修改的原始Agent response；旧源码和原始记录保留在revisions/citation_v2，修复版独立溯源。第三轮实际新调用验证了修正后的引用接口。

仍需处理的后续问题：最终Reflection建议暂停主目标（0 target），但Proposal目前要求至少一个target句子。最低角色约束未完整暴露在NextActionContract中；没有执行第四轮，也未自动改变方法或伪造修正输出。三轮Reflection终点结论均为refuted，Skill v000全程固定。

## 执行设置

- 两条独立续训轨迹，都从原 Real-only M0 checkpoint-50 开始；不从上一轮旧实验的 M2 开始。
- MAS 每轮 160 real + 96 synthetic；real-only shadow 每轮 256 real，其中 160 条与 MAS 相同。
- 每轮 Therapist 生成 6 个新句子，同一组 16 位患者分别读完 6 句。使用 epoch-8 Speech Generator；隐式粤语，无新增语言 prior。
- reference 从既有 57 人池中按 easy/medium/hard 分层确定性随机抽取 6/5/5 人；多一人的分层随轮次轮换。该分层不等同于 ASR 样本难度标签。
- 全部真实训练池仍为 Setting D 的 2117 条、66 位患者、48 个 content；不是 48 位患者。
- 每个有效 batch：20 real + 12 synthetic；micro-batch 8，grad-acc 4；lr 2e-5 constant，warmup 0。
- 保留已修复的 utterance-mean 累积归一化和原 LoRA 设置。每轮载入父 adapter 权重，重建 optimizer/scheduler；轮内连续训练。
- 每轮固定 256 条训练记录，观察新增 step 8/16/32/64，即 1/2/4/8 遍；不为每个观察点重新初始化。
- 默认 step64 为下一轮起点；不选择中途最好的 checkpoint 替代它。
- real 样本轮间不重复；句子轮间更换；旧 synthetic 不累计；reference 组轮间重新抽取，允许患者重现。
- 每轮真实评测后，由同一个 Therapist 做实际 Reflection，再产生下一轮处方。Skill v000 固定，没有伪造 Skill Optimizer 更新。

## 监督与判读

每轮 step0 和 8/16/32/64，对相同训练记录测量 dropout-off 的 real/synthetic 分项 teacher-forced mean utterance loss。沿用训练代码的 loss，测量恢复模型模式和随机状态，不影响后续训练。

每个观察点解码同一组 511 条合并反馈语音，报告 overall/easy/medium/hard、pooled 和 patient-macro CER。该集合同时用于反馈和报告，不称为独立封存 test。

连续两次 overall CER 比本轮父模型差至少 5 个百分点，同时固定训练集 loss 下降，停止实验并记录停止点，不伪装完成三轮，也不自动换较早 checkpoint。

训练 fit loss 降低只能证明更好拟合当前标签；真实 CER 改善才是迁移信号。跨轮数据不同，train fit loss 不直接横比。与先前 232+24、8-step 实验相比，同时改变合成规模、混合比例和训练剂量，因此不能把全部差异仅归因于增加训练步数。

## 验收目标

1. 每轮 6 文本 × 同一组 16 患者 = 96 个完整非空 WAV；reference/模型/音频 hash 与账本一致。
2. 每轮两条 256 行 manifest，共享 160 real；每个 batch32 正好 20/12；真实训练与反馈 content 不重叠。
3. 三轮完成时，每条轨迹新增 192 个 optimizer steps；保存全部 8/16/32/64 观察点和固定样本 loss。
4. 处方、实际训练结果、Reflection 与下一轮 EvidencePack 连贯；真实失败立即暴露，不用 fallback 补齐。
