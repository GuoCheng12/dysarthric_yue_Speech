# 真实录音课程早期实验（执行前冻结）

## 问题与范围

在已经看过全部 2117 条真实录音的 M0 上，Agent 能否比随机复习和 CER 难例复习更有效地分配额外训练？不使用 TTS，不更换 split，不建立训练前 Gate，不更新 Skill。

三个分支：random、difficulty、agent。每轮共 96 条真实录音：共享 32 条按患者×原始难度比例抽样的背景复习，加各方法选择的 64 条；轮内不重复，每患者/每 content 最多 8 条（包含背景）。random 按原始难度比例随机选择，difficulty 按当前 checkpoint 的训练录音 CER 降序选择，agent 读取全部候选录音当前诊断并选择。相同上限避免单患者/单句占满练习。

每轮 1 遍，batch 8 × grad_acc 4 = 32，恰好 3 个 optimizer updates。LR=2e-5 constant，warmup=0，LoRA r16/alpha4/dropout.05，原有全部七种投影模块；修复后的 utterance-mean 梯度路径。每轮重置 optimizer，只继承本分支前一轮 adapter；固定最终 step3，不选最好 step。

诊断使用当前模型在真实 train 上的 decoded transcript、CER、字符插删替计数、相对 zero-shot 的逐位置退化/修复证据和覆盖。字词错误不能当作声学病理诊断。Agent 只收到匿名患者/录音 ID，不收到本地路径或真实患者标识。48 个原文去重后作为 content 表提供。

已有合并 511 条语音只作探索性反馈，不称为独立测试。主表报告 mean-utterance CER，另保留 pooled CER 和 patient-macro CER、easy/medium/hard、固定所选训练录音 dropout-off loss。

## 执行顺序与止损

1. 审计数据、M0 和训练器；补全 M0 的 2117 条 train decode。
2. 同一输入独立请求两次 gpt-5.6-sol/xhigh，验证 64 个 ID、引用、约束、理由；固定使用第一次处方训练，第二次仅检查 smoke 一致性，禁止择优。
3. 三组第一轮训练和真实反馈。若 Agent overall 优于 M0、距较好对照不超过 0.1 个百分点，且 easy/medium/hard 各自不比 M0 增加超过 0.2 个百分点，则继续，最多三轮。阈值仅为早停的工程规则，不代表统计显著性。
4. 第一轮不满足时，至多一次针对性复核：冻结三组课程，仅把共同训练 seed 从 42 改成 43，从同一 M0 各重跑一轮。之后停止，不重新选课、不加步数、不搜索超参。任何技术失败至多修复/复核一次；如已使用该机会，不再额外重复不理想实验。
5. 后续每轮先真实训练/评估，再由同一 Therapist 做 Reflection；下一轮同时接收历史处方、实际结果、Reflection 和新训练诊断。Skill 固定。若后来出现同样明显退化，停止，不硬跑。

早期继续投入的依据：Agent 在固定末轮同时优于 M0 和两种对照至少 0.1 个百分点，并且 easy/medium 的增加均不超过 0.2 个百分点、hard 不差于 M0。若只是三组相似、规则不输 Agent，结论就是尚没有 Agent 额外价值，不扩展 MAS。既报告原始数值，也报告是否通过这个预先约定的筛选。

## 资源与验收

单 L20 串行 GPU 工作，API 与无 GPU 的准备可以重叠。不复制基础模型/音频；只保存末步 adapter、manifest、预测、fit loss、实际 request/response 与时间/用量。最多 9 个主实验 adapter；若第一轮不理想则三组一次复核，共 6 个。开始约 31 GiB 可用，不清理历史文件。

结构化输出接口依据：<https://developers.openai.com/api/docs/guides/structured-outputs>；保持用户指定第三方网关的 gpt-5.6-sol/xhigh，SDK retries=0，store=false，失败不伪造处方、不自动降级模型。
