# Setting E：已取得收益的 MAS 采样方法与决策总结

保存版本：`real_sampling_mas_seed42_590_v1`，2026-09-11。

这轮得到的正向结果是：**在相同 590 次参数更新、相同 18,880 次真实录音曝光下，由 MAS 根据 dev 反馈分配训练数据，test CER 比普通随机遍历更低。** 值得保留的是“观察错误—分配已有练习—执行训练—检查效果—修正下一阶段”这套方法和本轮完整轨迹。

本文总结 Agent 已保存的结构化决策、简短解释、反思与经验，并核对实际曝光和评估结果。它是可核对的决策摘要。

## 取得了什么收益

| 相同 step590 | 普通采样 | MAS 采样 | MAS 相对变化 |
|---|---:|---:|---:|
| test 总 CER | 15.7368% | **15.4813%** | **−0.2555 个百分点** |
| test 患者等权 CER | 13.5620% | **12.6193%** | **−0.9427 个百分点** |
| dev 总 CER | 17.4455% | 17.4653% | +0.0198 个百分点 |
| dev 患者等权 CER | 18.2491% | 18.5164% | +0.2673 个百分点 |

test 共 372 条录音、4,696 个目标字符；错误总数从 739 减少到 727，相对减少 1.62%。step319 至 step590 的连续 11 个评估节点，MAS 的两项 test CER 均低于普通采样。按各自 dev 患者等权 CER 选择 checkpoint，两组都选中 step580，其 test 总 CER 为 **15.6942% → 15.3961%**。

![相同训练节点的 test CER 与 MAS 分组概率轨迹](/data/qwen3-asr/records/therapist_mas_setting_e_v1/preserved_methods/real_sampling_mas_seed42_590_v1/analysis/method_trajectory.png)

这支持保留本轮方法作为后续比较的正向版本。当前结果来自 seed42；总体比较尚未分离动态调整、组别重加权和有放回抽样各自的贡献。患者等权收益对小样本患者较敏感，详表保留在原结果报告。这里确认的是相同更新预算下的收益；计入 Agent 等待与恢复开销，普通分支约 42.05 分钟、MAS 约 201.46 分钟，尚未取得相同墙钟时间优势。

## Agent 实际怎样调整

**1. 初期先把练习分给错误较多、又有相关训练材料的部分，同时保留基础能力。**

第 1 轮看到 dev 困难长句 CER 75.37%、困难短句 69.20%；中等长句也有 321 个错误，覆盖 7 位 dev 患者。它检查目标句在训练池中的分布后，将计划概率分为 easy 19%、medium 38%、hard 43%，长句合计 60%。实际 928 次曝光基本落实这一分配，随后 dev 总 CER 从 25.70% 降至 22.30%。反思同时记录 easy_short 退步，以及改善主要来自两位患者；没有把第一轮整体下降当作所有问题都解决了。

**2. 遇到反弹后，修改对训练价值的判断。**

第 2 轮将 easy 总比例从 19% 提高到 25%，但 dev CER 反弹到 24.00%；三个长句组退步、三个短句组改善。它在下一轮明确记下：提高 easy 并不必然保护能力；hard_long 曝光从 218 增至 237 次，效果却转差，曝光更多与收益并不呈简单单调关系。

因此，困难长句的计划概率从第 1 轮 24%，逐步降到第 4 轮 17%、第 7 轮 13%；easy 合计则逐步回到 32%。第 4 轮之后，六组 dev 总 CER 同时下降，7 位 dev 患者改善。这体现了“根据变化判断哪里值得继续练”，而不只按当前最高 CER 排序。

**3. 对小幅调整做后续核对，并允许撤回。**

第 8 轮把 hard_short 的 1 个百分点移给 easy_short。随后 hard_short 两项 dev CER 都变差；第 9 轮便撤回这一转移，恢复 hard_short 19%、easy_short 16%。下一次 hard_short CER 从 45.98% 降至 43.45%，dev 总 CER 从 18.89% 降至 18.59%。这是有完整记录的“尝试—看结果—有限撤回”实例；它没有把这一阶段的相关变化写成已经证明的单组因果。

**4. 中后期保留已经形成的分配，不要求每轮必须改动。**

第 13 至 16 轮连续使用同一组概率：easy_long/easy_short 各 16%，hard_long 13%、hard_short 19%，medium_long 21%、medium_short 15%。理由是局部句子、患者和组别的方向反复，继续观察比跟随每次波动来回改更合适。这四阶段 dev 总 CER 从 18.53% 降至 17.47%。

之后主要是 1–2 个百分点的转移或撤回。第 20、21 轮采用 easy 合计 32%、medium 36%、hard 32%。最后一轮 dev 完全持平，反思如实记为停滞，并指出替换减少被漏字增加抵消。经验记录保留了成功、失败与矛盾，供下一次使用。

**5. 语音学知识让诊断更具体，并帮助避免重复解释同一个错误。**

Agent 实际检查了声调 6→1、a/aa 韵腹，以及 n/l 声母、n/ng 韵尾等文本读音关联，也查看训练组对相应成分的覆盖。第 3 轮指出，句末“呢→咧”可能同时产生 e→i 和 n→l 两项关联，不能把它们当成两个独立发音问题。各组声调覆盖几乎一样时，它也没有据此声称能精确调节声调练习。

本轮把语音学信息用于理解错误、匹配已有练习材料和解释后续变化。实际控制仍是六组采样概率；这不是音频测得的 phone error rate，也尚未单独验证语音学提示的增益。

## 这些决策确实改变了训练曝光

| 训练组 | 池内录音 | 普通采样实际曝光 | MAS 实际曝光 | 普通占比 → MAS 占比 |
|---|---:|---:|---:|---:|
| easy_long | 556 | 5,637 | 2,877 | 29.86% → 15.24% |
| easy_short | 405 | 4,107 | 2,827 | 21.75% → 14.97% |
| medium_long | 303 | 3,075 | 4,220 | 16.29% → 22.35% |
| medium_short | 233 | 2,360 | 2,679 | 12.50% → 14.19% |
| hard_long | 171 | 1,742 | 2,793 | 9.23% → 14.79% |
| hard_short | 194 | 1,959 | 3,484 | 10.38% → 18.45% |
| 合计 | 1,862 | 18,880 | 18,880 | 100% → 100% |

easy 实际曝光合计 **51.61% → 30.21%**，medium **28.79% → 36.54%**，hard **19.60% → 33.25%**。MAS 虽然逐步降低了困难组的计划概率，整个训练中困难组仍明显高于普通采样；同时 easy 始终获得练习。

普通采样遍历了全部 1,862 条，每条 10–11 次。MAS 有放回抽样实际覆盖 1,860 条，已见录音各 1–31 次；未见的两条身份保留在曝光统计中。本方法改变练习次数和分配，不保证每个名义 epoch 看完所有数据。它也不是患者等权采样：vlink247 的曝光是普通 5,249 次、MAS 6,125 次。

## 五个能力如何组成闭环

| 能力 | 当前已执行的实现 |
|---|---|
| 诊断错误 | 查看 dev 整体、患者、句子、错误类型和读音成分关联的当前值与变化 |
| 找到练习 | 对照固定真实训练池中的同句材料、成分覆盖及六组分布 |
| 分配曝光 | Therapist 输出下一阶段六个采样概率，程序实际执行 |
| 检验收益 | 同一 Therapist 对照 dev 前后结果及实际曝光进行反思 |
| 记住并修正 | Skill Optimizer 整理文字经验，下一轮显式带入最近四阶段历史 |

实现为两个逻辑 LLM 角色、三个调用阶段：Therapist 制定分配、Therapist 反思、Skill Optimizer 整理经验。所有调用固定使用 laicai 的 `gpt-5.6-sol`。经验是显式传递的记录，不依赖模型记住此前对话。

可以据此描述当前研究主线：**PAT 将独立 dev 上观察到的识别错误转化为真实训练池的练习分配，结合执行后的真实效果积累经验，并调整下一阶段训练。** 这轮相同更新预算下的 test 优势，为这一主线提供了正向实验依据。

## 原样保留的训练方法

- Setting E：train 1,862 条/46 位患者，dev 394 条/10 位患者，test 372 条/10 位患者；患者、录音和音频哈希互不重叠，共享 60 个文本。所有 Agent 反馈与反思来自 dev。
- 原始 Qwen3-ASR-1.7B＋全新 Audio LoRA，seed42；两组起始 base 和 adapter 张量哈希一致。r16、alpha4、dropout0.05，仅训练 Audio LoRA。
- 逐录音平均交叉熵，AdamW，LR `2e-4`，weight decay 0，warmup5，590 步线性学习率；global batch32、microbatch4、每组两张 L20，梯度裁剪 1。
- 每 29 steps 制定一次分配，共 20×29＋最后 10 steps；优化器、scheduler、采样器与 RNG 连续。普通分支持续无放回随机遍历，MAS 先按概率选组，再在组内等概率有放回抽样。
- 固定六组“原始模型识别难度×长短句”，12 字及以下为短句。easy/medium/hard 不表示临床严重程度。
- Agent 的唯一训练操作是六组概率；语音、文本、分组、学习率、损失和可训练参数范围固定。四份中文提示词保持本轮原样。
- laicai Responses API，`gpt-5.6-sol`、xhigh、max_output_tokens8192、store=false。每次尝试总等待上限 600 秒；指定网络错误最多三次相同请求尝试，间隔 2 秒。SDK 内层重试为 0，外层重试有独立回执。
- 在反思 API 调用前保存 adapter、optimizer、scheduler、采样状态及各 rank RNG；恢复版本与来源保留。恢复后的 GPU 数值轨迹不承诺逐位一致。
- 评估采用 greedy Cantonese、空上下文、max_new_tokens256，原始每批 8 条；dev 患者等权 CER 选 checkpoint。test 仅报告，未提供给 Agent。

## 保留位置与复用

独立保存目录：

`/data/qwen3-asr/records/therapist_mas_setting_e_v1/preserved_methods/real_sampling_mas_seed42_590_v1`

- `runtime/`：完成版本的运行源码快照、四份提示词及相关启动/监督工具。
- `data/`：原始划分、六组成员与全部 2,628 条音频的 SHA-256；音频实体保留原位。
- `agent/`：21 轮公开结构化决策、反思、经验、输入与回执；原始响应仍可按来源记录追溯。
- `trajectory/`：两组各 21 阶段的计划与实际曝光，保留录音身份和顺序。
- `checkpoints/`：普通与 MAS 两组各自 step580、step590 的四份独立 checkpoint 副本，包含可用的训练状态与来源。
- `results/`、`analysis/`：完整曲线、两组 step580/590 逐条评估、全部 loss 事件、曝光汇总与逐轮分配 CSV。
- `provenance/`：历史执行版本、恢复来源、环境版本、复制来源、base 文件哈希和校验结果。
- `manifest.json`：保留包文件哈希；`DO_NOT_DELETE.md` 与仓库注册表记录保留要求。

`configs/completed_recovery_protocol.json` 是本次实际完成记录，含恢复设置。另存 `fresh_replication_protocol.template.json`，仅移除恢复专用字段并指向新的运行目录，训练和 Agent 设置保持不变；**此模板尚未启动新训练**。复跑时应使用快照中的对应源码及冻结数据，并重新完成必要的启动检查。原协议、`sources` 和旧账本均未改写。

保留包与列出的外部依赖已登记为保留对象。该标记用于后续清理时识别，未施加操作系统不可删除锁。四个 checkpoint 是实际副本，不依赖原 checkpoint 目录中的软链接。下一次方法修改使用新版本，当前成功版本继续作为对照。

核对入口：[21 轮决策、反思与经验](/data/qwen3-asr/records/therapist_mas_setting_e_v1/preserved_methods/real_sampling_mas_seed42_590_v1/agent/public_history.json)、[逐轮概率与实际曝光 CSV](/data/qwen3-asr/records/therapist_mas_setting_e_v1/preserved_methods/real_sampling_mas_seed42_590_v1/analysis/round_allocations.csv)、[完整评估节点 CSV](/data/qwen3-asr/records/therapist_mas_setting_e_v1/preserved_methods/real_sampling_mas_seed42_590_v1/analysis/endpoint_comparison/matched_steps.csv)、[保留注册表](/data/qwen3-asr/repo/dysarthric_yue_Speech/artifacts/registry/setting_e_mas_sampling_preserved_v1.json)。
