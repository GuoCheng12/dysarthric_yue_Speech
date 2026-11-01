# Therapist-inspired MAS 闭环：研究设计与实现审计

日期：2026-09-05。状态：审议稿，修改建议尚未实施。

审计范围仅限新的 therapist-inspired closed-loop adaptation framework，包括其 Evidence、Proposal、TTS/ASR 接口、Outcome、Reflection、Skill memory 和实验比较逻辑。历史实验只用于核对当前起点与比较条件，不重新评价整个项目。

审计基于当前工作区，包括尚未提交的改动；Git HEAD 为 `acb7403beaa7f83b2c899bd88fc3beeabeb76bdf`。本次只新增本文，没有修改实现、配置、prompt、测试或已有实验记录，没有启动训练、合成或付费 Agent 调用。

## 1. 结论与建议决策

**核心方向值得继续，但当前版本还不能视为已经验证的闭环学习系统，也不宜直接进入四轮正式训练。** 需要调整的重点是“如何定义可干预目标、如何观察真实响应、如何把响应带回下一轮”，而非继续增加 Agent 数量或校验层数。

建议保留：固定 Speech Generator、连续 ASR 更新、real replay、真实语音反馈、一个 Therapist 的 prescribe/reflect 两种模式、低频经验整理，以及不做训练前 proposal Gate 的路线。

建议优先修改六件事：

1. **把处方对象从单个错误签名，提升为可执行的目标音族与语境练习。** `k→∅`、`k→t`、`k→ng` 可以继续作为诊断，但三者共享的 `/k/` 文本载体不是三种独立干预。
2. **将“错误计数下降”与“识别真正改善”分开。** 当前已经复现整字删除使 phone 错误消失，却被标记为改善的情况。
3. **允许保护当前正确识别的能力。** 现有保护角色只能修复已经出现且达到阈值的回退，排除了预防遗忘和巩固已恢复目标。
4. **让反思读到真正的处方内容，并把建议送进下一轮。** 目前反思大量复制标签，下一轮又没有显式收到它的建议，CLI 还固定读取 v000 Skill。
5. **修复具体语义与运行缺陷，再接一个简洁的真实 round runner。** 重点是发音身份、剂量统计、重复事实一致性、浮点持平异常、来源核对及续训语义。
6. **将论文的关键比较改为预算匹配的闭环与开环比较。** 旧 M0 和旧 Full-1440 是有价值的参照，但无法单独隔离动态反馈的贡献。

本次没有发现现有 Setting D split 或冻结 reference 已经发生内容/音频泄漏。发现的是未来输入可以触发的隔离漏洞、实际 smoke 中的剂量解释错误，以及尚未接通的执行与学习接口。这些需要分开表述。

## 2. 从论文角度：故事成立在哪里，需要收紧哪里

### 2.1 有价值的研究问题应落在“有约束的反馈驱动练习设计”

目前最有价值的问题是：在患者语音稀缺、合成内容实现不完全可靠、适配收益与遗忘并存时，能否根据真实 ASR 的响应，调整下一轮的练习内容，使未参与反馈的真实 content 上获得更好的识别结果？

这个问题成立。Agent 的潜在价值在于将错误证据转化成多样、自然、带对照和保护意图的训练材料，以及根据后果改变设计。它不需要证明每个病理发音都能由 flat phones 描述，也不需要声称在治疗患者。

建议论文主张写成待验证的假设：**相同训练和合成预算下，反馈驱动的练习设计比固定设计更有效，并能减少对既有识别能力的损害。** “减少损害”需要最终真实指标支持；real replay 和保护角色本身不提供保证。

### 2.2 Agent + reflection + Skill 更新并不是充分的新颖性

最接近的已有工作是 [DataEnvGym](https://arxiv.org/html/2410.06215v2)：它已经把模型错误反馈、数据生成处方、生成引擎、学生训练与再评估组织成顺序决策闭环，并比较使用反馈与不使用反馈的策略。因此，不能把通用“Agent 观察错误后生成数据再训练”的循环本身写成主要创新。

文本反思与记忆方面，[Reflexion](https://arxiv.org/abs/2303.11366) 已采用反馈后的语言反思辅助后续决策；[ACE](https://arxiv.org/abs/2510.04618) 已研究通过生成、反思和整理来更新策略上下文。仅增加一个修改 SKILL.md 的角色，不足以区分我们的贡献。

语音方面，[Leung 等，Interspeech 2024](https://www.isca-archive.org/interspeech_2024/leung24_interspeech.html) 已研究 text-to-dysarthric-speech augmentation；[Synthetic Data Domain Adaptation for ASR，2026](https://arxiv.org/abs/2603.16920) 已把 LLM 文本生成与发音变化用于合成语音适配。这进一步要求我们用实验说明反馈、练习设计和能力保持的价值。

更有说服力的差异可以集中在三点：

- **从识别错误到可合成练习的转换。** 声学成分、词汇语境、同音字、整字漏读不是同一种问题，不能机械地按错误数补数据。
- **针对性练习与能力保持的联合设计。** 观察被修复的错误、被迁移成其他形式的错误，以及原来正确而后来变错的区域。
- **使用真实患者反馈持续调整。** 合成器只是执行工具，最终判断来自真实语音；不把合成音频的低 CER 当作训练价值。

这是建议的定位，不是已经证明的创新或穷尽相关工作的查新结论。

### 2.3 therapist-inspired 应体现可观察的决策，而不只体现角色名称

[ASHA 成人构音障碍资料](https://www.asha.org/practice-portal/clinical-topics/dysarthria-in-adults/)讨论了个体化目标、练习量与安排、任务变化和复杂度、保持与迁移、对照及功能性沟通。这些可以作为练习设计的灵感；将其迁移到 ASR 优化的有效性仍需要我们自己的实验。

目前系统已具备“评估—练习—复评”的结构，但以下行为尚不充分：保护正确能力、已改善目标的巩固、语境变化、根据结果明确调整一种练习设计。建议围绕这些行为形成方法，不要声称人类运动学习原则已经被证明适用于 LoRA。

两个长期角色可以保留。论文篇幅有限时，方法重点应是闭环策略；“multi-agent”和“self-improvement”不必同时成为主要 claim。若只有一次 Skill 更新、两轮后续应用，最稳妥的表述是本轮实验内的策略记忆调整。跨任务或跨患者泛化的 Skill 学习需要额外证据。

### 2.4 当前比较方案混合了动态反馈、额外训练和额外 real replay

当前草案每轮 `2117 real + 480 synthetic`，有效 batch=32，所以一轮为 `ceil(2597/32)=82` 个 optimizer steps。四轮增加：

| 项目 | 草案总量 |
|---|---:|
| 新合成语音 | 1920 条 |
| real replay 样本呈现 | 8468 次 |
| 新 optimizer steps | 328 步 |
| 起点 | Real-only step 50 的 M0 |

这与旧 Real-only step 50、旧 Full-1440 step 60 并不等价。即使 MAS 更好，也不能仅凭旧表判断收益来自实时反馈。依据见 [当前配置](/data/qwen3-asr/repo/dysarthric_yue_Speech/configs/therapist_setting_d_v1.yaml:61) 和 [baseline registry](/data/qwen3-asr/repo/dysarthric_yue_Speech/artifacts/registry/asr_baselines_checkpoints_v1.yaml:22)。

建议保留旧结果作为起点参照，优先增加下面的机制比较；本次仅定义，不启动：

| 比较对象 | 控制条件 | 能回答的问题 |
|---|---|---|
| M0 | 已有起点 | 整条流程有没有带来收益 |
| Real-only continuation | 同 M0、同新增 steps、同优化器/调度规则；明确 real 补足采样规则 | 相同新增训练步数下，合成干预是否优于继续重放真实数据 |
| 开环 Therapist | 同 M0，只用初始 Evidence 制定后续所有练习；同分轮预算、同 TTS/reference/训练安排 | 是否需要训练后的新反馈 |
| 闭环 Therapist | 每轮读取新反馈，其余条件相同 | 实时反馈的净价值 |

旧人工/静态 60 句增强可以继续列为参考。若用“开环 Therapist”作为预算匹配的 static 条件，能够同时控制 LLM 造句质量，减少新增实验数。所有组应报告时长、文本长度、TTS 调用量与训练量；相同条数不等于相同音频帧数或 token 计算量。

若论文主要强调闭环，**开环对照的优先级高于 without-skill-update**。Skill ablation 只有在信息回流确实接通、并且论文要声称 Skill 更新有增益时才是必要项。不需要为每张 proposal 恢复试训 Gate。

若进一步声称 Therapist 的推理优于简单“选错误最多的音族”，则应加入相同造句模型、相同预算的确定性选目标策略；否则将结论限定为反馈驱动处方的整体效果，不单独归功于复杂推理。这个对照可按最终 claim 选择，不必现在堆齐所有消融。

### 2.5 六个 feedback content 可以用于开发，但不能承担广泛泛化保证

真实 Setting D 是 48 train / 6 feedback / 6 test content。258 条 feedback 来自36位患者，但只有六种内容；它并不等于258个独立语言环境。Agent 反复读取聚合错误仍属于自适应使用反馈集，聚合与匿名化不会消除这种适应。[自适应数据分析研究](https://arxiv.org/abs/1506.02629)说明了重复利用反馈/holdout 的过拟合问题。

这不意味着当前设计不能做。建议明确叫“反馈开发集”，固定轮数和策略边界，把迁移结论放在其他真实 content 上。分析至少同时覆盖患者与 content 两个方向：逐 content 结果、逐患者结果、删去单个 content 后结论是否反转。仅患者 bootstrap 的区间不能表示新 content 的不确定性；只有六个测试 content 时也不应把形式上的窄区间解释成广泛语言泛化。

当前 M0 在 dev easy/medium 上较 zero-shot 有改善，而过去观察到的主要回退发生在 test easy/medium。因此，不能假设同一个六句反馈集天然能暴露所有泛化回退；这是代表性问题，不是给 Agent 更多解释就能解决的问题。依据：[已有 baseline 记录](/data/qwen3-asr/repo/dysarthric_yue_Speech/records/setting_d_lora_baseline_step_v1/README.md:28)。

现有 test 已在历史实验中多次被看过。“从现在起锁定、不再反馈给 MAS”是可以执行的规则，但不能追溯性地称其为项目从未使用过的测试集。当前仍可开展诚实的固定基准比较；若论文需要更强的独立验证，应另行保留真正未用于方法开发的内容/数据。也不能简单重新轮换 split：若新的测试 content 曾用于当前 Speech Generator 的训练，TTS 来源也必须重新处理。

### 2.6 先统一主指标，再谈改善与回退

历史常说的 dev 19.00% 是 mean-utterance CER；当前 Evidence 中同一 M0 的 pooled CER 是18.7004%。两个都正确，但不能混用。建议固定论文的主 CER 定义，同时报告另一个和 patient-macro CER；历史参照需要用同一份预测按统一定义汇总。

easy/medium/hard 当前是按 zero-shot CER 冻结的**语音级难度**，阈值为20%和50%，不等同于临床严重程度或参考池的患者级分层。这一边界在 prompt 中已经解释清楚，应继续保持。

## 3. 最重要的测量与处方逻辑问题

### M1：phone 计数下降可以来自整字删除，当前却判为改善〔高优先级，已复现〕

| 内容 | before | after |
|---|---|---|
| GT | 我想去見客 | 我想去見客 |
| ASR | 我想去見卡 | 我想去見 |
| CER | 20% | 20% |
| `coda k→∅` count | 1 | 0 |
| 当前 Compiler 判断 | — | `improved`，方向预期可为 `met` |

after 并没有识别对“客”。因为整个字被删除，不再属于可归因的字符 substitution，所以该 phone 错误从计数集合消失。相同问题也可能由 alignment/G2P 归因覆盖变化引起。

根因不是编辑距离算错。保守 phone extractor 的定义合理，但其**下界计数不能直接充当正确率**。目前 off-target 可能显示新增整字删除，却不会阻止 primary 被标为改善。

证据：[phone 事件归因](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/evidence.py:349)、[ClusterChange 标签](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/round_schema.py:470)、[outcome 计算](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/outcome.py:259)。

建议：保留 signature count 作为诊断；在相同 GT 载体位置上增加配对迁移：变正确、仍是原错误、换成其他替换、整字删除、无法可靠归因。无法归因应显式留下，不能算成恢复。Reflection 应看到这些迁移，不能仅凭 lower-bound count 下降学习“这类练习有效”。这属于训练后的测量修正，不是恢复 Gate。

验收标准：上表必须输出“未发生正确恢复，错误形态改变”；真正把“卡”识别成“客”时才能计入正确恢复。

### M2：`supported/refuted` 把方向符合度写成了科学判断〔高优先级，设计问题〕

当前 [规则](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/outcome.py:580)是：全部 outcome 达成→supported；全部 primary 的 compiled_verdict 为 not_met→refuted；其余→mixed。当前 primary 要求下降，因此完全持平也属于未达成；一条错误消失就可能满足下降。

这只能说明“预先写下的方向是否出现”，不能单独证明假说或机制；还会受 M1、训练剂量及有限内容的波动影响。更进一步，无条件强制的全局 safety 只有 easy/medium；采用 contrast/protection 时另要求对应 cluster outcome，但并未强制 hard/overall 不退化。因此 hard 和 overall 变差也不必然阻止 supported。当前测试中就存在 hard 退化而 supported 的情形。

建议将机械标签改为 expectation pattern，保留精确数值、方向和按项核对；把解释的证据强弱单独交给反思。允许“未观察到预期变化，原因尚不确定”，避免一轮未下降就形成“应避免该策略”的规则。这里不要求为小规模开发引入复杂统计打分系统。

### M3：计数单位混淆了引用、文本载体与实际发音〔高优先级，真实 smoke 已受影响〕

现有 `grounding` 只统计一条刺激主动引用的 clusters。Case 1 报告 `/k/` primary exposure=28，但未引用 primary 的 protection 句还有1个 `/k/` ×4 copies，全批文本的实际 `/k/` exposure 是32。

更重要的是，`k→∅`、`k→t`、`k→ng` 共享同一参考载体 `/k/`。ASR 训练只看 wav+GT，引用和角色标签不能把同一个音素呈现分成三种互不影响的治疗剂量。当前每句 primary cap 也只在该句引用 primary 时执行。

证据：[grounding 引用循环](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/grounding.py:19)、[剂量与上限](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/grounding.py:63)、[现有 smoke 报告](/data/qwen3-asr/records/therapist_mas_setting_d_v1/agent_smoke_v2/README.md)。

建议区分三个量：

- 处方声明投向某目标的 copies；
- 全批文本对某音族/语境的载体呈现，扫描所有句子，允许目标之间有真实重叠；
- 生成成功的音频数量、时长及实际经过何种内容核验。

合成文件通过非空、RMS、clipping、时长检查，不等于目标音被正确读出。当前 [realization](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/outcome.py:202)直接把文本 carrier 数复制为 realized 数，名称必须收紧；不能据此声称病理音素或内容已在声音中确认实现。建议增加少量分层试听/内容审计，而非用当前 ASR 的低 CER 自动筛除 hard 合成语音。

验收标准：Case 1 同时记录“primary 引用关联的文本 carrier exposure=28”和“全批 k 载体=32”；不能把共享音族三种错误的 exposure 相加成独立语音数量。

### M4：保护策略被写成“只能纠错”，与防遗忘目标不完全一致〔高优先级，设计问题〕

[Prompt](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/prompts/therapist.md:275)和 [validator](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/schema.py:1372)要求 protection cluster 当前比某 anchor 更差，并且所有刺激都必须引用 current targetable/actionable cluster。它还禁止以 primary 做 protection。

结果是：当前正确且稳定的模式不能被选为维护练习；上一轮修复到零错误的目标虽然留在历史中，也不能被继续巩固。两患者/两 content 的最低支持阈值还会排除较小但明确的回退。real replay 提供通用保持信号，但并不能补上 Agent 本身的这一决策缺口。

建议把“是否值得作为 primary”和“是否允许作为 maintenance/protection”分成两种条件。primary 仍要求错误证据；保护可来自已恢复目标、稳定正确模式的覆盖、或配对 newly-wrong 证据。保护不必伪造当前错误。第一版仍可保持患者分配和 real replay 固定，只调整生成内容及角色比例。

### M5：精确预算与强制 primary 可能使合法处方为空〔高优先级，正常状态未定义〕

当前必须生成60个 unique、480条 copies，并至少有一个 target；所有刺激需 current targetable/actionable。threshold-edge cluster 最多2句/16份，而 prompt 又禁止 null action。

当模型改善到没有 eligible cluster，或只剩一个 threshold-edge cluster 且无其他允许的保护/对照时，合同没有解。此时模型输出失败不代表模型推理差，增加 retries 也不能解决。

建议预先定义正常维护或停止条件。优先方案是在固定步骤框架中允许有证据的 maintenance/replay-only round；也可明确提前终止，但应重新定义与对照组的预算和最终轮次。它们都不同于“每张 proposal 预先试训后决定是否接受”。不要用事后临时放宽阈值填满预算。

### M6：有些保守限制合理，但不应写成普遍不可行性

同音字不能靠一个孤立音素区分，因此暂不作为 phonetic primary 合理。但包含语义和句法上下文的合成句仍可能帮助 ASR 选择字形。建议把 `tts_actionable=false` 解释为“不属于 v1 的发音载体干预范围”，不要写成“同音错误不可能通过 TTS 数据改善”。

`alignment_uncertainty` 的实现与说明相符，尚未发现算式错误；它衡量最优对齐是否不唯一，不是置信区间。phone 通常为0是严格事件筛选的结果，不能据此说2患者×2content已足够稳定。可改名为 alignment ambiguity，保留覆盖及支持计数。

## 4. 已复现的代码问题与执行缺口

### I1：CER 归一化仍污染发音载体和 content 隔离〔P1，执行前修复〕

同一根因有两个表现：

1. GT 的“冇”被 CER normalizer 改成“無”，character cluster 用“無”作为参考字。两患者/两 content 的“冇→有”可形成 targetable/actionable `無→有`。随后 grounding 拒绝包含“冇”的自然刺激，接受包含“無”的刺激；当前 G2P 中二者分别为 mou5、mou4。phone 路径已有 surface bridge，character 路径没有。
2. `identity_keys` 先 CER normalize 再算 Jyutping。“呢個咩字點讀呀。”与“呢個孭字點讀呀。”在当前 surface G2P 下同音，但第一句先将咩改为乜，导致四层 identity 全部不同，绕过所声明的同音内容隔离。

证据：[character 统计](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/evidence.py:638)、[actionability](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/evidence.py:765)、[character grounding](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/grounding.py:32)、[identity_keys](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/text.py:292)。

建议：评分身份继续使用 CER normalize；发音身份基于原始 surface，character 处方保留可回溯的 GT surface/readings；Jyutping 与 detoned 身份基于 TTS-facing surface。上面两个例子作为语义回归验收，不把当前实现的输出反过来当作测试真值。

当前 v6 选中的50个 cluster 未发现上述“冇/無”处方；实际 split 也另行用原始 surface 重算，未发现重叠。因此这是确定的边界缺陷，不是已证实的历史数据污染。

### I2：数学持平的 patient-macro CER 会使复评崩溃〔P1，已独立复现〕

三位患者各10个参考字，before 编辑数 `[1,2,3]`、after `[3,2,1]`，macro CER 严格都为20%。不同浮点累加次序产生尾数差异。Compiler 用 Fraction 判断 unchanged，却把浮点差传给按正负检查的 MetricChange，报 `metric response is inconsistent`。

证据：[macro 汇总](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/evidence.py:438)、[MetricChange 创建](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/evidence.py:1309)、[validator](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/round_schema.py:445)。

建议以同一准确统计结果决定 delta 与方向，浮点仅用于表示；这里是数值一致性问题，不要用研究上的“显著变化阈值”掩盖它。

### I3：正式 Reflection 校验仍接受同一事实的矛盾副本〔P1，已独立复现〕

使用真实 reflection smoke，在内存中修改派生字段并重算结果哈希，正式 `validate_reflection_pack(pack, protocol, evidence, proposal)` 仍接受：

- `after_metrics.easy.pooled_cer=0.069555`，但 `metric_changes.easy.after_value=0`、direction=improved；
- primary outcome after count=0，但 monitored cluster after count=5；
- realization carrier 数增加99，而已经按原 Proposal 核对的 sidecar 不变。

它核对了各字段自己的算术和引用，但没有把重复表达的同一个事实交叉核对。普通更新遗漏也能造成这种错误，不需要把问题理解为恶意篡改。强制 Agent 服从 Compiler 标签会放大后果。

证据：[RealSpeechResponse](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/round_schema.py:569)、[pack 校验](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/outcome.py:683)。

建议只保留一处事实来源，其他结果确定性派生；或者在正式入口从源 metrics/clusters/receipts 重编译后比较。增加哈希不能替代这一步。现有 source-aware 改动解决了部分 carrier 来源问题，但仍不等于所有派生事实已被核对。

### I4：one-primary 与 comparator 的代码契约比 prompt 宽〔P2，已复现〕

target 刺激只需引用集合与 primary 相交，因此能附加另一个 cluster 而不为它登记 outcome。contrast 使用 `any(...)`，只要一个 comparator 符合相同 level/reference，就能夹带另一个不符合的 comparator。

基于真实 Case 1 的内存修改，两种处方都通过 Proposal 与 grounding 校验。证据：[target 检查](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/schema.py:1313)、[comparator 检查](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/schema.py:1343)。

建议先决定 M3 所述音族级处方语义，再使 validator 逐项检查它。若暂保留旧 one-primary 规则，target 不得附加未分配角色的 cluster，每个 comparator 都应合法。

### I5：全局均衡不等于每种练习的患者分配均衡〔P2，真实 Agent smoke 的分配已出现〕

当前按 speaker ID 排序，按 Agent 输出顺序连续轮转。Case 2 和 Case 3 的确定性 materialization 分配如下，尚未合成音频：

| 角色 | hard 参考患者 | easy 参考患者 | medium 参考患者 |
|---|---:|---:|---:|
| target，16条 | 7 | 4 | 5 |
| protection，8条 | 6 | 2 | 0 |

这不证明 hard reference 无法保护 easy 语音，但角色策略与参考患者构成混杂。Agent 改变句子排列就能改变接受各类练习的患者，而这种变化在反思中不明确。

证据：[分配循环](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/materialize.py:312)。建议由 Harness 做固定、可复现的 role×reference strata 分配，记录实际交叉统计；无需开放患者选择给 Agent。可在配额内随机化患者顺序，而不是依赖 ID 的字典序。

### I6：reference 来源检查不足，且“全患者”名称不准确〔P2，当前数据干净〕

当前冻结池57位（19/19/19），train66位；`all_patients_balanced` 实际指全参考池。feedback36位和test33位均在该池内。

当前代码仅验证 speaker 属于 train、wav 存在和哈希正确，没有验证参考音频与 GT 确实属于这个 speaker 的 train 行。未来误选同一患者 dev 音频或配错文本也能通过。证据：[reference 读取](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/materialize.py:163)。

建议用 train 的 `(utterance, speaker, audio, GT)` 核对，并在 Evidence 中提供参考池覆盖的聚合统计。实际57条本次全部核对通过，不能将此缺口表述为当前泄漏。

### I7：RunStore 现在验证文件完整性，没有验证完整实验已执行〔P1，集成缺口〕

用一个无意义 txt 文件充当所有阶段 artifact，在临时测试目录中可以推进到 completed，且 `verify_integrity()` 通过。它检查的是顺序、名称、文件存在和是否变化；现有测试也用 `{}` 代替多数阶段文件。

证据：[advance](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/store.py:205)、[integrity](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/store.py:251)。这说明 Store 是组件，不是执行真实性的证明；不意味着已有科学结果被伪造。

建议保留简洁 Store，由一个正式 round runner 调用现有步骤、生成真实收据、进行阶段语义与父子 checkpoint 核对，再登记事件。优先打通一个真实数据路径，不要给任意 JSON 再堆多层自洽 hash。

### I8：续训语义尚未冻结，容易把“新增 steps”与“总 steps”混淆〔P1，集成前明确〕

ASR 脚本有 weights-only `--lora_init_adapter_path` 与 Trainer resume `--resume_from` 两条路线。前者创建新的 optimizer/scheduler；后者会恢复 global_step 等状态，但能否恢复 optimizer/scheduler 取决于 checkpoint 是否完整。当前 M0 有 scheduler、trainer_state 和 RNG 文件，却没有 optimizer 状态；按本地 Transformers 4.57.6 的加载条件，不能完整恢复优化状态。若从 step50 resume，却设置 max_steps=82，82表示总终点，不能保证新增82步；后续轮次可能已超过终点。

Protocol 只声明每轮重置 scheduler，没有说明 optimizer。warmup_ratio 等还有脚本默认值。证据：[adapter 初始化](/data/qwen3-asr/repo/dysarthric_yue_Speech/finetune/scripts/qwen3_asr_lora_sft.py:208)、[训练入口](/data/qwen3-asr/repo/dysarthric_yue_Speech/finetune/scripts/qwen3_asr_lora_sft.py:321)。

建议第一版选定唯一语义：加载上轮 adapter weights，每轮重置 optimizer/scheduler，记录本轮新增 steps；或者完整连续恢复所有状态，但必须明确跨轮数据与总步数。两者都可成立，不能混用。

本地 Transformers 4.57.6 的 ceil82 和最后5条 accumulation 归一化逻辑是合理的，没有证据需要再认定这里有此前同样的梯度 bug。另一方面，M0 选在历史 dev 最优 step50 后，再做四次高学习率重启属于较强干预；82步是否合适需要在统一协议下实测，不能默认延长一定有益。

### I9：失败路径缺少实验记录，all-or-nothing 会丢弃整批工作〔P2〕

当前合成中任一条失败，临时生成结果在 finally 被删除，也没有失败 summary；RunPhase 没有 failed/aborted，收据只允许 complete。文档却写失败会留在 journal。

证据：[合成执行与清理](/data/qwen3-asr/repo/dysarthric_yue_Speech/synthesis/cosyvoice3_patient_sft/synthesize_manifest.py:589)、[阶段定义](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/store.py:16)。

建议保留简单的失败收据与已完成样本的审计记录，明确停止原因；不自动重试、替换 reference 或重新挑处方。记录失败与添加 fallback 是两件事。这也避免人工改到成功的尝试从实验历史里消失。

### I10：复现配置与文档有漂移〔P2/P3〕

- ASR base model fingerprint 只有 safetensors index，未覆盖实际权重、config、tokenizer；ASR runtime 和完整 decoding 参数也未像 TTS 那样充分记录。
- `docs/evaluation_protocol.md:39` 仍称 phonology 是待实现，而代码已经实现。
- `docs/evidence_pack.md:190` 写986 candidates，指向的 v6 实际为972；50 included、44同时 targetable/actionable 正确。
- 一些文档把 linkage dose、实际文本 exposure 与真实声音实现混称为 actual。

建议保存一次模型关键文件摘要、实际包版本、训练与解码参数，更新单一状态说明；不需要扩建通用配置或安全框架。

## 5. Prompt、Reflection 与 Skill：需要让推理真正影响下一步

### 5.1 当前 Therapist prompt 有价值，但将机制选择收得过死

做得好的部分包括：完整说明 CER/分母/难度、区分诊断与病理、区分 anchor、解释 phone 下界、禁止捏造证据、明确普通 SFT 不读取 contrast 标签。这些应保留。

问题集中在 [prescription policy](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/prompts/therapist.md:384)及其动作边界：固定 narrow signature、强制满预算、仅允许 regressed cluster protection、对照必须第二种同-reference 错误、所有解释与输出高度模板化。部分限制是运行约束，部分其实是尚未验证的教学策略，却被同时写成不可变规则。

建议只把数据、预算上限、执行接口、禁止修改的训练变量写成硬边界；把“什么时候集中练习、什么时候分散语境、什么时候维护已学会模式”留给策略与记忆。不能把所有策略固定在 system prompt 中，再期待 Skill Optimizer 对同样策略做出实质调整。

### 5.2 Reflection 缺少处方本身，又过度要求复述

当前 ReflectionPack 有目标、剂量、索引、哈希和大量 before/after 统计，却不包含 Agent 自己生成的完整 unique stimulus 文本、句式和设计意图。原始患者逐条 transcript 可以保持隐藏；Agent 自己创造的句子没有同样的隐藏必要。

[reflection prompt](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/prompts/reflection.md:20)规定主要标签必须复制，`competing_explanations` 又只能来自 Compiler 预置的三种 confound。因此它很难表达“词汇过于集中”“对照未控制上下文”“实际内容实现不稳定”等待验证解释。真实 smoke 的解释基本就是将各个标签重新说一遍。

建议形成一个精简的 intervention digest：unique synthetic text、role、关键载体位置、设计意图、全批长度/词汇/音族覆盖、role×reference 分配和真实质量统计。逐 copy ID、路径与大部分哈希留在本地审计记录，不必全部进入推理上下文。

Reflection 输出重点应改为四项：

1. 哪些目标位置真正恢复，哪些只是换了一种错误；
2. 目标收益与保持/泛化结果之间出现了什么冲突；
3. 哪些解释直接被观测支持，哪些只是有依据但尚未验证的猜想；
4. 下一轮改变哪一项可执行的练习设计，以及依据是什么。

数字和确定事实仍由 Compiler 提供。允许“待验证解释”并不等于允许捏造事实；每条解释应标明观察依据与不确定性。无需增加 reviewer Agent。

### 5.3 Reflection 到下一轮的通路还没接通

当前 [InterventionSummary](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/schema.py:701)没有 reflection、lesson 或 next-round guidance；[CLI propose](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/cli.py:151)固定读取 `skills/v000/SKILL.md`。底层函数可传其他 Skill，但正式入口没有 active version 的选择与绑定。SkillOptimizer 目前仅是文档和三行 prompt。

因此现阶段可以说“prescribe 与 reflect 两个接口已可调用”，不能说“反思驱动下一处方、Skill 已形成自我改进”。

建议让每轮 ledger 保存处方、结果和反思；下一轮 Evidence 接收上一轮的简短解释与计划调整；本轮 prescribe/reflect 共用同一 active Skill version/hash。Skill 更新只允许提出小修改或 no_change，并保留相反证据。两轮仅是触发时间，不是经验已经可靠的标准。

## 6. 建议收敛后的最小闭环

| 环节 | 应承担的工作 | 最小验收点 |
|---|---|---|
| Evidence | paired CER、错误族、保持/回退、目标位置迁移、支持与覆盖 | 错误换形不被写成恢复 |
| Therapist.prescribe | 一个明确练习目标、语境/对照/维护设计、受限剂量 | 输出可执行且合同有解 |
| Harness | 原始发音身份、内容隔离、固定参考分配、TTS、real replay、固定训练 | 真实文件与收据、parent与新增steps一致 |
| Outcome | 从唯一事实来源生成真实前后变化 | 派生字段不能互相矛盾 |
| Therapist.reflect | 解释响应、保留未知、给出下一轮建议 | 有依据地维持或调整设计，下一轮实际收到该建议 |
| Ledger/Skill | 保存完整历史、将有依据的简短经验带回下一轮 | 下一请求确实包含相应记忆/版本 |

仍然是顺序干预。每轮完成后如实记录结果、继续从该模型出发；不增加训练前 score/Gate，不需要多 Agent committee。若引入正常 maintenance/停止动作，应在运行前明确协议及对照条件。

## 7. 修改顺序与后续验收

建议下一次实施按以下顺序分批完成，避免接口反复返工：

1. **先定测量与动作语义。** 决定目标音族/语境、纠正与迁移、保护资格、无 eligible target 的正常行为，以及机械 outcome 的命名。对应 M1–M5。
2. **修确定性代码。** 发音身份、全批 exposure、精确数值方向、唯一事实派生、角色验证、参考来源与分层分配。对应 I1–I6。
3. **接通一个最小 round runner。** 固定续训语义、真实收据、失败记录、ledger 和 active Skill。对应 I7–I10及第5节。
4. **用合成测试数据跑两轮接口联调。** 包含真实纠正、替换变删除、没有 eligible target、保护正确模式、数值持平、合成失败和 Skill 版本切换；这些是工程验收，不是 proposal 试训 Gate。
5. **再运行真实小规模连续干预。** 使用实际 TTS 和 ASR 后端检验句子内容、训练剂量和反馈解释。真实 Round 1 完成并能影响 Round 2 后，才值得讨论 SkillOptimizer 的具体学习收益。
6. **最后冻结论文比较。** 根据小规模执行情况一次性定预算与轮数，再做共同 M0、共同预算的开环/闭环实验。不要在正式 test 上逐轮选参数或最佳 round。

前两步应在下一次真实训练前完成。SkillOptimizer 的完整实现可以后置，但信息回流接口不能等到论文实验结束后再补。

## 8. 本次验证证据与完成边界

| 检查 | 结果 |
|---|---|
| 当前全仓已有测试 | 96 passed；本次未新增/修改测试 |
| GT客→ASR卡→整字删除反例 | CER 20%→20%，phone count 1→0，被现实现标 improved |
| macro CER 真正持平反例 | 触发 metric response is inconsistent |
| 同音内容隔离反例 | surface Jyutping 相同，当前四层 identity 全不同 |
| 正式 Reflection 入口的矛盾事实反例 | 通过校验，证实重复字段未交叉绑定 |
| 真正按原始 surface 重算 split 隔离 | train/dev、train/test、dev/test 的 surface/Jyutping/detoned 交集均0 |
| 实际冻结 reference 核对 | 57条均为对应患者 train 音频，GT匹配，未发现dev/test引用 |
| 既有 Agent smoke | 3个局部输入，均为6句/24条；不是完整50-cluster、60句/480条验证 |
| 既有 Reflection smoke | 实际 API 调用成功，但 before=M0、after=zero-shot 的人工接口情景；未运行TTS/ASR干预 |
| 真实多轮 MAS performance | 尚无可据此验收的完整闭环结果 |

96项测试通过与上述反例并不矛盾：现有测试较充分地覆盖了结构、标签与内部一致性，但部分测试的真值沿用了同一套实现定义，未覆盖“数字正确但语义错误”和跨阶段事实漂移。

语音身份隔离的核验范围是当前 PyCantonese 对原始 surface 的解析：60个content中59个完全解析，train有1个含未解析字位，比较时保留未知原字；dev/test各6个content均完全解析。这是文本/G2P层面的核对，不是声学同音性的实测。

本次结论是：**保留研究方向，先修正可干预目标与反馈语义，再让已实现组件形成真正的信息闭环。** 接下来最有价值的工作是修好这些具体问题并观察真实两轮响应，而不是继续增加 schema、hash 或 Agent。
