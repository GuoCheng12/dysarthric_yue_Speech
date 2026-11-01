# easy/medium 保护与 Reflection 回流实现

## 后续决定：自然句子优先，允许有理由的集中练习

用户确认：句子不必都是日常高频请求，可以是自然、有意义、专门针对错误设计的
完整粤语句子。已删除每句 primary carrier 最多两个的配置、schema 字段和拒绝逻辑；
没有改成另一个密度阈值，也没有加入发音查询工具。Therapist 在已有 rationale 中
解释集中练习或多语境练习的选择；Reflection 不再仅凭载体较多推断质量差或过拟合。
载体统计和引用落地检查保留。原有 threshold-edge 整批剂量限制本次未改，
它与取消的每句上限不同；训练预算、reference 和 ASR 配置均未改。

本次相关回归 187 项通过，修改的 Python 文件 ruff 通过。两句历史失败文本
`姐姐朝早去街市買菜。`、`阿詩今晚去睇戲。` 在新单元测试中均通过处方与载体检查，
仍各计三个 tone-1 载体。这不是重新运行在线 Agent 的结果。

已从原始预测重新构建并核验
`/data/qwen3-asr/records/therapist_mas_setting_d_v1/evidence/round_001_m0_evidence_pack_v9.json`。
与 v8 相比，仅 action contract 删除该上限；50 个错误簇、12 个能力锚点、预测指标和
60 文本 / 480 synthetic / 2117 real 的预算不变。v8 和历史 smoke 原样保留；
含旧字段的历史合同不能直接作为新请求输入，须重新构建，不做隐式兼容或改写。
本次没有在线 Agent 调用，也没有 TTS/ASR 训练。以下为修改前的实现与 smoke 历史记录。

## 原始实现与 smoke 记录

本次完成处方、评测、反思和下一轮输入之间的代码接口；不启动正式 TTS/ASR 训练。
没有恢复训练前 Gate、试训打分、rollback 或 best-round 选择。

最终验收状态：

| 项目 | 结果 |
| --- | --- |
| 代码回归 | 191 项通过，ruff 通过 |
| 自动出方在线 smoke | 两次均被载体计数上限拒绝，未执行 |
| Reflection 在线 smoke | 补全动作边界后，结构和人工语义审读通过 |
| 下一轮回流 | 实际反思、经验、建议完整进入下一轮 EvidencePack 与 API 请求 |
| 下一轮 Agent / TTS / ASR | 本次未执行 |

因此，保护与反思回流接口已完成，但不能说完整自动闭环 smoke 已通过。

## 1. 保护不再只从错误中寻找

两种角色分开：

- `protection`：修复已经观察到的 easy/medium 退化。必须是同一难度组的
  current 错误数高于 M0、previous 或 zero-shot；仅在 hard 增加错误不合格。
- `maintenance`：练习已经能正确识别的内容载体，保留原有能力、巩固新获得的能力。
  不要求先存在错误，也不伪造 error cluster。

新版 EvidencePack 在原有错误簇外增加 `capability_anchors`。每条是
GT 的声母/韵腹/韵尾/声调载体 × frozen easy/medium 分组，显示当前、M0、previous
正确数量及保留、丢失、新增正确。这里“正确”要求该位置的整个字被正确识别且
对齐唯一；一个错误字恰好有相同声母，不算成功。这不是对患者病理构音的测量。

锚点需要至少两位患者、两个 content 的当前正确支持，最多选 12 个，按难度和
成分类型轮流选取。真实 M0 dev 得到 12 个锚点，easy/medium 各 6 个。
完整正式预算的输入已另存为：

`/data/qwen3-asr/records/therapist_mas_setting_d_v1/evidence/round_001_m0_evidence_pack_v8.json`

处方必须为可用的 easy/medium 锚点组分别安排 maintenance，并且新句子的文本
确实包含被引用的载体。每个锚点对应一个 `capability_lost_count` 的轮后检查。
剂量由 Agent 在固定总预算内分配；没有预先强制某个 target/maintenance 比例。

维护检查固定“本轮之前正确的那些位置”：

`丢失数 = 原先正确的位置数 − 其中训练后仍正确的位置数`

before 丢失数为 0，after 为本轮丢失数。别的位置新识别正确不能抵消旧位置丢失。
对齐不确定、插入错误另行显示。easy/medium 整体 CER 检查也保留，两者互补。
这些都是轮后观察，不是训练前筛选或训练中的硬约束。

训练方式不变：从上一轮 adapter 权重开始；固定学习率、LoRA 和 step；每轮完整
replay 2117 条 real，加本轮 synthetic，不累积上一轮 synthetic。正式草案仍是
每轮 60 个文本、480 条合成分配；reference policy 和 Speech Generator 没有改变。

## 2. Reflection 是 Therapist 的第二种调用模式

不是新设一个审稿 Agent。程序计算事实，同一个 Therapist 负责解释并调整下一轮。

ReflectionPack 现在直接包含实际处方句子、角色、copies、理由、预期结果；同时给出
实际剂量、真实语音配对结果、历史，以及下一轮完整的 `next_action_contract`。
Agent 可以看到“到底练了什么”，也知道哪些下一轮动作是固定边界，不只看到簇 ID。
当前 ReflectionPack JSON 版本为 3，Reflection 输出 JSON 版本为 4。

输出区分：

- `compiled_expectation_verdict`：程序对预登记方向检查的机械汇总。
- `hypothesis_verdict`：Agent 对假设的解释，可以是 inconclusive，不能冒充已证实因果。
- `competing_explanations`：允许提出新的解释，但须引用观测并给出可检验的预测。
- `lesson_candidates`：条件、建议、限制和预测，仍是暂定经验。
- `next_round_guidance`：下一轮保留/调整什么、为什么、预期观察什么；不在此生成下一张处方。

`primary_practice_dose_direction` 专指 target＋contrast 的 copies 份额变化。
synthetic 总量、unique text 数、real replay 和训练步数不变，只在角色间重新分配。
下一轮 maintenance 针对当前仍正确或新获得的能力；已经丢失的位置要通过下一轮
证据支持的 protection 或 primary 进行修复，不能只改标签就称为保护。

## 3. 反思不再留在单独文件里

`compile_intervention_summary` 将实际 Reflection 的解释、经验和下一轮建议写入
带 hash 的 episode。`committed` 验证来源，`export-history` 导出完整历史，下一轮
EvidencePack 必须包含每个已完成轮次。即使 Skill 未更新，上一轮建议也立即进入下一轮。

Skill 的版本激活与 episode 分开：一轮的 prescribe/reflect 使用同一份 Skill。
按当前计划，round 2 后的更新只影响 round 3；也可以明确选择 `no_change`。
`propose` 不再隐式读取 v000，必须传 `--skill` 或 `--skills-dir`。
本次接通版本存储和激活，没有声称已完成 SkillOptimizer 的模型调用。

## 4. 验证范围

全套测试 191 项通过，ruff 检查通过。关键覆盖：

- 零错误的能力也能进入 maintenance。
- hard-only regression 不能冒充 easy/medium protection。
- maintenance 引用和句子载体不一致时，整张处方校验失败，不替换句子。
- 一个单元用例的正确总数 3→4，但丢失原有正确位置 1 个、另新增 2 个：maintenance
  仍判定未达标，不能用总数净增长隐藏遗忘。这是单元用例，不是真实干预结果。
- Reflection 收到实际句子；编造引用或篡改数值不通过；有依据的新解释可通过。
- 上轮建议出现在下轮输入；Skill 更新只影响后续轮次，历史不可覆盖。
- API Response 按公开字段 alias 序列化，保存后可重新加载；不会把内部 `schema_`
  错当成 API 的 `schema` 字段。

在线 smoke 使用 `gpt-5.6-sol / xhigh`，链路为：出方 → 反思 → 下一轮出方。
每张 smoke 处方只设计 6 个文本、24 个分配，不合成音频。
为了在没有正式训练的情况下检查回流接口，轮后反馈使用真实 M0 与 zero-shot dev
预测构造的 counterfactual fixture，receipt 也明确是占位数据。
整条记录标注 `counterfactual_interface_smoke`，不能提交为正式训练历史。

运行入口：`examples/therapist_maintenance_reflection_smoke.py`。
首轮在线失败记录目录：
`/data/qwen3-asr/records/therapist_mas_setting_d_v1/maintenance_reflection_smoke_v1/`

第一张处方在语义层安排了 target 8、contrast 4、protection 4、maintenance 8 条。
但 `姐姐朝早去街市買菜。` 中有三个主目标 tone-1 载体（第二个姐、朝、街），
超过每句两个的原有限额，整张处方被拒绝，没有合成或自动替换。
请求耗时 267.3 秒。拒绝输出和原因已保留。
指令补充了完整句子计数和叠字语境读音的例子，上限没有放宽。

修订后独立 smoke 结果目录：
`/data/qwen3-asr/records/therapist_mas_setting_d_v1/maintenance_reflection_smoke_v2/`

第二次出方仍未通过同一上限，耗时 319.9 秒。句子 `阿詩今晚去睇戲。` 的程序读音中，
PyCantonese 将“阿”映为 `o1`，与詩、今一起计作三个 tone-1 载体。这说明只补一段
提示词不能可靠统一 Agent 与实际 G2P 计数口径；不能据此声称完整自动出方已通过。
没有放宽约束、自动替换或继续盲重试。

为独立验证 Reflection，另外构造明确标注的手工测试输入，只把被拒绝处方中的
第一句替换为已核验的 `我哋去買新衫。`，其余保持同一测试设计。
这是独立测试夹具，不算 Agent 出方成功，也不是生产管线的 fallback。
运行入口通过显式 `--reflection-only-fixture` 选择该测试；只调用 Reflection，
之后编译下一轮 EvidencePack 和 API 请求，不调用下一轮出方。
相关记录：

`/data/qwen3-asr/records/therapist_mas_setting_d_v1/maintenance_reflection_only_smoke_v1/`

这次 Reflection 的结构和回流均通过，但人工审读发现它建议减少合成总量，并混用
maintenance 与修复丢失能力。原始输出未改；另外保存 `human_review.json` 标注
“结构通过、语义需修正”。由此补充了完整下一轮 ActionContract，并重命名含糊的
剂量字段。不是用字符串过滤器掩盖输出，也没有重写模型结论。
同一测试输入的修订版 Reflection 复测目录为：

`/data/qwen3-asr/records/therapist_mas_setting_d_v1/maintenance_reflection_only_smoke_v2/`

最后一次 Reflection 使用 `gpt-5.6-sol / xhigh`，耗时 209.5 秒，返回后通过结构校验、
存档重载和人工审读。它明确提出：从原先 13 条 target＋contrast 中重新分配份额，
保持 24 条合成、6 个文本、2117 条 real replay 和既定步数不变；maintenance 针对
下一轮仍正确/新获得的能力，已丢失能力则由下一轮证据支持的 protection/primary 修复。
它也基于实际句子指出两条目标练习都围绕购物/新衫，提出重新考虑场景分配的假设。

程序的机械预期汇总为 `refuted`，Agent 对因果假设判断为 `inconclusive`，并明确说明
原因是 counterfactual fixture，而非真实干预。两者成功分离，没有强制 Agent 复述因果标签。
已验证实际 Reflection 的 rationale、lesson candidates、next-round guidance 与下一轮
EvidencePack / API request 中的内容完全一致。下一轮请求已编译，未调用下一轮 Agent。

角色说明是处方意图，不是声称 maintenance 数据在真实训练中绝不可能顺带修复其他错误。
它们都仍使用普通 ASR SFT，没有特殊角色 loss。

下一步的生成端问题是：让 Agent 与执行核验使用同一套可查询的发音分析，或重新讨论
是否需要如此严格的每句载体上限。本次没有擅自修改这项处方边界。

只有之后真正执行 Speech Generator → ASR 更新 → 真实 dev 评测，才能回答
maintenance 是否保护了 easy/medium，以及闭环是否改善 ASR。
