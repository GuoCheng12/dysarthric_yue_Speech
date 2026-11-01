# Therapist Harness 独立请求审计与处方 smoke

## 审计范围

逐份阅读实际运行入口、全部 prompt 文件、初始 Skill、两种 API 请求的
system/developer/user 消息、JSON output schema，以及它们引用的计数和校验实现。
按独立、无历史聊天、无音频、无工具的模型调用审读，不假设 Agent 看过项目对话。

| 入口 | instruction / developer | user 输入 | 状态 |
| --- | --- | --- | --- |
| Therapist prescribe | `prompts/therapist.md` + 当轮 Skill | task + evidence + evidence_sha256 | 已审计、修订，运行一次在线 smoke |
| Therapist reflect | `prompts/reflection.md` + 同轮冻结 Skill | task + reflection_pack + reflection_pack_sha256 | 已审计、修订；使用实际 prompt 离线构造并校验请求 |
| Skill Optimizer | `prompts/skill_optimizer.md` | 尚无 request builder 或 user contract | 原三行草稿不构成独立可执行 Agent；已显式标注 draft，未调用 |
| 初始策略 | `skills/v000/SKILL.md` | 随 developer 消息注入 | 已补齐 maintenance 和自然集中练习的已批准原则 |

其他 Harness 模块负责数据、训练、计数和记录，没有发现额外隐藏的 LLM prompt。
`prompt_id`、TTS reference transcript 是数据字段，不是额外的 Therapist instruction。
本次不是 Skill Optimizer 学习：初始 Skill 的修改是人工修正既定策略，未新增经验规律。
历史 smoke 的 request、Skill 副本及结果未改写。

## 发现与修正

| 问题 | 原先可能如何被独立 Agent 误解 | 修正 |
| --- | --- | --- |
| “same Therapist” 没有明确无会话记忆 | 以为可以依赖上一轮聊天 | 两个 instruction 都说明 fresh/stateless；记忆只来自本次 history |
| user 消息只有 JSON 数据和 hash | 任务需完全从长 instruction 中推断 | 加明确 task：出一张完整处方，或反思而不出新处方 |
| Reflection 使用 Python property 简写 | 以为输入顶层有 response/realization/training | 增加真实 JSON 路径图，说明 response 是 round_result.response 的简写 |
| Reflection 缺少独立指标词典 | 自行猜 pooled/macro、difficulty、uncertainty 的口径 | 补 CER 聚合、before/after、冻结 utterance 分组、错误簇、coverage、off-target 等定义 |
| 初始 Skill 只写 error evidence | maintenance 仿佛也必须找错误 | 明确错误修复与 current-correct rehearsal 的证据来源不同 |
| tts_actionable 的短定义比长定义更强 | 把可请求的文本载体误当合成或收益保证 | 统一为本协议的文本可落地性，不预言声学实现或 ASR 收益 |
| “禁止计算一切数字”过宽 | 连解释结果也像被禁止 | 固定已测数字和方向；允许有标记的解释与未来定性假设 |
| 固定 copies 被写成固定 minutes | 错以为换句子不能改变总时长 | schema description 与 Reflection 均说明时长在合成后测量 |
| “maintenance prevents loss”像效果承诺 | 把处方意图说成已证明作用 | 改为 intent，不保证有效，也不否认普通 SFT 的跨角色影响 |
| carrier dose 多种字段混淆 | 引用计数当独立物理重复；整段时长当目标音时长 | 区分 linked/whole-batch、重叠载体、requested-text-only 与 linked_audio_seconds |
| 内容隔离与工具边界不清 | 声称查过未提供的语料或听过音频 | 明确无工具/音频；禁止的是整句 identity，不是共享词汇；隔离检查由 Harness 后做 |
| hash、版本与输出动作未集中说明 | 编造 hash、沿用输入 schema version、只输出示例 | 给出原值复制路径；输出版本服从 output schema；输出完整预算内处方 |

语义定义保留在 instruction，策略原则在 Skill，本轮事实和预算在 user payload。
历史建议不覆盖当前合同；生成句子和引用文本是数据，不是执行指令。
没有增加发音工具、自动修句、重试、Gate、角色或训练变量。

### user 请求变更的记录校验

新增 `task` 后，完整回归发现 RunStore 仍按旧 user payload 做严格比较。
已让 builder 和 RunStore 共用相同任务常量，保留严格来源校验。
新增测试确认：篡改 task 仍被拒绝，而真实 builder 构造的处方/反思链能通过。
这是本次接口变更需要同步的修正，不是通过放松校验掩盖问题。

## 验证方式与范围

- 完整 `tests/`：196 passed；本次修改的 Python 文件 ruff 通过。
- actual Reflection instruction + initial Skill 可独立构造完整请求，真实序列化路径、
  prescribed proposal、round_result.response、next_action_contract 都在 user 消息中。
- 两种请求不使用 previous_response_id，不提供 tools；schema 与事实原值校验保留。
- 本次 Reflection 只做离线接口检查，没有额外付费调用，更没有真实 ASR 干预。
- 自然性、解释合理性以及数值引用仍需逐条人工审读；结构校验不证明“零幻觉”。

参考了 [OpenAI Prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering)
关于明确角色、指令和相关上下文的通用原则；修正依据主要是本地真实请求与代码。
模型仍为指定的 gpt-5.6-sol / xhigh，未迁移模型。实时 model-guidance 链接返回了
GPT-6 内容，因此没有将那一页当作 GPT-5.6 的专属行为依据。

## 在线处方 smoke

入口为 `examples/therapist_maintenance_reflection_smoke.py --prescription-only`。
此显式模式只调用一次 prescribe，进行引用、载体、内容隔离和 reference 分配校验，
不进入反思测试夹具，不生成执行 receipts，不运行 TTS/ASR。

记录目录：
`/data/qwen3-asr/records/therapist_mas_setting_d_v1/prescription_prompt_audit_smoke_v1/`

使用真实 M0 dev 的 50 个 error clusters 和 12 个 capability anchors，历史为空。
预算为 6 个 unique 文本、24 synthetic 分配；正式草案的 60/480/2117 未改。
system 33917 字符、developer 1366 字符、user 115444 字符（字符数不是 token 数）。
实际请求存于 `01_prescribe/request.json`，包含完整 instruction、Skill、user 和 output schema。

### 在线结果

一次调用完成，耗时 188.31 秒；API usage 为 input 50512、output 8825 tokens。
未自动重试。返回 6 句，每句 4 copies，共 24 条分配；结构、引用、预算、载体、
四层内容 identity 隔离与冻结 reference 分配校验通过。原始 response 可由 SDK
重新加载；已核验实际发送的 instruction/Skill 与本次审计文件一致。
没有生成 WAV、训练 ASR、调用 Reflection 或创建真实干预记录。

主目标为 tone 1 → tone 6 的 ASR transcript confusion，当前 20 次，来自 12 位患者、
5 个 content；其中 hard 12 次、medium 8 次、easy 0 次。Agent 未将其说成患者实际
声调发音错误。对照为 tone 1 → tone 3；protection 为 tone 6 → tone 4 的 easy/medium
退化；maintenance 选择 easy onset-z 与 medium onset-m 的当前正确能力。

| 原始输出句子 | 角色 / copies | Agent 的设计理由（中文概括） | 人工语言审读 |
| --- | --- | --- | --- |
| 今晚阿欣喺街市精心揀新鮮蔬菜。 | target / 4 | 在街市场景集中练习第一声，改变词汇周边 | 语义通顺，精心/蔬菜偏书面；集中设计明显 |
| 醫生今朝提醒阿芝準時返公司。 | target / 4 | 换到诊所与工作语境，避免市场句的表面改写 | 语法成立，但医生提醒准时返公司的动机较刻意；六句中语用最弱 |
| 阿詩今日去新店試幾款咖啡。 | contrast / 4 | 第一声/第三声载体共现，平衡已有混淆，不宣称 contrastive loss | 意思清楚；属于专门设计的完整句子 |
| 阿玉食完粥就落樓寄文件。 | protection / 4 | 练习第六声，针对 easy/medium 已观察到的退化 | 日常事件顺序清楚 |
| 子晴早啲整理書桌再出門。 | maintenance easy / 4 | 保留 /z/ 载体位置的整字识别能力 | 可理解；書桌较书面，香港口语常可改作書枱，原输出未改 |
| 媽媽聽朝買米煮晚飯。 | maintenance medium / 4 | 保留 /m/ 载体位置的整字识别能力 | 意思清楚 |

两条 target 分别含 9、8 个程序标注的 tone-1 载体，已不受旧上限阻拦。
全批（包括其他角色的附带载体）共有 120 次 tone-1 文本呈现；其中显式关联主目标的
三句为 88 次。tone 1→6 与 tone 1→3 的全批计数都为 120，因为共享 GT tone-1，
不能相加成 240 次独立练习。统计仍是文本覆盖，不是实际声学剂量。

### 数值与语义核对

逐项核对了全部 6 条 assessment claims 和 expected outcome 中的具体数字：
current 与 M0 的 36 位患者均持平；相对 zero-shot，easy/medium 各有 23/5 条退化，
总体 4 位患者退化；hard CER 为 0.5088235294117647、54 条中 27 条 critical；
主目标 20 次、12 人、5 content；protection 10 vs 5 次且 easy 为 5 vs 1、medium
为 3 vs 1；两条 capability 的 M0-lost 均为 0；对照当前 10 次，均与输入吻合。

Agent 正确区分总组 CER 改善与其中部分语音退化，没有沿用项目旧对话中“easy 总体
一定退化”的说法；明确语料隔离是稍后 Harness 执行的检查，没有冒充自己调用工具。
没有发现本次输出编造患者、历史试验或已实现的训练收益。

结论：技术链路通过；文本整体可解释，但有上述语言/语用改进空间，不把语法合法
等同于最自然的香港口语。没有改写 Agent 原句来制造通过结果，也没有因集中载体
较多就判坏。单次出方不能证明长期稳定性，且本次没有重新在线验证 Reflection。

## 仍需后续研究决定的边界

本次不擅自更改一轮单主目标、句长、重复份数和 threshold-edge 整批剂量限制。
这些是当前协议的设计选择，不是言语治疗已证明的普遍定律。
Skill Optimizer 尚需独立定义输入、输出和调用流程，不能仅凭其草稿说已经搭好。
没有一次真实 ASR 干预，就不能用本次 smoke 证明保护有效、处方有用或 Skill 学习有效。
