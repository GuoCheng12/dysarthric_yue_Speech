# 两轮小剂量 Therapist 闭环

本轮执行记录：`/data/qwen3-asr/records/therapist_mas_setting_d_v1/therapist_small_dose_loop2_v1`。
入口：`examples/therapist_small_dose_loop.py`。这是实际在线 Agent / TTS / ASR / Reflection，不是接口 fixture。

## 固定设置

- 两条轨迹都从原 Real-only M0 checkpoint-50 开始，不从上一个小型校准实验的 checkpoint 开始。
- MAS 每轮：6 个新句子，Speech Generator epoch 8 合成 24 条，加入 232 条 real replay。
- 对照每轮：相同 232 条 real，再加 24 条 real；不调用处方，也不作为训练前 Gate。
- 8 × 4 batch，8 个新 optimizer steps，lr=2e-5 constant、无 warmup，utterance-mean 梯度路径。
- 每个有效 batch：29 条共享 real + 3 条替换位置。配对顺序固定，SequentialSampler；每行恰好见一次。
- 原 2117 real 是抽样池，不是每轮训练全集。按 speaker × frozen utterance difficulty 分层抽样，轮间不复用 replay，抽样 seed 为 43 / 44。
- 两条轨迹第二轮分别从自己的第一轮 step 8 继续，重置 optimizer/scheduler，仅加载各自 LoRA 权重。
- step 4 / 8 解码同一份 511 条 merged evaluation；始终继续 step 8，不据此挑选 best、回滚或拒绝处方。
- 511 条同时用于反馈和展示评测；没有独立 sealed test。原 dev/test 仅为解码分片。
- Therapist 固定 gpt-5.6-sol / xhigh；Skill v000 固定，实际 Reflection episode 立即回流下一轮 EvidencePack。
- real-only 对照结果单独保存供实验解释；不伪装成 ReflectionPack 中不存在的 Agent 证据。

## 核验与记录

每轮保存 evidence、完整请求/原始响应/耗时、proposal、allocation sidecar、TTS manifest/输出时长与文件 hash、real 抽样名单、成对训练顺序、训练命令/父 checkpoint hash、逐 step loss、4/8 两次完整解码、RoundResult、ReflectionPack、原始 Reflection 与 ledger。

开始前修正了两处旧接口假设：prompt 不再声称每轮 replay 全量 real；协议允许明确没有独立 sealed test。同时 RunStore 复评支持与 EvidencePack 相同的 JSONL/CSV 预测读取，不再只读 CSV。

验收条件是两轮实际执行到 `committed` 并保留完整溯源，而不是 CER 必须改善。全局旧 `completed` 状态对应尚未实现的独立 final-test 合约，本实验不伪造该阶段。

## 实际结果

两轮训练、全部评测、两次真实 Reflection 均已完成，两轮事件都到达 `committed`。没有启动第三轮，没有替换原 M0。

以下除 pooled / patient-macro 两列外，CER 都是逐句 CER 的平均值，单位为百分比。各行均使用同一份 511 条真实语音。每轮报告固定 step 8，不按结果选 checkpoint。

| 模型 | 平均逐句 CER | easy | medium | hard | pooled CER | patient-macro CER |
|---|---:|---:|---:|---:|---:|---:|
| 原 M0 | 21.96 | 7.40 | 24.76 | 56.16 | 21.33 | 19.12 |
| Real-only，第 1 轮 | 21.88 | 7.50 | 24.96 | 55.19 | 21.27 | 19.22 |
| MAS，第 1 轮 | 21.69 | 7.36 | 24.59 | 55.10 | 21.08 | 19.23 |
| Real-only，第 2 轮 | 21.83 | 7.44 | 24.99 | 55.05 | 21.24 | 19.34 |
| MAS，第 2 轮 | 21.87 | 7.32 | 24.58 | 56.16 | 21.24 | 19.26 |

全部中间点的平均逐句 CER：

| 累计新增 optimizer steps | Real-only | MAS |
|---|---:|---:|
| 4 | 21.8496 | 21.6959 |
| 8 | 21.8801 | 21.6856 |
| 12 | 21.8770 | 21.7922 |
| 16 | 21.8296 | 21.8718 |

第一轮 MAS 相对同轮对照改善约 0.195 个百分点；第二轮反而高约 0.042 个百分点。第二轮两组 pooled CER 精确相同，均为 21.2439903846%（1414 次编辑 / 6656 个 GT 字符），patient-macro 则 MAS 较低。结论不是“某一组全面胜出”，而是第一轮的小幅总体领先没有稳定保持。

## 两轮处方与真实回流

每句均生成 4 条语音。

| 轮次 | 角色 | 句子 |
|---|---|---|
| 1 | target | 司機試咗新路線先至離開。 |
| 1 | target | 子怡知道件事之後，先致電律師。 |
| 1 | contrast | 老師先將車費寫喺白板度。 |
| 1 | protection | 阿森啱啱斬好鹹魚。 |
| 1 | maintenance | 朝早姐姐坐船去中環。 |
| 1 | maintenance | 阿嘉下晝返屋企沖茶。 |
| 2 | target | 等二嬸搬好啲椅子先開會。 |
| 2 | protection | 今晚臨時改喺深水埗集合。 |
| 2 | maintenance | 阿祖租咗個帳篷畀校方做道具。 |
| 2 | maintenance | 畫家下星期喺沙田搞展覽。 |
| 2 | maintenance | 舅父今日改搭小巴去舊街。 |
| 2 | maintenance | 東涌嗰間舖賣芒果雪糕。 |

第一轮主目标是 Jyutping 韵核 `i → yu` 混淆，同时对照 `i → e`、修复韵尾 `m → n`，维持 easy 的 `z` 声母和 medium 的 `aa` 韵核。

第一轮总体 CER 改善，但主目标签名错误 64 → 69，配对目标位置恢复尚不明确；对照签名 18 → 19，保护签名 29 → 30。两个 maintenance 锚点各丢失 2 个原本正确的位置。36 位患者中 pooled CER 10 位改善、20 位持平、6 位变差。

实际 Therapist Reflection 判定原假设 `refuted`，没有把总体 CER 的改善当成定向修复成功。它建议降低 target+contrast 剂量、增加 maintenance、更换语境。该 Reflection 的完整 ledger 已逐字段进入第二轮 EvidencePack，第二轮确实将 target+contrast 从 12 条降到 4 条，maintenance 从 8 条增到 16 条，protection 保持 4 条，移除 contrast。

第二轮主目标 69 → 69，`m → n` 30 → 30，仍未建立配对恢复。`z.easy` 本轮 lost=0、新正确=3，保持成功；但 `g.easy` lost=2、`aa.medium` lost=3、`ng.medium` lost=2。easy/medium pooled CER 继续改善，hard 收益却回到 M0 附近。相对第一轮，患者 pooled CER 6 位改善、22 位持平、8 位变差。

第二轮真实 Reflection 仍判定假设 `refuted`，下一轮建议是 `retarget`：根据新 EvidencePack 更换主目标，保留 target+contrast 合计 4 条的小剂量，适当将 maintenance 份额转给有证据支持的 protection，仍保留所需 easy/medium maintenance。这只是已记录的建议，未自动执行第三轮。

Real-only 的 `i → yu` 签名也从 64 → 70 → 72；因此不能把 MAS 的目标次数上升简单归因于 synthetic。签名次数不等于该单位所有位置的识别质量；完整配对恢复/迁移诊断单独保存在 `shadow_target_diagnostic.json` 和累计诊断中。

以同一个 M0 为前快照累计比较两轮，`i` 载体位置的 Real-only 恢复/退化为 12/8，MAS 为 7/12；所以 MAS 的精确 `i → yu` 次数更少，也不能直接解释成该韵核所有位置识别得更好。反过来，原 M0 正确的 `z.easy` 位置累计丢失数是 Real-only 2、MAS 1，`aa.medium` 是 Real-only 4、MAS 2，体现了部分保持迹象，而非所有保护都失败。两类结果需同时保留。

## 实际剂量，不只看角色名称

| 项目 | 第 1 轮 | 第 2 轮 |
|---|---:|---:|
| 共享 real | 232 条，42 位患者 | 232 条，43 位患者 |
| 共享 real 覆盖内容 | 48 | 48 |
| 共享 real 时长 | 26.66 分钟 | 26.69 分钟 |
| Synthetic | 24 条，20 位患者 | 24 条，20 位患者 |
| Synthetic 时长 | 2.236 分钟 | 2.503 分钟 |
| Reference 患者 easy/medium/hard 分层对应合成条数 | 10 / 8 / 6 | 7 / 10 / 7 |
| 全批 `i` 文字载体覆盖 | 72 | 40 |
| 全批 `z` 文字载体覆盖 | 56 | 32 |
| 全批 `aa` 文字载体覆盖 | 44 | 56 |
| 全批 `m` 韵尾文字载体覆盖 | 20 | 24 |

两轮共 12 个新句子、48 条音频、4.739 分钟合成语音。MAS 实际见到 464 个不同 real 和 48 个 synthetic；对照见到 512 个不同 real。训练总曝光均为 512 行，而不是各自完整 replay 两遍 2117 条。

角色剂量与实际覆盖并不一一对应：第二轮 maintenance 条数翻倍，但所有句子合计的 `z` 覆盖反而减少。固定 reference policy 也不等于每轮使用完全相同的患者集合或分层份额；角色分配会影响实际 materialization。上表的文字覆盖不声称是音频中的发音次数。

## 结论与下一步建议

1. 小剂量设置下，两轮均没有重现旧的严重退化。可以继续使用这条训练路径，暂不提高学习率或恢复每轮全量 real 重训。
2. 工程闭环确实工作：真实评测 → Reflection → 下一轮新处方；但当前处方还没有稳定超过 real-only，不能将两轮结果写成定向修复机制已成立。
3. maintenance 有局部成功，也有未保持的位置；不能用“maintenance 条数更多”代表全面保护，更不能只看 easy/medium 总 CER 而忽略 hard 和 patient-macro。
4. 下一轮修改优先补齐结构化的实际干预剂量记忆：全批/分角色载体覆盖、真实生成时长、实际 reference 分层份额。目前完整 sidecar 已保存，但历史摘要主要保留角色 copies，缺少统一的跨轮实际剂量表。先让 Agent 看清实际改了什么，再判断怎样改处方。
5. 如果后续要解释角色份额的作用，建议预先固定每轮 24 个 reference 槽位的分层配额，避免角色调整同时改变患者组成。这是待讨论的改动，本实验未中途修改策略。

Real-only shadow 结果用于本次人工诊断，没有伪装成 Therapist ReflectionPack 中原本不存在的证据，也未用作接受/拒绝 Gate。

## 执行问题与验收

- 第一轮首次处方遇到网关 `stream_read_error`；保留失败请求和错误记录，人工监督下以完全相同输入重试一次成功。没有自动 fallback、换模型或手写 Agent 输出。
- CosyVoice 有短目标文本提示；其判断直接比较字符串长度，reference 字符串包含模型要求的 instruction 前缀，不能单凭该提示认定截断。48 条 WAV 均校验可读、非静音、有限采样值及完整时长/hash；未把这些文件检查等同于发音正确性验证。
- 修改范围为小剂量顺序执行入口、无独立 sealed-test 的协议/素材读取、统一 JSONL/CSV 预测读取，以及对应 prompt/测试。未改动 ASR 核心梯度算法或旧实验产物。
- 39 项存储/素材/训练合约测试及 41 项证据/Reflection/角色相关测试通过。
- `training_acceptance.json` 已核验 8 份完整快照、4088 条预测的指标重算、48 条音频、两组数据配对、轮间 replay 不重叠、父 checkpoint 与源文件 hash；原 M0 不变。
- `acceptance.json` 是最终验收入口：两轮真实提交、四次完整 Agent 响应、失败请求与监督重试输入完全相同。四次成功 API 调用累计约 18.0 分钟，271,822 tokens，不包含失败流的未知计费使用量。
- 所有 source manifest、请求/响应、时长、训练日志、checkpoint、结果与 ledger 均保留在执行目录，不覆盖旧 pilot 或上次小型校准结果。
