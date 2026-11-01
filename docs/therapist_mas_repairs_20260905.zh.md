# Therapist MAS：本轮修复及验收

## 范围与结论

本轮执行用户确认的 1、2、5：扎根患者语音识别、修复错误迁移的误判，以及审计发现的确定性实现问题。没有改保护资格、保护数据策略、Reflection 角色归属或 Skill 学习设计；这些留到下一轮讨论。

结果是 Evidence → 处方校验 → materialization → Outcome → Reflection 输入的测量/来源合同更正确，而不是已经训练出了更好的 ASR。没有调用 Agent API、运行 TTS 或 ASR 训练，没有覆盖旧 smoke 和历史 checkpoint。

## 1. 在患者语音上扎根的具体落点

保留 therapist-inspired 的评估—练习—复评思路，但把目标明确为：**根据真实患者的识别错误和训练响应，设计患者条件化的粤语练习内容，让 ASR 更好地识别未见 content。**

本轮落实到代码和 prompt 的内容：

- 练习使用原始粤语发音载体，而不是 CER 等价替换后的字。
- 同一类目标在真实患者语音上逐 GT 位置配对，报告恢复、退化、错误迁移和不确定位置；包含 easy/medium/hard 分层，以及各迁移的患者/content 支持数。
- 患者参考语音固定并核对 train 来源；练习角色与参考难度分层的分配由 Harness 固定，不再由句子排列偶然决定。
- requested-text 的 phone exposure 与生成音频中的真实声学实现严格分开；不把“字里有 /k/”说成“已经成功复现患者的病理 /k/”。

这仍是研究假设与可执行方法定位。当前没有新的性能结果证明闭环胜过静态增强，也没有声学证据证明某个 ASR 错误就是患者的病理构音规则。

## 2. 重点修复：错误换形不再等于恢复

下面四例均以 GT `我想去見客`、before ASR `我想去見卡` 为起点。before CER 都是 20%，原 `coda k→∅` 计数为 1。

| after ASR | CER 前→后 | 原错误计数 | 旧版呈现 | 新版呈现 |
|---|---:|---:|---|---|
| 我想去見 | 20%→20% | 1→0 | improved | 恢复 0；转为整字删除 1；unchanged |
| 我想去見格 | 20%→20% | 1→0 | improved | 恢复 0；转为其他错字 1；unchanged |
| 我想去見客卡 | 20%→20% | 1→0 | improved | 恢复 1，但新增插入 1；inconclusive |
| 我想去見客 | 20%→0% | 1→0 | improved | 恢复 1、无新增插入；improved |

为什么修复后更正确：旧版只看某个错误签名有没有消失。新版固定同一批 GT 载体位置，区分 `correct / signature_error / other_substitution / whole_character_deletion / unresolved`；不能确定的对齐不猜测为恢复。额外插入没有 GT 位置，因此另在含目标的 utterance cohort 中逐条计数，避免遗漏“改对目标字但又多说一个字”的情况。

保留原 `ClusterChange.response` 作为 **exact-signature frequency only**，不修改历史 CER 计分方法。新增 `target_positions.response` 才是目标识别结论，Reflection 的 target finding 也改成读取这个字段。

cluster outcome 的 `met` 现在同时要求原先声明的 count/rate 方向与 paired recovery 条件成立：净恢复、未确认正确的位置减少，以及无新增 unresolved/插入负担。non-increase 也不能掩盖目标位置的净退化。这里是事后观测的判定，不是训练前 Gate，不决定是否接受、回滚或拒绝训练。

插入增长会把目标响应标为 inconclusive；这是一条明确的归因限制，并不意味着每个新增插入必然来自目标，也不抹掉已经发生的局部恢复。`before_error_count/after_error_count` 包含 unresolved，准确含义是“未确认正确的位置数”，不是全部已确认错误。

回归测试还覆盖：一个患者恢复、另一个新错时，不能只报恢复；重复字造成多条最优对齐时，不算作确定恢复。all-optimal GT 对齐与独立穷举小字符串的 oracle 一致。

## 3. 其他修复前后及原因

| 问题 | 修复前 | 修复后与原因 |
|---|---|---|
| 发音身份污染 | 原文 `冇` 归一化为 `無` 后，要求刺激使用 `無` | 字符 cluster 保留 attested surface；接受 `冇`，不强迫变为声调不同的 `無`。CER 仍按原规则计分 |
| 同音 content 隔离 | `咩/孭` 原始 G2P 同为 me1，归一化却使 identity 不同 | Jyutping/detoned identity 从原始发音文本生成，能够发现这类同音重叠 |
| patient-macro 数值持平 | 三患者编辑数 [1,2,3]→[3,2,1]，20%→20% 却因浮点方向不一致而崩溃 | 用同一个精确 Fraction 统计产生方向与 delta；输出 20%→20%、delta=0、unchanged |
| 派生事实矛盾 | metrics、metric_changes、outcome、剂量可互相矛盾，重算 hash 后仍通过 | 从冻结 plan、metrics、sidecar、synthesis receipt 重算并对照；6 类矛盾修改均被拒绝 |
| 处方角色合同 | target 可夹带另一个 cluster；contrast 只需一个 comparator 合法 | target 只能引用 primary cluster；每个 comparator 都需符合现行同-reference 合同。没有改变 protection 资格 |
| 未引用的额外剂量 | Case 01 只计引用，/k/ exposure=28 | 全部句子重算为 32；protection 句中未引用的 /k/ 也计入。共享 /k/ 的多个错误签名不能相加成独立剂量 |
| 角色×患者分配 | Case 02/03 的 protection 8 条落到 easy/medium/hard 为 2/0/6 | 固定分层后为 3/3/2；target 16 条为 6/5/5。改变句子展示顺序不再改变角色分配 |
| reference 来源 | 只检查 speaker 属于 train、文件存在且 hash 匹配 | 精确匹配 train 的 speaker/audio/GT；提供 utt_id 时也核对，避免同患者 dev 音频或错误文本混入 |
| 阶段成功记录 | 任意占位文件可逐阶段登记为成功 | typed/source-bound 检查实际 manifest、WAV、checkpoint、trainer state、prediction CSV 与重编译结果。有效测试路径到 reflected；占位成功路径不再接受 |
| 续训计数 | resume step50 且 max_steps82 可能只新增32步 | 仅加载上轮 adapter weights，optimizer/scheduler/global step 每轮重置，本轮确实新增82步；已提供纯命令构造器，尚未启动训练 |
| 合成失败 | 一条失败会删除之前完成的临时结果，也没有失败记录 | 完整 WAV 保留，写单独 failure.json 后原异常停止；无重试、换 reference、补量或自动续跑 |
| 模型与运行记录 | base fingerprint 只有 safetensors index，部分参数依赖默认值 | 当前 draft 配置绑定完整 ASR 模型目录；训练命令显式 warmup/续训语义，训练元数据记录完整参数和实际包版本 |

`whole_batch_text_carrier_exposures` 是新增的全批文本覆盖统计；原 by-role/by-cluster linkage 剂量仍保留以便追溯。`realized_carrier_exposures` 的历史字段名没有被当成声学证明，结构与 prompt 明确标注 `requested_text_only_not_acoustic_verification`。

## 4. 当前真实数据与新 EvidencePack

重新读取原始发音文本做身份隔离检查：

- Setting D train/dev/test 仍为 2117/258/253 条、48/6/6 unique content。
- raw、CER-normalized、Jyutping、detoned-Jyutping 四种 identity 的 split 两两交集均为 0。
- 冻结参考池仍为 57 位患者，easy/medium/hard 各 19；不是全部 66 位 train 患者。
- 57 条冻结 reference 均精确匹配对应患者的 train 音频及 GT，未发现当前 reference 泄漏。

重新从原有 M0/zero-shot predictions 构建并校验了：

`/data/qwen3-asr/records/therapist_mas_setting_d_v1/evidence/round_001_m0_evidence_pack_v7.json`

972 candidate clusters、50 included、44 同时 targetable/actionable，与旧版数量相同。变化在发音载体与合同正确性，不是通过改 CER 计算制造性能提升。v7 canonical SHA256 为 `572b862efaa5c63bacb310bd33843e6b143d75b3d4bfc8d3c377abc005f80b50`。

完整 ASR 模型目录 fingerprint 为 `01901caa7c806df96e7c7c0520bda09ddcc9a32dc0135ae7859dc1e96181e891`。当前 ASR 环境实际为 torch 2.5.1、transformers 4.57.6、peft 0.19.1、datasets 4.8.4、qwen-asr 0.0.6。

## 5. 验收与未做的工作

最终全仓测试 **152 passed**（21.57 秒）；本轮相关源码/测试 Ruff 检查通过，修改范围内 whitespace 检查通过。

验收使用确定性反例、独立对齐穷举、现有真实文件来源核对和 mock 执行链，不把 mock 的假音频/权重称为真实训练结果。完整测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
  /data/qwen3-asr/venvs/qwen3-asr/bin/pytest -q -p no:cacheprovider
```

重点测试位于 `tests/test_therapist_target_recovery.py`、`test_therapist_surface_grounding.py`、`test_therapist_carrier_dose.py`、`test_therapist_role_contract.py`、`test_therapist_outcome.py`、`test_therapist_store_semantics.py`、`test_therapist_round_training.py` 和 `test_synthesize_manifest_failure_receipt.py`。

尚未把组件包装成真正执行 TTS→ASR→解码的完整 round runner，也没有把 Reflection/Skill/ledger 下一轮回流设计擅自定下来。`committed/completed` 尚无正式语义合同，因此现在明确拒绝用文件登记宣告完成；不是新增 fallback。执行期的完整 decoding 参数收据要在正式 runner 接通时生成，不能用本轮离线验证冒充。

旧结果载荷保留不动；新 RoundResult 使用 schema_version=3，需重建旧 evidence 的下游载荷，不混用旧事实定义。

下一轮只讨论用户保留的两项：怎样设计有意义的 easy/medium 保护，以及 Reflection 的角色与下一轮信息回流。实际 ASR 是否变好，要在这些接口确定后通过真实干预评测回答。
