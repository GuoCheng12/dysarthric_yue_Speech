MAS 目标探索：IPO 与字符风险训练（2026-09-10）

这两项实验承接用户追加的顺序：收尾梯度路由 probe，然后比较 IPO/DPO，再测试字符级最小风险。梯度 probe 没有找到同时支持两个独立真实训练 anchor 集的合成模块，故没有启动该路由训练。原有 MAS→TTS→DPO 仍为主线。

IPO/DPO 对照

- 两组均复用既有 R＋MAS broad curriculum：真实音频曝光 32 次、合成曝光 32 次（22 条音频、52 个独特偏好对），共同真实 KL 曝光 256 次。不重新生成音频或采样，不改变 SG 粗审。
- 共同初始 R-only、seed42、4 步、lr=1e-4、r16/alpha4、KL 权重10、合成权重0.25。逐项曝光顺序和来源回执保持一致。
- 两组均对完整转写加 EOS 取平均 token log probability，beta=1。与旧 edit+EOS summed-score DPO 的比较不能单独归因于 IPO；本次新配对才隔离偏好损失的差异。
- 设 h 为 chosen/rejected 的 policy/reference 平均 log-ratio 差。DPO 为 `-logsigmoid(beta*h)`；IPO 为 `0.5*(beta*h-0.5)^2`。这是标准平方 IPO 的明确正比例缩放，保持有限最优间隔，并使两种损失在 h=0 的导数相同。使用同一个 KL 系数时，不能忽略这个缩放选择。
- 实际 GPU smoke 已核对真实和合成各一对：初始 policy/reference 分数一致、两种损失的完整梯度逐位一致、参数未更新；初始模型 8 条评估转写与原 R-only 一致。
- 这两组都是混合数据。若出现收益，还需要相应的真实数据单独训练对照，才能论证 MAS 的增量价值。

字符风险对照

- 首个小实验复用既有真实 DPO 曝光中的 32 条真实音频及首次出现顺序。它是历史错误较多的样本池，不代表随机抽取全部 2117 条；也未使用新的开发结果挑选。
- 对每条音频用共同初始 R-only 做 beam-4，长度惩罚0、空上下文、粤语、最多256个新 token。不把 GT 插入候选。保留四个原始结果、EOS/截断信息；风险集合按实际 token 序列去重，排除没有结束的候选，真实 GT CE 仍保留。
- 计分使用实际生成的 token，包括首个 EOS；不对解码后的文本重新分词来替代生成路径。候选概率用当前 policy 的完整序列 log probability 做 softmax，梯度穿过 softmax。
- 成本使用与评估相同的文本归一化后字符编辑**次数**，对应 MWER 的字符级改写；不在每句话内除以 GT 长度。固定训练集合中，总字错与 pooled CER 只有固定分母的差别。
- 风险为 `sum(q_i * (edits_i - mean(edits)))`。相同成本或只有一个有效候选时，风险梯度为零；没有有效候选时只保留真实 GT CE。不制造负例。
- 对照为 `0.01 * GT summed-token CE` 与 `risk + 0.01 * GT summed-token CE`；两组同样32条音频、4次更新、共同初始模型与优化器设置。
- 第一轮只生成一次 parent N-best，然后每条音频训练一次；当前 policy 会重新计算候选概率。这是固定候选集的小规模 MWER 近似，不宣称实现了每次更新重做 beam search。
- beam-4 的两条真实音频 smoke 与实际 GPU 风险反传 smoke 均通过：低成本候选的概率梯度方向正确，参数未更新。完整32条有128个独特有效候选，无截断，30条有成本差异，22条没有正确候选。

评估、存储与下一步

两组均评估完整511条真实开发反馈，报告 pooled CER、patient macro、hard、vlink247 与其他患者。原验收条件不改变：pooled 严格改善，patient macro 增量不超过0.001，hard pooled 增量不超过0.005。511条是反复使用的开发反馈，不是独立测试集。

新 pilot 在训练进程内完成最终模型评估。明确不保存未通过原验收的新失败候选权重，保存完整预测、曝光、配置、源码哈希和参数摘要；通过的候选尝试无损保存原始 safetensors，若部分文件存于 `/dev` tmpfs，必须标明易失，不能当成持久归档或晋级。初始 R-only 与旧实验产物不删除。之前已退出的十个失败候选内存权重无法恢复，此事实另有记录。

若风险目标显示收益，下一步让 MAS 在已有训练语音中选择练习对象、错误覆盖及曝光预算，并与相同预算的固定选择对照；字符错误仅作为训练成本，不改变 SG 粗审。

四组完整结果

| 条件 | 字错 | CER % | hard 字错 | vlink247 字错 | 其他字错 |
|---|---:|---:|---:|---:|---:|
| 初始 R-only | 1384 | 20.793269 | 688 | 385 | 999 |
| R＋MAS / DPO（mean） | 1402 | 21.063702 | 709 | 396 | 1006 |
| R＋MAS / IPO（mean） | 1408 | 21.153846 | 719 | 407 | 1001 |
| 32 real / CE | 1385 | 20.808293 | 698 | 395 | 990 |
| 32 real / 字符风险＋CE | 1398 | 21.003606 | 714 | 399 | 999 |

四组均未通过原验收，没有晋级。前两组的真实/合成/KL曝光相同；后两组的真实曝光和CE系数相同。两对之间数据与KL配置不同，不能把跨对差异单独归于训练目标。四组实际初始可训练参数摘要及冻结合并底座摘要均相同。

IPO相对其DPO对照变化+0.090144个百分点，患者bootstrap 95%区间[-0.255496,+0.387620]；字符风险相对其CE对照变化+0.195312个百分点，区间[-0.059367,+0.451519]。单次、小预算结果未显示收益，不是对两种损失全部配置的排除。

IPO/DPO在全部511条评估完成后，最终回执遇到 NumPy bool 的JSON序列化错误。完整预测、指标、比较和代码哈希验证均已保留；用明确记录原进程退出码1和失败阶段的完成回执恢复收尾，没有重跑或更改结果。字符风险使用新v2执行版本，仅对最终布尔值显式转为Python bool；v1源码和采样/反传smoke保留，v2以来源记录和链接复用。v2两组正常完成。

代码入口：`examples/mas_sequence_objective_pilot_v1.py`（冻结的已完成实验）、`examples/mas_character_risk_pilot_v2.py`。损失与数值测试：`src/therapist_harness/asr_sequence_objectives.py`、`tests/test_asr_sequence_objectives.py`。运行记录：`/tmp/mas_IPO_DPO_mean_v1`、`/tmp/mas_character_risk_v2`；原始风险采样与smoke在`/tmp/mas_character_risk_v1`。

研究依据：[IPO 原论文](https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html)、[TRL 的 IPO 实现](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)、[MWER 原论文](https://arxiv.org/abs/1712.01818)。这些研究支持尝试目标的理由，并不保证在本任务上产生收益。
