# MAS 从生成方案到实际训练分配：2026-09-10

当前还没有通过原验收门槛、优于初始 R-only 的新模型。初始模型在固定511条开发反馈上为1384/6656字错（20.793269%），hard为688/1248。以下都是开发实验；不能把反复使用的511条称为独立封存测试集。

## 已完成的机制试验

| 条件 | 总字错 | hard字错 | 主要配对比较 |
|---|---:|---:|---|
| R+MAS，mean-score DPO | 1402 | 709 | 同数据、同曝光的IPO |
| R+MAS，scaled mean-score IPO | 1408 | 719 | 上一行 |
| 32 real，GT CE | 1385 | 698 | 同数据的字符风险+CE |
| 32 real，最小预期字符错误+CE | 1398 | 714 | 上一行 |
| 随机16 real，DPO | 1394 | 701 | MAS headroom选择16 real |
| MAS选择16 real，DPO | 1393 | 702 | 上一行 |
| 真实DPO，仅更新音频塔 | 1398 | 710 | 已有全模块真实DPO1399/hard714 |

各组均未通过原门槛。不能把不同数据、不同score mask、KL或预算的行当成纯loss消融。IPO只测试一个固定beta和四步预算；字符风险使用initial R-only的新beam4，32条真实语音128个实际候选，用字符编辑次数作为成本、原始序列概率作为分布，并以0.01倍GT token-sum CE辅助。它们的负结果不等于这些损失在所有配置均无效。

音频塔消融严格复用原始32真实音频/91实际偏好对、256次KL、4步、edit+EOS DPO beta0.1、lr1e-4、KL10。144个音频LoRA张量可训练，392个文本LoRA张量及底座均核对不变。MAS反思据此停止相邻冻结比例、学习率和loss扫描。

## 新假设及 MAS 的具体决定

整批TTS没有带来收益，仍可能包含作用不同的样本。新probe逐条测量既有22条SG音频、52个实际训练偏好对的DPO梯度。每条音频内平均其错误对，不按错误数量增加权重；保持原edit+EOS beta0.1和共同initial R-only，无optimizer更新。

查询侧来自已经固定的64条真实训练GT pool，与原真实DPO、KL、reference catalog及511开发音频身份和哈希隔离。A/B各8条用于MAS决策，随后16条C不进入MAS请求，仅在选择冻结后作训练域迁移诊断。GT anchor使用每条音频完整GT+EOS的平均token NLL。

训练梯度与GT anchor梯度内积为正，仅表示在足够小的负梯度步下一阶GT loss下降；它不证明Adam四步更新有效，不证明CER下降，也不证明合成音频逐字正确。完整probe150.2573秒，无模型更新；22条中6条对A/B均为正。

实际MAS看到了目标文本、reference录音卡片、实际rejected、已有三文本审查及A/B梯度。它选择8条：market.c0005、market.c0007、clothing.c0003、clothing.c0004、driving.c0007、food.c0003、evening.c0004、family.c0001。完整音频ID和逐条理由保留在原始decision.json，不用这里的简称定位数据。

MAS选择覆盖6个目标文本、8段reference录音、7名说话人，含target/contrast/protection/maintenance四类。它允许单条负向梯度由组合中的其他样本补偿，以保留内容覆盖；没有重写任何SG粗审标签或chosen。其选择同时考虑了审查文本与目标的支持程度，因此本试验评估整个选样策略，不能单独声称是reference质量或梯度排序的因果效果。

| 八条合成梯度的均值内积 | A（选择） | B（选择） | C（选择后诊断） |
|---|---:|---:|---:|
| 随机对照 | +0.00159580 | -0.00179681 | -0.00015198 |
| MAS | +0.00075111 | +0.00013948 | -0.00063456 |

C没有支持同向迁移。选择及训练计划已冻结，不根据C重选，也不把A/B正值当作效果承诺。

## 配对 DPO 执行

两臂各8条不同的既有SG音频，每条4次，共32合成曝光、每步合成权重0.25。随机对照在观测梯度前固定（PCG64 seed42，单个目标文本最多两条）；MAS遵守相同文本上限。真实32音频/91偏好对及256次KL的次序和权重逐项一致。数据成员变化是干预，不能说合成曝光身份相同。

两个训练臂均从相同initial R-only残差初始化，DPO reference始终为冻结initial R-only ASR。复用有模型及文件哈希绑定的实际reference logp；KL在函数等同initial R-only的模型上、任何optimizer step前计算并detach，逐项核对旧teacher摘要。SG reference speech是另一个概念，独立记录。

新训练入口的GPU smoke已通过：真实和合成各一个实际偏好对均有有限非零梯度，旧reference分数及KL精确一致，前8条初始评估预测一致，无optimizer step，39.8049秒。正式两臂使用两张GPU；训练目标、三文本粗审、真实GT chosen及合成粗审后intended chosen均保持原样。

原验收仍要求pooled严格改善、patient macro不超过initial+0.001、hard pooled不超过initial+0.005。MAS增量也用相同方向与容差相对随机对照检查。未通过的本轮权重不保存，完整预测、来源和训练日志保留；通过的adapter以原生safetensors写入/data并逐tensor核对。无自动晋级。

## 入口与产物

- `examples/mas_synthetic_audio_utility_probe_v1.py`：无更新逐音频梯度与A/B/C测量。
- `examples/reflect_mas_synthetic_audio_utility_v1.py`：实际MAS调用，明确不向请求提供C。
- `examples/mas_synthetic_utility_allocation_dpo_v1.py`：smoke、随机及MAS两个配对DPO臂。
- `examples/report_mas_synthetic_utility_allocation_v1.py`：原门槛与增量评估、共同起点和实际曝光核对。
- `/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1/synthetic_audio_utility_probe_v1`：probe、原始MAS请求/响应/决定。
- `/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1/synthetic_utility_allocation_DPO_v1`：冻结计划、smoke、正式训练与评估。

这些是复用现有框架和冻结历史产物的实验入口，仍依赖已有/tmp运行助手及数据来源；不是声称已打包为独立可移植训练器。

梯度选样有相关方法基础：[LESS（ICML 2024）](https://proceedings.mlr.press/v235/xia24c.html)使用优化器相关的影响估计选择指令训练数据。此处只是一个初始点上的音频DPO梯度probe，没有复现LESS的Adam影响轨迹、warmup及完整方法，也不能套用其文本任务收益。

## 正式结果及下一项验证

配对训练和511评估均已正常结束。随机8SG为1382字错（20.763221%）、hard694，通过原门槛；MAS8SG为1409字错（21.168870%）、hard713，未通过。随机组仅比initial少2字错（-0.030048个百分点），不能据此声称稳定收益。MAS相对随机多27字错（+0.405649个百分点），不支持这次MAS效用选样策略。

随机组原生adapter已保存于本轮uniform/checkpoint-4，完整目录SHA256为5879fd59602f88747d081c89ebb4dc1921097917f2c5682bb58358e2e989fad7；独立重新加载后，全部残差/底座参数摘要及前8条预测精确一致，29.2766秒。无晋级；MAS失败权重按预声明策略未保存。

下一步仅复现这个小幅正向的固定随机子集：优化初始化seed43/44各跑真实DPO与真实+同8条SG的配对对照。reference、数据、候选采样和选样都维持seed42来源，真实/KL顺序和所有超参数保持不变；这不是三个完整pipeline随机种子。新入口examples/mas_synthetic_fixed_subset_replication_v1.py，输出B/synthetic_utility_fixed_subset_replication_v1。复现正向随机对照不意味着继续推广已经失败的MAS梯度选样规则。

追加目标位置核对：existing EvidencePack方法在新结果上重算，输出本轮real_target_recovery.json。随机组initial→uniform的i→yu文字转写signature为67→70；所有GT i-carrier位置错误363→366，确定恢复2处、退化4处，未决90→91。MAS组signature67→64，但所有i-carrier错误363→364、恢复2/退化2、未决90→95。两组都没有证明原定目标位置纠错成功；不能把随机组总字错少2或MAS signature少3解释成声学/临床改善。

## 三个优化seed已完成

| seed | R DPO字错 | R+固定SG字错 | R hard | R+SG hard |
|---|---:|---:|---:|---:|
|42|1399|1382|714|694|
|43|1400|1390|708|707|
|44|1410|1410|718|716|
|平均|1403|1394|713.33|705.67|

R+SG平均比R少9错，但比initial多10错（+0.150240pp），只1/3优化seed过原门槛；未形成稳定改善。四个新复现实验全部正常结束，失败权重不保存。单个42候选仍已持久保存、无晋级。

内容诊断在复现ROOT/content_transfer：训练2117语音/48个规范化句子，开发511/12句，整句重合0。随机42对initial的患者×内容诊断bootstrap95%区间[-0.412805,+0.357514]pp；三seed均值为[-0.239254,+0.568152]pp。此诊断不改原门槛、不排除任何数据。

实际MAS反思已完成（复现ROOT/MAS_reflection/decision.json）：选择最后一个固定子集消融，删除91次real-DPO pair曝光，仅SG-DPO+原256次realKL；所有SG曝光、mask和超参数保持不变。既要过original gate，又需低于完整R+SG42的1382错，才支持real-DPO masking假设。若失败则关闭固定子集，转新matched SG probe，不再扫相邻loss/lr/freeze/ranking/quota。新入口examples/mas_fixed_subset_s_dpo_real_kl_v1.py，当前smoke。

固定8子集最终消融完成：SG-DPO+realKL为1382字错、hard695、vlink247392/others990；未过原门槛，也未低于完整R+SG42的1382。否定本配方的real-DPO masking假设，关闭固定子集。新24音频探针另存B/literal_matched_SG24_v1：实际MAS根据当前字级替换、训练reference事实和历史生成记录设计3对目标/反向正确用字句子，选择2患者各2reference。固定instruct2+cantonese_zh、同case生成/ASR种子；A为预先偏好reference、B为同患者比较reference。先各1条smoke、核验复用后完成24条，再作实际偏好对交付报告。新入口examples/mas_literal_matched_sg_probe_v1.py；旧冻结源码与账本不改。
