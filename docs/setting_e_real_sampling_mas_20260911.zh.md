# Setting E：MAS 调度真实训练数据

2026-09-11 完成状态：普通与 MAS 两组均已完成 590 steps。
相同更新预算下 test 总 CER **15.7368% → 15.4813%**。
本轮方法已独立留存；成功总结、21 轮 Agent 决策解释与保留位置见
`docs/setting_e_mas_success_preserved_v1_20260911.zh.md`。
本文其余部分保留实现与恢复过程的历史记录。

## 这次检验什么

在原始 Qwen3-ASR-1.7B 上重新建立同 seed42 的 Audio LoRA，比较普通随机遍历与 MAS 调整采样比例。两组都用全部 1862 条真实录音；没有 TTS，没有 DPO，没有从旧 R-only adapter 继续训练。

按用户最新要求，恢复原始 Setting E 的 train/dev/test 三份划分：

| 用途 | 真实录音 | 患者 | 文本 |
|---|---:|---:|---:|
| train：梯度训练、可练习内容目录 | 1862 | 46 | 60 |
| dev：全部 MAS 诊断、策略反馈、反思和经验 | 394 | 10 | 60 |
| test：程序报告比较结果 | 372 | 10 | 60 |

三份患者、录音 ID 和音频哈希互不重叠，文本覆盖相同。沿用原始划分文件；不重新洗牌或根据新结果选患者。原来合并的 766 条恢复为 dev394＋test372。test 转写、成绩和患者资料不进入 Agent、经验或下一阶段决策。训练池只提供静态练习内容、既有分组和实际曝光，不提供训练转写、损失或识别成绩。历史上这 372 条 test 已被观察过，不称新盲测。

本版本为 `sampling_curriculum_dev_phonology_20260911_v2`。旧 training-diagnostics 版本的协议、源码快照、smoke 和账本保留；旧版准备完成的正式比较从未启动，本版本不读取旧经验。

## 角色与五个能力

保留原 Therapist 的提案和反思语义，补全原设计里尚未执行的 Skill Optimizer。实现为两个逻辑 LLM 角色、三个调用阶段，不宣称历史上已有独立的诊断、提案、裁判委员会。

1. 程序计算 dev 当前错误、各组前后变化、患者和内容分布，提供具体转写例子。
2. Therapist 诊断错误、找到对应的固定训练组、分配下一阶段的采样概率。
3. 固定训练器执行，保存实际录音、患者、数据组的训练曝光。
4. 同一个 Therapist 只根据 dev 前后结果进行反思。
5. Skill Optimizer 整理经验文字，供下一次 Therapist 参考；不修改提示词规则或训练参数。

所有 LLM 调用固定为 laicai `https://api.laicaishare.xyz` 下的 `gpt-5.6-sol`，沿用 `xhigh`；核对实际返回的 model 字段。使用已有凭据，密钥只进入进程环境，不写入实验文件。提示词用中文完整说明任务、数据来源、数字单位和字段含义；每次调用显式带历史，不能假定模型记得聊天。

请求使用现有 Responses API 与严格结构化输出。格式参考 [OpenAI 官方文档](https://developers.openai.com/api/docs/guides/structured-outputs)，laicai 的实际兼容性另外通过真实请求验证。

## 唯一训练操作

训练前固定六组，每条录音恰好属于一组。长度按本次固定分组规则，12 字及以下为较短句；历史 easy/medium/hard 只表示原始模型识别难度，不表示临床严重程度。

| 固定组 | 训练录音 |
|---|---:|
| 较容易、较短句 | 405 |
| 较容易、较长句 | 556 |
| 中等困难、较短句 | 233 |
| 中等困难、较长句 | 303 |
| 较困难、较短句 | 194 |
| 较困难、较长句 | 171 |

MAS 每次只输出六个概率，总和为 1；先按概率选组，再在组内等概率选录音。没有另一个固定覆盖混合系数。采用有放回抽样，因此一阶段可能重复某些录音、暂时未见另一些录音。普通对照持续无放回随机遍历，跨阶段保持遍历进度。

六组是本轮有限动作空间；Agent 对具体句子或患者的诊断只能映射到这些组，不能单独指定那一句或那位患者。这是首轮实现的能力边界。

## 语音学分析如何参与决策

复用 `phonology.py` 和 PyCantonese 5.0.0，将目标文本与 dev 转写中对齐明确的错字拆为声母、韵腹、韵尾和声调差异。文字转读音依赖词典及语境，不是从患者音频测量发音；不把此统计命名为实测 phone error rate。分解方法见 [PyCantonese 文档](https://docs.pycantonese.org/stable/jyutping.html)，类别解释依据 [香港语言学学会粤拼方案](https://jyutping.org/en/jyutping/)。

- 保留读音所需的原始字面形式，再与项目 CER 归一化对齐；归一化已接受的写法差异不计入语音混淆。
- 对每个混淆提供 dev 出现次数、目标成分总出现次数、涉及患者数、具体字词所在句子、前后变化。最多显示 12 个常见成分混淆，按前后两端较大的次数排序，避免改善到零的错误消失。
- 同音错字、整字漏字或多字、对齐不明、词典未收录另行说明；多个成分次数不能加起来当 CER。字典覆盖率随材料保存。
- 将错误的目标成分映射到真实 train 句子，列出录音数、患者覆盖、六组各自的覆盖比例。另提供 dev 每个目标句子的成绩、粤拼及其对应训练材料分布。
- MAS 可据此提出跨句子、跨患者的关联假设，并依据后续 dev 变化修正。若各组都含有相同比例的目标成分，现有动作无法精确加练该成分，提示词要求直接说明，不能宣称已经实施精确音素课程。

所有 Agent 提案、反思和经验输入都使用 dev 语义和明确中文术语；测试反馈及训练诊断输入在入口拒绝。每次阶段调用显式携带当前 dev 证据、上一阶段结果及最近经验。

## 固定训练与比较协议

- seed42，原始模型＋全新 LoRA；r16、alpha4、dropout0.05，只更新 Audio LoRA。
- 原有逐录音平均交叉熵；AdamW、LR2e-4、warmup5、线性学习率；全局 batch32、microbatch4。
- 当前正式训练共 21 阶段：前 20 阶段各 29 步，最后 10 步，共 590 步／18880 次曝光，约为训练池 10.14 次遍历的预算。MAS 阶段不叫完整 epoch。
- 两分支各使用 2 张 L20，均完整运行 590 步。按用户最新预算取消旧 1800 秒提前停止规则；实际墙钟与 GPU 分配时间照常统计。
- 初始和每阶段结束，对 dev 和 test 用同一 greedy Cantonese、空上下文、cap256 评测；合并 LoRA 后恢复基础权重原值。
- 优化器、学习率进度和训练随机数状态跨阶段连续。数据采样使用独立随机数发生器，Agent 耗时不改变 dropout 随机数。
- 检查点候选保留起点；只按 dev 患者等权 CER 选择改善检查点。同分保留较早候选。
- 主报告列固定步数结果和共同用时下最新已完成的评估，不能按 test 最优点挑选。患者等权 CER 和总体 CER 同时列出。
- 同步数不等于同计算时间：计时单列加载、训练、评估、Agent/等待、checkpoint 和整轮墙钟，GPU 分配秒数为卡数乘墙钟秒数。预算外收尾评估单独标记，不计入预算内成绩。

训练阶段结束才评测，因此时间对比的分辨率是评估间隔；报告必须保留取用 checkpoint 的真实步数和完成时间，不插值成绩。smoke 若与旧任务共享 GPU，只证明正确性，不作为吞吐或性能比较。

## 入口与记录

```bash
cd /data/qwen3-asr/repo/dysarthric_yue_Speech
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/prepare_setting_e_sampling_curriculum_20260911.py prepare
/data/qwen3-asr/venvs/qwen3-asr/bin/python examples/prepare_setting_e_sampling_curriculum_20260911.py api-smoke
```

每次执行使用新 root。准备产物、API 请求与实际响应、采样方案、真实曝光、反思、经验、优化器状态和源码快照分别记录，不覆盖旧实验。正式训练前需要为两分支释放独立 GPU；旧四卡增强实验已按用户指令停止，停机产物保留；本次只运行新版本 smoke，正式比较尚未开始。

训练入口 `examples/run_setting_e_sampling_curriculum_20260911.py` 提供 `prepare`、`cpu-smoke`、`smoke`、`supervise` 和 `report`。当前 GPU smoke 使用独立 `smoke_content_overlap_v2` root，两阶段各两步；每份 16 条且共享相同 4 个文本。包含真实梯度检查和真实 Agent 调用，不拿 smoke 的 CER 代表课程效果。

## 第一次 dev 版本验证（smoke_v1，历史记录）

33 项相关测试通过。完整 dev394 的真实 API 接入通过；两卡普通采样与两卡 MAS 并行 smoke 均完成两次更新。实际缩小为 train/dev/test 各 16 条，分别覆盖 13/7/7 位患者；缩小子集仅共享 1 个文本，因此只用于代码与来源检查，不代表完整 Setting E 的内容覆盖或性能。完整协议仍要求三份都覆盖 60 个文本。

闭环 6 次真实调用均为 laicai gpt-5.6-sol。审计从 dev 预测与实际训练曝光重建每次提案、反思输入并核对哈希；第二轮带入第一轮经验。两分支初始化和源码一致、梯度检查通过、冻结参数保持不变。普通 smoke 用时 90.1 秒，MAS 489.9 秒，其中 Agent/等待 398.4 秒；这是功能检查耗时，不用于推断正式训练吞吐。GPU 已释放，正式对照尚未启动。

验证记录：`/data/qwen3-asr/records/therapist_mas_setting_e_v1/sampling_curriculum_dev_phonology_20260911_v2/readiness.json`。

## 用户要求复跑：smoke_content_overlap_v2

本次两分支同时占用两张 L20，均通过。train/dev/test 各 16 条，分别覆盖 13/9/7 位互不重叠的患者，三份都覆盖相同 4 个文本；每个文本在每份数据里由 4 位不同患者提供。两阶段各 2 步，共 4 步／128 次曝光，第一阶段含非零学习率的实际更新。

普通采样用时 97.0 秒；MAS 用时 549.3 秒（约 9.2 分钟），其中 Agent/等待 452.1 秒。两分支纯训练均约 8.2 秒。六次闭环调用均为指定 laicai gpt-5.6-sol；第二轮收到第一轮经验。审计重建全部提案和反思的 dev 输入，验证哈希、共同初始化、梯度、冻结参数、按 dev 验收检查点和实际曝光。相关 runner 测试 4 项通过。四卡已释放，正式对照尚未启动。

结果：`/data/qwen3-asr/records/therapist_mas_setting_e_v1/sampling_curriculum_dev_phonology_20260911_v2/smoke_content_overlap_v2/smoke_summary.json`。

本次只改变 smoke 取样与阶段长度。正式数据、MAS 提示词、训练设置保持原样；由于 runner 源码哈希变化，正式启动前应在新执行目录重新冻结源码，保留旧准备记录，不覆写旧账本。

## 正式运行：每 29 步调整，总计 590 步

用户已授权并启动正式对照。执行目录：`/data/qwen3-asr/records/therapist_mas_setting_e_v1/sampling_curriculum_dev_every29_total590_seed42_20260911_v4`。GPU0/1 为普通随机遍历，GPU2/3 为 MAS 调度；两组均原始 Qwen3-ASR＋全新 Audio LoRA、seed42、LR2e-4、global batch32。

MAS 在 step0、29、58……580 决定下一阶段比例；两组在 step0、29、58……580、590 完整评估 dev394 和 test372。所有 MAS 反馈和反思只来自 dev。最后阶段只有10步，优化器和学习率总长度严格为590，跨阶段连续；不再按旧30分钟上限提前停止。实际耗时与等步数结果分别报告。

10项相关调度/采样/连续状态检查通过，包含29×20＋10＝590、18880次曝光、尾阶段与学习率总进度。沿用最新GPU smoke已验证的前向、反向、冻结参数、MAS代码和提示词；源码迁移差异保存在新执行目录。全部2628个真实音频哈希复核通过。首次完整dev/test预测及新LoRA初始化在两分支间完全一致。

监督：`tmux attach -t setting-e-mas-29`。monitor窗口显示两组loss、更新步数、dev/test CER及Agent阶段；train窗口运行持久监督进程，完成后自动生成comparison.json。

第一次正式启动 v3 在初次 laicai 请求中发生 `APIError: stream_read_error`，MAS 未更新，普通分支在第29步评估期间被旧联合监督逻辑终止。记录保留在 v3，未复用其训练结果。当前 v4 两组都重新从相同初始化开始；增加仅针对该流式传输错误的最多3次相同请求尝试，逐次保存回执，其他错误不自动重试。两分支独立运行，MAS的API失败不再连带结束普通分支。26项相关检查通过；`recovery.json`保存失败来源、迁移和两次执行的区别。


## v5 超时恢复（2026-09-11）

v4 普通组已完成590步；MAS在232步完成评测后，反思请求发生ReadTimeout，完整checkpoint只保存到203步。用户要求接续完成，并在等待超过10分钟时自动重试。v4的代码快照、账本及全部产物保持原样。

v5执行目录为`/data/qwen3-asr/records/therapist_mas_setting_e_v1/sampling_curriculum_dev_every29_total590_seed42_20260911_v5`。普通组通过来源链接复用v4完成结果；MAS从203步恢复adapter、AdamW、590步学习率进度、各rank随机状态及独立采样器。第8轮复用已验证的原始决策，核对928次采样顺序，并在232步比较重跑loss与dev/test转写。原有评估前缀的来源和哈希存于`resume_import.json`。

每次MAS API尝试有600秒总墙钟上限（包括持续心跳的流）。ReadTimeout、SDK超时、总时限及stream_read_error使用相同请求重试，总计最多3次尝试、间隔2秒；请求内容、模型和推理强度不变。每次开始及结果都有独立记录；无效模型输出仍直接报错。NCCL等待上限为90分钟，允许rank0完成反思、经验的有限重试。

新checkpoint在反思前保存，除训练状态外还保存dev反馈和待完成反思内容，避免API故障丢失训练进度。监督器退出会保存failure.json，tmux明确显示STOPPED；请求进行时显示尝试次数与600秒计时。累计执行耗时包括v4失败等待及重放开销，不隐藏恢复成本；无人占卡的离线修复间隔另留来源时间戳。

31项相关自动检查通过，覆盖总时限中断、相同请求重试、连续优化器/学习率/采样/dropout状态，以及Setting E数据与590步预算。实际GPU恢复和232步重放检查结果写入正式执行日志。监督命令仍为`tmux attach -t setting-e-mas-29`。


当前执行迁移至`sampling_curriculum_dev_every29_total590_seed42_20260911_v7`。v5在新更新前发现PEFT加载器与Transformers的EmbeddingParallel导入不兼容；v6改为直接恢复现有LoRA参数存储，权重、优化器、调度、采样及各rank随机状态逐项一致，前3步loss与梯度范数完全一致，随后反向计算出现数值差异（29步最大loss差0.00670588）。v6被原先过严的逐位重放检查停止，未进行新MAS决策。两次恢复的开销与日志都保留。

v7验证恢复状态、第一步更新、完整学习率和928次样本顺序；后续loss差异和dev/test转写差异全部记录，不以CER好坏决定是否接续，也不声称跨进程逐位复现。重新测得的dev结果用于下一轮反思，test仅报告。32项自动检查通过。当前tmux的monitor及resume-v7窗口指向v7；v4普通组590步结果仍直接复用。


## v8 服务端断开连接后的接续

v7完成377步训练、dev/test评估和完整checkpoint之后，第13轮反思在506秒时收到服务端断开连接（`httpx.RemoteProtocolError: Server disconnected without sending a response`），SDK包装为`APIConnectionError`。没有达到600秒时限；v7的超时分支未覆盖此连接错误，任务退出。

v8仅将SDK连接错误、HTTP网络错误及远端协议断开纳入相同请求的有限重试。600秒总时限、最多3次尝试和2秒间隔不变；输出验证标准、提示词、模型及训练配置不变。37项相关检查通过，377步checkpoint与所有历史来源哈希核对通过，待恢复反思请求与原失败请求完全相同。

新的执行目录为`/data/qwen3-asr/records/therapist_mas_setting_e_v1/sampling_curriculum_dev_every29_total590_seed42_20260911_v8`；从377步和待完成反思恢复，之后训练378～590，不重跑任何已完成训练步。v7产物保持原样，普通组仍复用v4的590步结果；累计用时包含此前失败等待。tmux仍为`setting-e-mas-29`，运行窗口为`resume-v8`。


## v9 临时网络中断恢复

v8完成406步训练、评估及完整checkpoint后，第14轮反思连续3次失败：第一次600.002秒读超时并自动重试；第二、三次各约40秒，底层为`Network is unreachable`。达到已设定的3次上限后监督器退出。此次超时与连接重试均按预期工作。

2026-09-11 18:22 UTC，对相同laicai/gpt-5.6-sol的短请求检查4.03秒正常返回。v9使用未改动的v8源码，从406步及相同待完成反思接续，不重复训练；旧记录保留，累计时间包含失败请求。运行目录为`sampling_curriculum_dev_every29_total590_seed42_20260911_v9`，tmux的monitor与resume-v9指向该执行。
