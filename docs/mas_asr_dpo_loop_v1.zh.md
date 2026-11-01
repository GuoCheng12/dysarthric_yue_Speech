# MAS 面向 ASR DPO 的数据闭环 v1

2026-09-08 网络重试修订：新 `prepare` 的 MAS／Reflection 请求冻结 `agent.max_retries=4`，文字粗审同样使用 SDK 的 4 次重试（总计最多 5 次发送）。单次建连超时 20 秒，MAS 读取超时仍为 900 秒，粗审读取超时仍为 180 秒。仅重试 SDK 判定的临时连接／超时及可重试 HTTP 状态；不重跑 TTS，不重试审查结论或解析验证失败。MAS 已开始接收的流若中途断开，仍会显式报错，不自动重新请求。失败历史目录保留，重跑使用 `mas_tts_only_dpo_seed42_20260908_retry_v2`。

2026-09-08 用户进一步确认先运行 **R-only 起点＋仅 MAS TTS 偏好对的 DPO**。`prepare --synthetic-only` 冻结单一 `mas_dpo` 分支、零真实候选与零真实曝光，使用常驻审核 ASR。2117 条真实训练池仍用于 reference 选择和隔离校验，511 条真实反馈用于评测；两者均不进入本次 DPO 偏好采样。纯合成目标为平均 DPO 损失，合成权重为 1。此设置应称为 synthetic-only DPO，不能称为 R＋S 混合偏好训练。首轮运行目录为 `/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_tts_only_dpo_seed42_20260908_v1`，计划 96 条音频、每条 4 次采样、12 步、每步 8 条合成曝光。

实现入口是 [mas_asr_dpo_loop.py](/data/qwen3-asr/repo/dysarthric_yue_Speech/examples/mas_asr_dpo_loop.py)。流程复用现有 EvidencePack、Proposal、SG、真实语音评测、Reflection 和 RunStore 账本：

> 真实错误反馈 → MAS 新 proposal → SG → 固定审核 ASR 转写＋轻量文字 Agent → 当前 ASR 采样 → 偏好对 → DPO → 真实评估与反思 → 下一轮 proposal。

**2026-09-08 状态：原四分支在线 smoke 已通过。正式 96 条、两个 DPO 分支任务于 12:36 UTC 启动，并按用户要求于 13:25 UTC 停止在第 50 条生成＋审查完成后；未开始正式 DPO 更新。旧账本、50 条音频及原始转写／审查、源码快照保留在 [旧运行目录](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo96_seed42_20260908_v1/README.zh.md)。常驻审核 ASR 与双卡执行的验证记录位于 [执行优化验证目录](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_execution_v2_validation_20260908_v1)。**

完整结果与检查范围见 [本轮报告](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_transcript_smoke_20260908_v1/report.zh.md) 与 [完成回执](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_transcript_smoke_20260908_v1/completion.json)。smoke 验证实现与数据闭环，不作为方法增益结论。

历史 MAS 凭据已恢复，`gpt-5.6-sol` 可正常调用。旧直接音频模式仍可显式选择，但当前网关的音频模型请求返回 HTTP 400、Realtime 返回 HTTP 404，未验证实际听音能力；这不再阻碍默认转写审查路径。相关检查见 [API 核验记录](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_api_recovery_20260908_v1/report.zh.md)。

## 干预与对照

MAS 根据当前真实错误设计 target、contrast、protection、maintenance 内容。contrast 是有效的反向练习内容，不是故意错误的 chosen。审查、采样和训练参数由 Harness 固定；MAS 在下一轮读取真实效果与实际交付，调整练习处方。

Proposal 沿用现有完整句子及句尾标点约束；同一目标从 SG、粗审到 chosen 保持原文。仅标点不同不会被判为 ASR 错误或构造偏好对，但标点仍参与实际序列损失。此次实现 smoke 不代表已统一真实与合成标签的标点格式。

每轮四个分支从同一个当前 MAS-DPO parent 开始；第一轮共同 parent 为 seed42 R-only。下一轮只推进 MAS-DPO 的固定末端。这是每轮同 parent 的局部对照，不是四条各自独立累积的训练轨迹。

2026-09-08 用户指定后续只运行两个 DPO 分支。新运行可在 `prepare` 时加 `--dpo-only`，将 `comparison=dpo_only` 写入冻结协议；后续 `run` 只构造、训练和评测 `r_dpo`、`mas_dpo`。默认仍支持原四组配置。两分支沿用相同的逐轮 parent 规则，冻结 DPO reference 始终不变。

| 分支 | 训练材料 | 目标 |
|---|---|---|
| `r_sft` | 所选真实语音全部标签 | SFT |
| `r_dpo` | 真实语音的实际错误采样偏好对 | DPO |
| `mas_sft` | 真实语音＋粗审通过的 MAS 合成音频 | SFT |
| `mas_dpo` | 真实语音＋粗审通过的 MAS 合成音频所产生的偏好对 | DPO |

同一目标下 R 与 R＋S 使用独立于 S 的真实采样流，保持真实曝光顺序一致。SFT 保留全部已准入音频；DPO 跳过全部采样正确的音频。因此 SFT／DPO 的有效训练池可能不同，报告时同时给出池大小、偏好对数量与曝光数量。

2117 条真实训练池与 511 条开发反馈在音频、内容及已有文本／读音身份上隔离；TTS reference speech 来自真实训练池。511 条反复用于开发反馈，不再称作独立封存测试集。

## 审查保持简单

[transcribe_sg_review.py](/data/qwen3-asr/repo/dysarthric_yue_Speech/examples/transcribe_sg_review.py) 使用固定 seed42 R-only 审核 ASR，独立转写候选音频：Cantonese、空上下文、greedy、`num_beams=1`。ASR 不接收目标文本或 reference 文本，避免把答案作为转写提示。

[transcript_review.py](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/transcript_review.py) 再把 ASR 转写、目标文本、reference 文本这三段数据交给现有 `gpt-5.6-sol` 文字 Agent。一次 Responses 调用，`reasoning=low`、最多 1024 输出 token，结论仅为 `pass`／`obvious_failure` 和一句理由；三个文本都作为数据，不作为可执行指令。

Agent 仅根据文本迹象判断很明显的 reference 内容泄漏、严重截断、明显离题或机械长循环。ASR 可能转错，因此与目标不一致本身不构成失败。患者含糊发音、难识别和不确定的可疑现象不作为排除理由；没有 CER 淘汰阈值、评分体系、逐音素或逐字审核。正常 EOS 的空转写作为证据不足交给 Agent 保守通过；未正常结束或触及 token 上限属于上游转写技术异常，不触发 SG 质量重生成。

记录明确标注 `review_mode=asr_transcript`、`audio_inspected=False`、`transcript_inspected=True`，绑定转写原始记录、实际音频与固定审核 ASR 的哈希。文字 Agent 不能宣称听过录音；`pass` 不证明逐字保真或声学／临床属性。

有限 rulebook：初始 zero-shot；明显失败时用 `seed + 1` 重试 zero-shot；再次明显失败时用 `seed + 2` 生成 instruct2；第三次仍明显失败则排除。每次新音频都重新粗审。技术异常直接停止，不伪装成质量重试，也不改变患者 reference。

每条 assignment 保留最终音频、各次尝试、独立转写、审查结论与准入状态；存在 WAV 文件不等于允许训练。旧 [audio_review.py](/data/qwen3-asr/repo/dysarthric_yue_Speech/src/therapist_harness/audio_review.py) 的直接音频审查及能力 probe 保留在 `--review-mode audio` 下，只有该显式模式需要可用的音频模型。

| 角色 | 固定／变化 | 用途 |
|---|---|---|
| TTS reference speech | 训练池内的患者 reference 分配 | SG 声音条件，是一段音频 |
| 审核 ASR | 固定 seed42 起点 R-only | 一次 greedy 转写，提供粗审文本证据 |
| 当前 ASR | 随 MAS-DPO parent 跨轮更新 | 每音频 4 次随机采样，产生实际 rejected |
| DPO reference model | 固定起点 R-only | 计算 DPO reference log probability |

默认审核 ASR 与 DPO reference 使用同一 R-only 起点，但角色和记录分开；当前 ASR 的 4 次随机采样不能冒充固定审核转写。

## 偏好对与训练

### 执行优化 v2

`prepare --dual-gpu` 将 `execution_version=persistent_dual_gpu_v2` 与两个 DPO 分支写入冻结协议。训练池、生成预算、SG seed/reference、解码及损失参数、Agent prompt、rulebook 均沿用原配置。

- GPU 0：SG 与独立常驻审核 ASR、三文本 Agent；随后只采样合成语音。
- GPU 1：本轮共同 parent 对真实候选采样，构造真实偏好对，直接推进 R/DPO 训练和评测，无需等待 SG。
- 两侧偏好数据就绪后合并，GPU 0 运行 R＋MAS TTS/DPO。它复用已采样的真实偏好对；真实曝光流与 R/DPO 一致。两个分支共享本轮 parent，下一轮仍仅由 MAS-DPO 推进。

`transcribe_sg_review.py --worker` 通过 JSON-lines 请求／响应队列连续处理音频；每次请求只有绝对音频路径与输出记录路径。独立进程避开 CosyVoice 的 Transformers overlay，启动时校验模型、加载一次，结束时复核 base 与 adapter 哈希。每条转写绑定音频哈希及 worker session，只有 worker 正常关闭且模型文件未变，整批审查才能进入采样。单次 greedy、Cantonese、空上下文及 EOS／截断规则不变，单音频旧模式仍保留。

真实与合成侧分别保留 `sampling_real/`、`sampling_synthetic/` 下的实际采样配置、回执、候选和偏好对。合并回执标为 `current_asr_sampling_merged_v2`，引用两个原始作业及哈希，不冒充一次整体采样；注册时重新验证共同 parent、base model、全部解码参数、随机种子、样本身份／顺序和源文件哈希。

冻结旧运行不能改写 `sources.json` 后继续。迁移在新目录建立新协议与新账本，保持原 experiment run_id、sample_id、proposal、文本、reference、seed 和顺序。评测计划按新协议重新编译，并核对唯一变化为协议哈希。已完成 assignment 在核验原始音频、独立转写和审查后复用，并记录来源；未完成尝试不准入。

`execution_timeline.jsonl` 记录真实采样／训练／评测、SG、合成采样、合并等待及混合分支阶段。生成回执分别记录 worker 冷启动、SG 加载、逐次 TTS、纯 ASR 转写、请求耗时和 Agent 时间。吞吐报告区分稳定逐条处理与含冷启动的整批平均；整轮对照固定同一 proposal，实际测量从已确定处方到两个 DPO 分支评测完成，MAS 与 Reflection API 时间另报。

双卡在线 smoke 和新实验分别使用 `smoke <新目录> --dual-gpu`、`prepare <新目录> --dual-gpu --rounds 2`。已经停止的旧运行使用以下迁移命令，再对新目录执行 `run`；迁移本身只核验、准备产物，不启动生成或训练。

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python \
  /data/qwen3-asr/repo/dysarthric_yue_Speech/examples/mas_asr_dpo_loop.py \
  migrate /data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo96_seed42_20260908_exec_v2 \
  --from-root /data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo96_seed42_20260908_v1
```

上述两个目录是本次已存在的历史／迁移产物，不应重复初始化或覆盖。正式迁移预检确认全部 2117 条真实候选身份及音频哈希唯一，且与 511 条开发反馈隔离。

2026-09-08 的优化验证包括 244 项组件／契约检查、4 条相同音频的旧进程／常驻 worker 精确转写对照，以及实际 6 条新音频＋8 条真实候选的双卡在线闭环。在线 smoke 已完成两个 DPO 更新、两个分支各 511 条评测、真实 Reflection 和下一轮 proposal。相同处方串行复跑与新流程的音频、56 次候选、20 个偏好对、最终模型参数及评测预测一致。性能结论和网络故障耗时单独保留在 [验证报告](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_execution_v2_validation_20260908_v1/report.zh.md)，不将小规模速度外推为正式全池耗时或方法效果。

[sample_asr_preferences.py](/data/qwen3-asr/repo/dysarthric_yue_Speech/examples/sample_asr_preferences.py) 对每条准入音频用当前 parent 随机采样 4 次，保存原始输出、转录、token、随机种子和 parent 哈希。

- 同一音频、同一空上下文和强制 Cantonese 条件对应 chosen／rejected；真实 chosen 为训练标签，合成 chosen 为粗审通过的目标文本。
- rejected 仅来自当前 ASR 实际错误转录。按现有文本归一化判定相同／正确并去重，但保存原始 chosen／rejected。
- 全部正确则不构造 DPO 对；4 次都是同一个错误仍保留 1 对。实际空转录可以作为 rejected，不人为编造负样本。
- 一个音频的不同错误对权重合计为 1，避免错误种类更多的音频无意增加整体训练剂量。

[qwen3_asr_audio_dpo.py](/data/qwen3-asr/repo/dysarthric_yue_Speech/finetune/scripts/qwen3_asr_audio_dpo.py) 提供共用的音频 SFT／DPO 入口。DPO reference model 始终是**冻结的起点 R-only ASR**；它与 SG 的 reference speech 是两个不同概念，后续 parent 更新不会移动 DPO reference。

训练使用标准 DPO 的 policy/reference chosen-minus-rejected log-probability 差，仅计算转录 token 与 EOS 的总 log probability；音频、prompt 和强制语言前缀不参与监督。这里的 SFT 控制也采用一致的前缀 mask，因此历史旧 SFT 运行不能直接充当本轮配对控制。

默认准备参数：seed42，96 条合成音频＝6 个新文本各 16 个 reference 分配，每音频采样 4 次；12 个更新步，每步 32 个 R 音频曝光＋8 个 S 音频曝光，学习率 `1e-5`，DPO `beta=0.1`，S 权重 `lambda=0.05`。这表示 2117 条是真实候选池，12 步不保证每条真实音频都获得训练曝光。实际合成准入和偏好对可能少于请求数量；不自动补造负样本。

## 使用

工作目录为 `/data/qwen3-asr/repo/dysarthric_yue_Speech`，Python 为 `/data/qwen3-asr/venvs/qwen3-asr/bin/python`。默认 `asr_transcript` 复用现有 MAS 的文字模型、endpoint 和密钥环境变量，**无需额外音频模型参数**。凭证通过运行环境安全注入，不写在命令、协议或文档中。实际 `run/smoke` 需要凭据时，交互式终端可通过 `getpass` 输入缺失变量；非交互执行应预先注入。`prepare` 不需要凭据。

下面使用默认转写粗审，MAS 服务密钥已按协议注入 `OPENAI_API_KEY`。需要直接听音时才显式指定 `--review-mode audio` 以及 `--audio-review-model`、`--audio-review-base-url` 和对应密钥环境变量名；该模式仍须通过实际音频能力 probe。

完整在线 smoke 使用独立新目录：6 条新音频、8 条真实候选、每条采样 4 次、每个可更新分支 1 步；随后做真实评测、Reflection，并生成第二轮 proposal，不做第二轮更新。

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python \
  /data/qwen3-asr/repo/dysarthric_yue_Speech/examples/mas_asr_dpo_loop.py \
  smoke /data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_transcript_live_smoke_v1
```

正式规模先 `prepare`，只冻结配置和输入，不生成、不训练：

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python \
  /data/qwen3-asr/repo/dysarthric_yue_Speech/examples/mas_asr_dpo_loop.py \
  prepare /data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_asr_dpo_v1
```

完成在线 smoke 后，`run <root>` 执行已准备的协议，`status <root>` 查看阶段。还支持 `assess`、`prescribe`、`generate`、`sample`、`train --arm ...`、`evaluate --arm ...`、`reassess`、`reflect`，通过 `--round` 指定轮次。来源代码改变时应准备新运行，不覆盖冻结运行的记录。

仅运行两个 DPO 分支的正式配置：

```bash
/data/qwen3-asr/venvs/qwen3-asr/bin/python \
  /data/qwen3-asr/repo/dysarthric_yue_Speech/examples/mas_asr_dpo_loop.py \
  prepare /data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo96_seed42_20260908_v1 \
  --dpo-only --rounds 2

/data/qwen3-asr/venvs/qwen3-asr/bin/python \
  /data/qwen3-asr/repo/dysarthric_yue_Speech/examples/mas_asr_dpo_loop.py \
  run /data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo96_seed42_20260908_v1
```

## 可追溯记录与 smoke 证据

`PreferenceLoopConfig`、`TrainingReceipt.preference_delivery` 和 `EpisodeMemory.preference_delivery` 为可选扩展；未启用时不增加旧记录的空字段，保留旧序列化哈希。每轮记录生成请求、粗审模式与失败、实际转写／候选、偏好对、冻结 reference cache、真实训练曝光以及四分支评测。注册时回读转写记录和音频，核对固定审核 ASR 与 greedy 配置，重新构造偏好对和曝光序列，阻止其他 parent 的采样、移动 reference、未审查样本或篡改权重进入账本。

Reflection 及下一轮 EvidencePack 同时包含真实效果与实际交付。生成 carrier 数量仍表示生成内容覆盖；DPO 剂量读取 `preference_delivery`，不能将全正确而没有偏好对、或粗审失败而未训练的内容解释成学习假设已被否定。

当前转写粗审运行目录：[mas_dpo_transcript_smoke_20260908_v1](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_transcript_smoke_20260908_v1)。实际 [component_transcription.json](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_transcript_smoke_20260908_v1/component_transcription.json) 的 greedy 转写为“你遲啲記得帶齊申請文件”；[component_review.json](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_transcript_smoke_20260908_v1/component_review.json) 记录真实 `gpt-5.6-sol` 调用判为 `pass`，`audio_inspected=False`、`transcript_inspected=True`。新粗审组件与随后整轮在线闭环 smoke 均已完成；完整结果见本文顶部的本轮报告。

此前 GPU 组件记录保留在 [mas_dpo_code_smoke_20260908_v1](/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_dpo_code_smoke_20260908_v1)。

| 已实际执行 | 结果及范围 |
|---|---|
| 当前 R-only ASR 的 GPU 采样 | 8 条真实音频、32 次实际采样、24 个去重错误对；全部正常 EOS，无 token 上限截断 |
| SG 运行时 | 用一条既有 MAS assignment 生成 6.04 秒、24 kHz 音频；该技术运行当时未粗审、准入为 0；后续转写粗审单独记录，不改写原记录 |
| R-DPO／R-SFT 更新 | 各实际更新 1 步；DPO 初始 loss 为 `0.693147`，policy 参数改变，冻结 reference 文件未改变 |
| 保存模型后评测 | 两个新 checkpoint 各评测相同的 12 条开发反馈，共 24 次新推理；独立重算 CER 一致，parent 的对应 12 条使用已有预测 |
| CPU 正确性与闭环接口 | 覆盖音频附件、简单 rulebook、去重／全正确跳过、mask／DPO 梯度、分支和隔离、冻结 reference／曝光绑定、Reflection → ledger → 下一轮反馈 |

CPU 闭环测试中的模拟评估明确标为 `counterfactual_interface_smoke`。组件检查不替代整轮在线闭环 smoke；默认转写模式不宣称直接听音能力，小样本反馈数值也不作为方法增益结论。
